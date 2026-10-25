from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class _SharedTripleAttnBase(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_pdrop: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head}).")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.q_shared = nn.Linear(d_model, d_model)
        self.k_shared = nn.Linear(d_model, d_model)
        self.v_shared = nn.Linear(d_model, d_model)
        self.proj_shared = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        return x.view(B, L, self.n_head, self.d_head).transpose(1, 2)

    def _qkv(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._shape(self.q_shared(h)),
            self._shape(self.k_shared(h)),
            self._shape(self.v_shared(h)),
        )

    @staticmethod
    def _apply_masks(
        score: torch.Tensor,
        causal_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        B: int,
        L: int,
    ) -> torch.Tensor:
        score = score.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:
            pad_mask = attention_mask.view(B, 1, 1, L)
            score = score.masked_fill(pad_mask == 0, float("-inf"))
        return score


class TripleAttn_SharedRoute(_SharedTripleAttnBase):

    def __init__(self, d_model: int, n_head: int, attn_pdrop: float):
        super().__init__(d_model, n_head, attn_pdrop)
        init = torch.zeros(3, 3, n_head)
        init[0, 0] = 1.0
        init[1, 1] = 1.0
        init[2, 2] = 1.0
        self.alpha = nn.Parameter(init)

    def forward(self, h_sh, h_s, h_c, attention_mask=None):
        B, L, _ = h_sh.size()
        causal_mask = torch.tril(torch.ones(L, L, device=h_sh.device)).view(1, 1, L, L)

        q_sh, k_sh, v_sh = self._qkv(h_sh)
        q_se, k_se, v_se = self._qkv(h_s)
        q_cf, k_cf, v_cf = self._qkv(h_c)

        scale = math.sqrt(self.d_head)
        score_sh = torch.matmul(q_sh, k_sh.transpose(-2, -1)) / scale
        score_se = torch.matmul(q_se, k_se.transpose(-2, -1)) / scale
        score_cf = torch.matmul(q_cf, k_cf.transpose(-2, -1)) / scale
        scores = [score_sh, score_se, score_cf]
        values = [v_sh, v_se, v_cf]

        outs = []
        for i in range(3):
            mixed = (
                self.alpha[i, 0].view(1, -1, 1, 1) * scores[0]
                + self.alpha[i, 1].view(1, -1, 1, 1) * scores[1]
                + self.alpha[i, 2].view(1, -1, 1, 1) * scores[2]
            )
            mixed = self._apply_masks(mixed, causal_mask, attention_mask, B, L)
            attn = self.attn_drop(F.softmax(mixed, dim=-1))
            out = torch.matmul(attn, values[i]).transpose(1, 2).contiguous().view(B, L, -1)
            outs.append(self.proj_shared(out))
        return outs[0], outs[1], outs[2]


class TripleAttn_SharedRouteFixedDiag(_SharedTripleAttnBase):
    """Same score routing as SharedRoute, but diagonal α[i,i] is fixed to 1."""

    def __init__(self, d_model: int, n_head: int, attn_pdrop: float):
        super().__init__(d_model, n_head, attn_pdrop)
        self.alpha = nn.Parameter(torch.zeros(3, 3, n_head))
        diag = torch.eye(3, dtype=torch.bool).unsqueeze(-1).expand(3, 3, n_head).contiguous()
        self.register_buffer("alpha_diag_mask", diag)

    def effective_alpha(self) -> torch.Tensor:
        ones = torch.ones_like(self.alpha)
        return torch.where(self.alpha_diag_mask, ones, self.alpha)

    def forward(self, h_sh, h_s, h_c, attention_mask=None):
        B, L, _ = h_sh.size()
        causal_mask = torch.tril(torch.ones(L, L, device=h_sh.device)).view(1, 1, L, L)

        q_sh, k_sh, v_sh = self._qkv(h_sh)
        q_se, k_se, v_se = self._qkv(h_s)
        q_cf, k_cf, v_cf = self._qkv(h_c)

        scale = math.sqrt(self.d_head)
        score_sh = torch.matmul(q_sh, k_sh.transpose(-2, -1)) / scale
        score_se = torch.matmul(q_se, k_se.transpose(-2, -1)) / scale
        score_cf = torch.matmul(q_cf, k_cf.transpose(-2, -1)) / scale
        scores = [score_sh, score_se, score_cf]
        values = [v_sh, v_se, v_cf]
        alpha = self.effective_alpha()

        outs = []
        for i in range(3):
            mixed = (
                alpha[i, 0].view(1, -1, 1, 1) * scores[0]
                + alpha[i, 1].view(1, -1, 1, 1) * scores[1]
                + alpha[i, 2].view(1, -1, 1, 1) * scores[2]
            )
            mixed = self._apply_masks(mixed, causal_mask, attention_mask, B, L)
            attn = self.attn_drop(F.softmax(mixed, dim=-1))
            out = torch.matmul(attn, values[i]).transpose(1, 2).contiguous().view(B, L, -1)
            outs.append(self.proj_shared(out))
        return outs[0], outs[1], outs[2]


class SharedTripleTransformerBlock(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["n_embd"]
        eps = config["layer_norm_epsilon"]
        n_inner = config["n_inner"] if config.get("n_inner") is not None else 4 * d_model
        resid_p = config["resid_pdrop"]

        self.ln_1 = nn.LayerNorm(d_model, eps=eps)
        self.ln_2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, n_inner),
            nn.GELU(),
            nn.Linear(n_inner, d_model),
            nn.Dropout(resid_p),
        )

        attn_type = (config.get("triple_attn_type") or "shared_route").strip().lower()
        attn_pdrop = config["attn_pdrop"]
        n_head = config["n_head"]
        if attn_type == "shared_route":
            self.attn = TripleAttn_SharedRoute(d_model, n_head, attn_pdrop)
        elif attn_type in ("shared_route_fixed_diag", "route_fixed_diag", "fixed_diag"):
            self.attn = TripleAttn_SharedRouteFixedDiag(d_model, n_head, attn_pdrop)
        else:
            raise ValueError(
                f"Unknown triple_attn_type={attn_type!r}. "
                f"Available: shared_route | shared_route_fixed_diag"
            )
        print(f"[SCOPE] triple_attn_type={attn_type}")

    def forward(self, h_sh, h_s, h_c, attention_mask=None):
        h_sh_n = self.ln_1(h_sh)
        h_s_n = self.ln_1(h_s)
        h_c_n = self.ln_1(h_c)
        a_sh, a_s, a_c = self.attn(h_sh_n, h_s_n, h_c_n, attention_mask)
        h_sh = h_sh + a_sh
        h_s = h_s + a_s
        h_c = h_c + a_c

        h_sh_n2 = self.ln_2(h_sh)
        h_s_n2 = self.ln_2(h_s)
        h_c_n2 = self.ln_2(h_c)
        h_sh = h_sh + self.mlp(h_sh_n2)
        h_s = h_s + self.mlp(h_s_n2)
        h_c = h_c + self.mlp(h_c_n2)
        return h_sh, h_s, h_c


class SCOPE(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset, tokenizer: AbstractTokenizer):
        super().__init__(config, dataset, tokenizer)

        self.n_digit_total = tokenizer.n_digit
        self.n_digit_sh = config.get("n_digit_sh", 8)
        self.n_digit_sem = config.get("n_digit_sem", 16)
        self.n_digit_cf = config.get(
            "n_digit_cf", self.n_digit_total - self.n_digit_sh - self.n_digit_sem
        )
        assert self.n_digit_sh + self.n_digit_sem + self.n_digit_cf == self.n_digit_total, (
            f"n_digit_sh ({self.n_digit_sh}) + n_digit_sem ({self.n_digit_sem}) + "
            f"n_digit_cf ({self.n_digit_cf}) != n_digit_total ({self.n_digit_total})"
        )
        self.codebook_size = config.get("codebook_size", 256)
        n_embd = config["n_embd"]
        V_sh = self.n_digit_sh * self.codebook_size
        V_sem = self.n_digit_sem * self.codebook_size
        V_cf = self.n_digit_cf * self.codebook_size

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        self.freeze_sh = bool(config.get("freeze_sh", False))
        self.freeze_sem = bool(config.get("freeze_sem", False))
        initializer_range = config.get('initializer_range', 0.02)

        sh_stack = self._load_quantizer_levels(config, branch="sh")
        if sh_stack is None:
            d_sh_orig = int(config.get("d_latent", 128))
            print(f"[SCOPE] Fallback: random {d_sh_orig}d ShID codebook (no quantizer ckpt).")
            sh_stack = torch.randn(self.n_digit_sh, self.codebook_size, d_sh_orig) * initializer_range
        d_sh_orig = sh_stack.shape[-1]

        if self.freeze_sh:
            self.wte_sh_quant = nn.Embedding(V_sh, d_sh_orig)
            self.wte_sh_quant.weight.data.copy_(sh_stack.reshape(V_sh, d_sh_orig))
            self.wte_sh_quant.weight.requires_grad = False
            self.sh_proj = nn.Linear(d_sh_orig, n_embd, bias=False)
            nn.init.orthogonal_(self.sh_proj.weight)
            self.wte_sh = None
        else:
            self.wte_sh = nn.Embedding(V_sh, n_embd)
            if d_sh_orig <= n_embd:
                padded = torch.zeros(self.n_digit_sh, self.codebook_size, n_embd)
                padded[:, :, :d_sh_orig] = sh_stack
            else:
                padded = sh_stack[..., :n_embd]
            self.wte_sh.weight.data.copy_(padded.reshape(V_sh, n_embd))
            self.wte_sh_quant = None
            self.sh_proj = None

        sem_stack = self._load_quantizer_levels(config, branch="sem")
        if sem_stack is None:
            print("[SCOPE] Fallback: random 128d SID codebook (no quantizer ckpt).")
            sem_stack = torch.randn(self.n_digit_sem, self.codebook_size, 128) * initializer_range
        d_sem_orig = sem_stack.shape[-1]

        if self.freeze_sem:
            self.wte_sem_quant = nn.Embedding(V_sem, d_sem_orig)
            self.wte_sem_quant.weight.data.copy_(sem_stack.reshape(V_sem, d_sem_orig))
            self.wte_sem_quant.weight.requires_grad = False
            self.sem_proj = nn.Linear(d_sem_orig, n_embd, bias=False)
            nn.init.orthogonal_(self.sem_proj.weight)
            self.wte_sem = None
        else:
            self.wte_sem = nn.Embedding(V_sem, n_embd)
            if d_sem_orig <= n_embd:
                padded = torch.zeros(self.n_digit_sem, self.codebook_size, n_embd)
                padded[:, :, :d_sem_orig] = sem_stack
            else:
                padded = sem_stack[..., :n_embd]
            self.wte_sem.weight.data.copy_(padded.reshape(V_sem, n_embd))
            self.wte_sem_quant = None
            self.sem_proj = None

        cf_stack = self._load_quantizer_levels(config, branch="cf")
        if cf_stack is None:
            print("[SCOPE] Fallback: random CID codebook (no quantizer ckpt). "
                  "Init: normal(std=initializer_range).")
            self.wte_cf = nn.Embedding(V_cf, n_embd)
            nn.init.normal_(self.wte_cf.weight, std=initializer_range)
            d_cf_orig_used = "random"
        else:
            d_cf_orig_used = cf_stack.shape[-1]
            self.wte_cf = nn.Embedding(V_cf, n_embd)
            if d_cf_orig_used <= n_embd:
                padded = torch.zeros(self.n_digit_cf, self.codebook_size, n_embd)
                padded[:, :, :d_cf_orig_used] = cf_stack
            else:
                padded = cf_stack[..., :n_embd]
            self.wte_cf.weight.data.copy_(padded.reshape(V_cf, n_embd))

        self.wpe = nn.Embedding(tokenizer.max_token_seq_len, n_embd)
        nn.init.normal_(self.wpe.weight, std=initializer_range)

        sh_mode = "frozen+adapter" if self.freeze_sh else "learnable (codebook-init)"
        sem_mode = "frozen+adapter" if self.freeze_sem else "learnable (codebook-init)"
        cf_mode = "learnable (codebook-init)" if cf_stack is not None else "learnable (random-init, fallback)"
        print(f"[SCOPE] ShID={sh_mode} (d={d_sh_orig}) | SID={sem_mode} (d={d_sem_orig}) | "
              f"CID={cf_mode} (d={d_cf_orig_used}) | n_embd={n_embd}")

        self.emb_drop = nn.Dropout(config['embd_pdrop'])

        self.blocks = nn.ModuleList([
            SharedTripleTransformerBlock(config) for _ in range(config['n_layer'])
        ])

        self.ln_f = nn.LayerNorm(n_embd, eps=config['layer_norm_epsilon'])

        self.pred_heads_sh = nn.ModuleList([ResBlock(n_embd) for _ in range(self.n_digit_sh)])
        self.pred_heads_sem = nn.ModuleList([ResBlock(n_embd) for _ in range(self.n_digit_sem)])
        self.pred_heads_cf = nn.ModuleList([ResBlock(n_embd) for _ in range(self.n_digit_cf)])

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.lambda_orth = config.get("lambda_orth_triple", config.get("lambda_orth", 0.1))
        self.downstream_orth_streams = (
            config.get("downstream_orth_streams") or "two"
        ).strip().lower()
        if self.downstream_orth_streams not in ("two", "all3"):
            raise ValueError(
                f"downstream_orth_streams must be 'two' or 'all3', "
                f"got {self.downstream_orth_streams!r}"
            )
        if self.downstream_orth_streams == "all3":
            print("[SCOPE] downstream_orth_streams=all3 "
                  "(sh⊥sem + sh⊥cf + sem⊥cf)")

        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        self.fusion_method = (config.get("fusion_method") or "mean_log_prob").strip().lower()
        if self.fusion_method not in ("mean_log_prob", "entropy_weighted"):
            raise ValueError(
                f"fusion_method must be 'mean_log_prob' or 'entropy_weighted', got {self.fusion_method!r}"
            )
        if self.fusion_method == "entropy_weighted":
            print("[SCOPE] generate() fusion_method=entropy_weighted (per-user softmax(-H) reweighting)")

        self.aux_head_s_to_c = ResBlock(n_embd)
        self.aux_head_c_to_s = ResBlock(n_embd)
        self.aux_head_sh_to_s = ResBlock(n_embd)
        self.aux_head_sh_to_c = ResBlock(n_embd)
        for p in (
            *self.aux_head_s_to_c.parameters(),
            *self.aux_head_c_to_s.parameters(),
            *self.aux_head_sh_to_s.parameters(),
            *self.aux_head_sh_to_c.parameters(),
        ):
            p.requires_grad_(False)
        self.aux_topk = int(config.get("aux_barrier_topk", 10))
        print(f"[SCOPE] aux barrier heads enabled (monitor-only, top-{self.aux_topk})")

    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.wte_cf.parameters() if p.requires_grad)
        if self.wte_sh is not None:
            emb_params += sum(p.numel() for p in self.wte_sh.parameters() if p.requires_grad)
        if self.wte_sem is not None:
            emb_params += sum(p.numel() for p in self.wte_sem.parameters() if p.requires_grad)
        return (
            f'#Learnable Embedding parameters: {emb_params}\n'
            f'#Total trainable parameters: {total_params}\n'
        )

    def _load_quantizer_levels(self, config: dict, branch: str) -> Optional[torch.Tensor]:
        ckpt_path = (config.get("quantizer_ckpt") or "").strip()
        meta_path = (config.get("quantizer_meta") or "").strip()
        variant = (config.get("triple_rqvae_variant") or "baseline").strip()

        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"[SCOPE] quantizer_ckpt not found: {ckpt_path}")
            return None

        ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        if variant != "baseline":
            raise ValueError(
                f"SCOPE release only supports triple_rqvae_variant='baseline', got {variant!r}"
            )
        from tokenizer_triple.triple_rqvae import TripleQuantizer
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            d_sem_orig = int(meta["d_sem"])
            cf_meta_key = config.get("cf_meta_key", "d_cf")
            if cf_meta_key not in meta:
                raise KeyError(
                    f"[SCOPE] meta missing field '{cf_meta_key}'; available: "
                    f"{[k for k in meta if k.startswith('d_cf')]}"
                )
            d_cf_orig = int(meta[cf_meta_key])
        else:
            print(f"[SCOPE] quantizer_meta not found: {meta_path}. Infer dimensions from checkpoint.")
            d_sem_orig = int(state["rqvae_sem.blocks.0.vq.embed.weight"].shape[1])
            d_cf_orig = int(state["rqvae_cf.blocks.0.vq.embed.weight"].shape[1])

        kwargs = {
            "d_sem": d_sem_orig,
            "d_cf": d_cf_orig,
            "n_codes_sh": self.codebook_size,
            "n_codes_sem": self.codebook_size,
            "n_codes_cf": self.codebook_size,
            "n_levels_sh": self.n_digit_sh,
            "n_levels_sem": self.n_digit_sem,
            "n_levels_cf": self.n_digit_cf,
            "hidden_sh": int(config.get("hidden_sh", 256)),
            "d_latent": int(config.get("d_latent", 128)),
            "lambda_orth": float(config.get("lambda_orth", 0.1)),
        }
        q = TripleQuantizer(**kwargs)
        q.load_legacy_compatible_state_dict(state)

        if branch == "sh":
            n_lv = self.n_digit_sh
            blocks = q.rqvae_sh.blocks
        elif branch == "sem":
            n_lv = self.n_digit_sem
            blocks = q.rqvae_sem.blocks
        elif branch == "cf":
            n_lv = self.n_digit_cf
            blocks = q.rqvae_cf.blocks
        else:
            raise ValueError(branch)

        levels = [blocks[lv].vq.embed.weight.data for lv in range(n_lv)]
        return torch.stack(levels, dim=0)

    def _split_global_tokens(self, item_tokens: torch.Tensor):
        B, L, _ = item_tokens.size()
        device = item_tokens.device
        cs = self.codebook_size

        tokens_sh = torch.zeros((B, L, self.n_digit_sh), dtype=torch.long, device=device)
        tokens_sem = torch.zeros((B, L, self.n_digit_sem), dtype=torch.long, device=device)
        tokens_cf = torch.zeros((B, L, self.n_digit_cf), dtype=torch.long, device=device)

        for i in range(self.n_digit_sh):
            t = item_tokens[..., i]
            valid = t > 0
            idx = t - 1
            tokens_sh[..., i] = torch.where(valid, idx, torch.zeros_like(idx))

        sh_offset = self.n_digit_sh * cs
        for j in range(self.n_digit_sem):
            real = self.n_digit_sh + j
            t = item_tokens[..., real]
            valid = t > 0
            idx = t - 1 - sh_offset
            tokens_sem[..., j] = torch.where(valid, idx, torch.zeros_like(idx))

        sh_sem_offset = (self.n_digit_sh + self.n_digit_sem) * cs
        for k in range(self.n_digit_cf):
            real = self.n_digit_sh + self.n_digit_sem + k
            t = item_tokens[..., real]
            valid = t > 0
            idx = t - 1 - sh_sem_offset
            tokens_cf[..., k] = torch.where(valid, idx, torch.zeros_like(idx))

        return tokens_sh, tokens_sem, tokens_cf

    def _sh_embed_from_tokens(self, tokens_sh: torch.Tensor) -> torch.Tensor:
        if self.freeze_sh:
            h = self.wte_sh_quant(tokens_sh)
            h = self.sh_proj(h)
        else:
            h = self.wte_sh(tokens_sh)
        return h.mean(dim=-2)

    def _sem_embed_from_tokens(self, tokens_sem: torch.Tensor) -> torch.Tensor:
        if self.freeze_sem:
            h = self.wte_sem_quant(tokens_sem)
            h = self.sem_proj(h)
        else:
            h = self.wte_sem(tokens_sem)
        return h.mean(dim=-2)

    def _sh_logits_weight_chunked(self) -> torch.Tensor:
        if self.freeze_sh:
            raw = self.wte_sh_quant.weight.view(self.n_digit_sh, self.codebook_size, -1)
            return self.sh_proj(raw)
        return self.wte_sh.weight.view(self.n_digit_sh, self.codebook_size, -1)

    def _sem_logits_weight_chunked(self) -> torch.Tensor:
        if self.freeze_sem:
            raw = self.wte_sem_quant.weight.view(self.n_digit_sem, self.codebook_size, -1)
            return self.sem_proj(raw)
        return self.wte_sem.weight.view(self.n_digit_sem, self.codebook_size, -1)

    def _input_knockout_streams_active(self) -> set[str]:
        multi = getattr(self, '_input_knockout_streams', None)
        if multi is not None:
            return set(multi)
        one = getattr(self, '_input_knockout_stream', None)
        return {one} if one else set()

    def forward(self, batch: dict, return_loss=True):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        L = input_ids.size(1)

        item_tokens = self.item_id2tokens[input_ids]
        tokens_sh, tokens_sem, tokens_cf = self._split_global_tokens(item_tokens)

        E_sh = self._sh_embed_from_tokens(tokens_sh)
        E_sem = self._sem_embed_from_tokens(tokens_sem)
        E_cf = self.wte_cf(tokens_cf).mean(dim=-2)

        knock_streams = self._input_knockout_streams_active()
        if 'sh' in knock_streams:
            E_sh = torch.zeros_like(E_sh)
        if 'sem' in knock_streams:
            E_sem = torch.zeros_like(E_sem)
        if 'cf' in knock_streams:
            E_cf = torch.zeros_like(E_cf)

        pos = torch.arange(L, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_emb = self.wpe(pos)

        H_sh = self.emb_drop(E_sh + pos_emb)
        H_s = self.emb_drop(E_sem + pos_emb)
        H_c = self.emb_drop(E_cf + pos_emb)

        for block in self.blocks:
            H_sh, H_s, H_c = block(H_sh, H_s, H_c, attention_mask)

        H_sh = self.ln_f(H_sh)
        H_s = self.ln_f(H_s)
        H_c = self.ln_f(H_c)

        final_states_sh = torch.stack([head(H_sh) for head in self.pred_heads_sh], dim=-2)
        final_states_sem = torch.stack([head(H_s) for head in self.pred_heads_sem], dim=-2)
        final_states_cf = torch.stack([head(H_c) for head in self.pred_heads_cf], dim=-2)

        outputs = type('Outputs', (object,), {})()
        outputs.final_states_sh = final_states_sh
        outputs.final_states_sem = final_states_sem
        outputs.final_states_cf = final_states_cf
        outputs.H_sh_base = H_sh
        outputs.H_sem_base = H_s
        outputs.H_cf_base = H_c

        if return_loss:
            labels = batch['labels']
            label_mask = labels.view(-1) != -100
            n_embd = self.config['n_embd']

            sh_states = F.normalize(final_states_sh.view(-1, self.n_digit_sh, n_embd)[label_mask], dim=-1)
            s_states = F.normalize(final_states_sem.view(-1, self.n_digit_sem, n_embd)[label_mask], dim=-1)
            c_states = F.normalize(final_states_cf.view(-1, self.n_digit_cf, n_embd)[label_mask], dim=-1)

            w_sh = F.normalize(self._sh_logits_weight_chunked(), dim=-1)
            w_sem = F.normalize(self._sem_logits_weight_chunked(), dim=-1)
            w_cf = F.normalize(self.wte_cf.weight.view(self.n_digit_cf, self.codebook_size, -1), dim=-1)

            valid_labels = labels.view(-1)[label_mask]
            target_tokens = self.item_id2tokens[valid_labels]

            cs = self.codebook_size

            loss_sh = 0.0
            for i in range(self.n_digit_sh):
                logits = (sh_states[:, i, :] @ w_sh[i].T) / self.temperature
                targets = target_tokens[:, i] - i * cs - 1
                loss_sh = loss_sh + self.loss_fct(logits, targets)
            loss_sh = loss_sh / self.n_digit_sh

            loss_sem = 0.0
            for j in range(self.n_digit_sem):
                real = self.n_digit_sh + j
                logits = (s_states[:, j, :] @ w_sem[j].T) / self.temperature
                targets = target_tokens[:, real] - real * cs - 1
                loss_sem = loss_sem + self.loss_fct(logits, targets)
            loss_sem = loss_sem / self.n_digit_sem

            loss_cf = 0.0
            for k in range(self.n_digit_cf):
                real = self.n_digit_sh + self.n_digit_sem + k
                logits = (c_states[:, k, :] @ w_cf[k].T) / self.temperature
                targets = target_tokens[:, real] - real * cs - 1
                loss_cf = loss_cf + self.loss_fct(logits, targets)
            loss_cf = loss_cf / self.n_digit_cf

            loss_orth, aux_loss = self._downstream_orth_loss(outputs, label_mask, n_embd)

            loss_mtp = (
                loss_sh * self.n_digit_sh
                + loss_sem * self.n_digit_sem
                + loss_cf * self.n_digit_cf
            ) / self.n_digit_total

            outputs.loss = loss_mtp + self.lambda_orth * loss_orth + aux_loss
            outputs.loss_sh = loss_sh
            outputs.loss_sem = loss_sem
            outputs.loss_cf = loss_cf
            outputs.loss_orth = loss_orth

            with torch.no_grad():
                aux_h_s = self.aux_head_s_to_c(outputs.H_sem_base)
                aux_h_c = self.aux_head_c_to_s(outputs.H_cf_base)
                aux_h_sh_s = self.aux_head_sh_to_s(outputs.H_sh_base)
                aux_h_sh_c = self.aux_head_sh_to_c(outputs.H_sh_base)

                aux_h_s = F.normalize(aux_h_s.view(-1, n_embd)[label_mask], dim=-1)
                aux_h_c = F.normalize(aux_h_c.view(-1, n_embd)[label_mask], dim=-1)
                aux_h_sh_s_flat = F.normalize(aux_h_sh_s.view(-1, n_embd)[label_mask], dim=-1)
                aux_h_sh_c_flat = F.normalize(aux_h_sh_c.view(-1, n_embd)[label_mask], dim=-1)

                w_sem_l0 = w_sem[0]
                w_cf_l0 = w_cf[0]
                target_sid_l0 = target_tokens[:, self.n_digit_sh] - self.n_digit_sh * cs - 1
                target_cid_l0 = target_tokens[:, self.n_digit_sh + self.n_digit_sem] \
                    - (self.n_digit_sh + self.n_digit_sem) * cs - 1

                k = max(1, min(self.aux_topk, self.codebook_size))

                def _topk_acc(logits, target):
                    topk = logits.topk(k, dim=-1).indices
                    return (topk == target.view(-1, 1)).any(dim=-1).float().mean()

                logits_s_to_c = (aux_h_s @ w_cf_l0.T) / self.temperature
                logits_c_to_s = (aux_h_c @ w_sem_l0.T) / self.temperature
                logits_sh_to_s = (aux_h_sh_s_flat @ w_sem_l0.T) / self.temperature
                logits_sh_to_c = (aux_h_sh_c_flat @ w_cf_l0.T) / self.temperature

                outputs.aux_acc_s_to_c = _topk_acc(logits_s_to_c, target_cid_l0)
                outputs.aux_acc_c_to_s = _topk_acc(logits_c_to_s, target_sid_l0)
                outputs.aux_acc_sh_to_s = _topk_acc(logits_sh_to_s, target_sid_l0)
                outputs.aux_acc_sh_to_c = _topk_acc(logits_sh_to_c, target_cid_l0)
                outputs.aux_topk = k

        return outputs

    def _downstream_orth_loss(self, outputs, label_mask, n_embd):
        h_s_b = outputs.H_sem_base.view(-1, n_embd)[label_mask]
        h_c_b = outputs.H_cf_base.view(-1, n_embd)[label_mask]
        n_s = F.normalize(h_s_b, p=2, dim=-1)
        n_c = F.normalize(h_c_b, p=2, dim=-1)
        loss_orth_sem_cf = (n_s * n_c).sum(dim=-1).pow(2).mean()
        if self.downstream_orth_streams == "all3":
            h_sh_b = outputs.H_sh_base.view(-1, n_embd)[label_mask]
            n_sh = F.normalize(h_sh_b, p=2, dim=-1)
            loss_orth_sh_sem = (n_sh * n_s).sum(dim=-1).pow(2).mean()
            loss_orth_sh_cf = (n_sh * n_c).sum(dim=-1).pow(2).mean()
            loss_orth = (loss_orth_sh_sem + loss_orth_sh_cf + loss_orth_sem_cf) / 3.0
        else:
            loss_orth = loss_orth_sem_cf
        return loss_orth, loss_orth.new_zeros(())

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        seq_lens = batch['seq_lens'] - 1
        batch_idx = torch.arange(batch['input_ids'].size(0), device=batch['input_ids'].device)

        sh_states = F.normalize(outputs.final_states_sh[batch_idx, seq_lens], dim=-1)
        s_states = F.normalize(outputs.final_states_sem[batch_idx, seq_lens], dim=-1)
        c_states = F.normalize(outputs.final_states_cf[batch_idx, seq_lens], dim=-1)

        w_sh = F.normalize(self._sh_logits_weight_chunked(), dim=-1)
        w_sem = F.normalize(self._sem_logits_weight_chunked(), dim=-1)
        w_cf = F.normalize(self.wte_cf.weight.view(self.n_digit_cf, self.codebook_size, -1), dim=-1)

        logits_sh = [(sh_states[:, i, :] @ w_sh[i].T) / self.temperature for i in range(self.n_digit_sh)]
        logits_sem = [(s_states[:, j, :] @ w_sem[j].T) / self.temperature for j in range(self.n_digit_sem)]
        logits_cf = [(c_states[:, k, :] @ w_cf[k].T) / self.temperature for k in range(self.n_digit_cf)]

        log_sh = [F.log_softmax(l, dim=-1) for l in logits_sh]
        log_sem = [F.log_softmax(l, dim=-1) for l in logits_sem]
        log_cf = [F.log_softmax(l, dim=-1) for l in logits_cf]

        n_sh, n_sem, n_cf = self.n_digit_sh, self.n_digit_sem, self.n_digit_cf
        n_total = n_sh + n_sem + n_cf

        if self.fusion_method == "entropy_weighted":
            def _stream_mean_entropy(logs):
                H = torch.stack([(-(l.exp() * l).sum(dim=-1)) for l in logs], dim=0)
                return H.mean(dim=0)

            H_sh = _stream_mean_entropy(log_sh)
            H_sem = _stream_mean_entropy(log_sem)
            H_cf = _stream_mean_entropy(log_cf)
            w = F.softmax(-torch.stack([H_sh, H_sem, H_cf], dim=-1), dim=-1)

            scale_sh = (w[:, 0] * (n_total / max(n_sh, 1))).unsqueeze(-1)
            scale_sem = (w[:, 1] * (n_total / max(n_sem, 1))).unsqueeze(-1)
            scale_cf = (w[:, 2] * (n_total / max(n_cf, 1))).unsqueeze(-1)
            log_sh = [l * scale_sh for l in log_sh]
            log_sem = [l * scale_sem for l in log_sem]
            log_cf = [l * scale_cf for l in log_cf]

        logits_all = log_sh + log_sem + log_cf
        token_logits = torch.cat(logits_all, dim=-1)

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            return self.graph_propagation(token_logits, n_return_sequences)

        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        return preds.unsqueeze(-1)

    def build_ii_sim_mat(self):
        n_items = self.dataset.n_items
        cs = self.codebook_size
        n_total = self.n_digit_total
        n_embd = self.config['n_embd']

        sh_flat = self._sh_logits_weight_chunked().reshape(self.n_digit_sh * cs, n_embd)
        sem_flat = self._sem_logits_weight_chunked().reshape(self.n_digit_sem * cs, n_embd)
        cf_flat = self.wte_cf.weight
        full_weight = torch.cat([sh_flat, sem_flat, cf_flat], dim=0)

        token_embs = full_weight.view(n_total, cs, n_embd)
        token_embs = F.normalize(token_embs, dim=-1)

        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        item_item_sim = torch.zeros((n_items, n_items), device=token_embs.device, dtype=torch.float32)

        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]

                bi = i_end - i_start
                bj = j_end - j_start
                sum_block = torch.zeros((bi, bj), device=token_embs.device, dtype=torch.float32)

                for p in range(n_total):
                    row_inds = tokens_i[:, p] - p * cs - 1
                    col_inds = tokens_j[:, p] - p * cs - 1
                    temp = token_sims_01[p].index_select(0, row_inds)
                    temp = temp.index_select(1, col_inds)
                    sum_block += temp

                avg_block = sum_block / n_total
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building full item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]
        visited_nodes = {i: set() for i in range(batch_size)}

        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items, (batch_size, self.num_beams), dtype=torch.long, device=token_logits.device
        )
        for b in range(batch_size):
            visited_nodes[b].update(topk_nodes_sorted[b].cpu().tolist())

        for _ in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
            next_nodes = []
            for b in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[b])
                visited_nodes[b].update(neighbors_in_batch.cpu().tolist())
                scores = torch.gather(
                    token_logits[b].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    -1,
                    (self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        visited_counts = torch.FloatTensor([[len(visited_nodes[b])] for b in range(batch_size)])
        return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1), visited_counts
