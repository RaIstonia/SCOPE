from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer_triple.rqvae import VQLayer, RQVAEBlock, RQVAE  # noqa: F401

class TripleQuantizer(nn.Module):
    def __init__(
        self,
        d_sem: int,
        d_cf: int,
        n_codes_sh: int = 256,
        n_codes_sem: int = 256,
        n_codes_cf: int = 256,
        n_levels_sh: int = 8,
        n_levels_sem: int = 16,
        n_levels_cf: int = 8,
        hidden_sh: int = 256,
        d_latent: int = 128,
        lambda_orth: float = 0.1,
        lambda_orth_sh_sem: float = 1.0,
        lambda_orth_sh_cf: float = 1.0,
        lambda_orth_priv: float = 1.0,
        lambda_vq_sh: float = 1.0,
        lambda_vq_sem: float = 1.0,
        lambda_vq_cf: float = 1.0,
    ):
        super().__init__()
        self.d_sem = d_sem
        self.d_cf = d_cf
        self.d_latent = d_latent
        self.lambda_orth = lambda_orth
        self.lambda_orth_sh_sem = lambda_orth_sh_sem
        self.lambda_orth_sh_cf = lambda_orth_sh_cf
        self.lambda_orth_priv = lambda_orth_priv
        self.lambda_vq_sh = lambda_vq_sh
        self.lambda_vq_sem = lambda_vq_sem
        self.lambda_vq_cf = lambda_vq_cf

        # ============================================================

        # ============================================================
        self.proj_sem_to_hidden = nn.Linear(d_sem, hidden_sh)
        self.proj_cf_to_hidden = nn.Linear(d_cf, hidden_sh)
        self.enc_sh = nn.Sequential(
            nn.LayerNorm(2 * hidden_sh),
            nn.GELU(),
            nn.Linear(2 * hidden_sh, hidden_sh),
            nn.LayerNorm(hidden_sh),
            nn.GELU(),
            nn.Linear(hidden_sh, d_latent),
        )
        self.rqvae_sh = RQVAE(d_latent, n_codes_sh, n_levels_sh)

        self.dec_sh_to_sem = nn.Sequential(
            nn.Linear(d_latent, hidden_sh),
            nn.LayerNorm(hidden_sh),
            nn.GELU(),
            nn.Linear(hidden_sh, d_sem),
        )
        self.dec_sh_to_cf = nn.Sequential(
            nn.Linear(d_latent, hidden_sh),
            nn.LayerNorm(hidden_sh),
            nn.GELU(),
            nn.Linear(hidden_sh, d_cf),
        )

        # ============================================================

        # ============================================================
        self.rqvae_sem = RQVAE(d_sem, n_codes_sem, n_levels_sem)

        # ============================================================
        # 3) Collaborative private branch
        # ============================================================
        self.rqvae_cf = RQVAE(d_cf, n_codes_cf, n_levels_cf)

        self.W1 = nn.Linear(d_sem, d_cf, bias=False)

        self.n_levels_sh = n_levels_sh
        self.n_levels_sem = n_levels_sem
        self.n_levels_cf = n_levels_cf

    _LEGACY_STATE_PREFIXES = (
        "enc_cf.",
        "enc_sh_from_sem.",
        "enc_sh_from_cf.",
    )

    def load_legacy_compatible_state_dict(self, state_dict):

        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(self._LEGACY_STATE_PREFIXES)
        }
        return self.load_state_dict(filtered_state, strict=True)

    @staticmethod
    def _cos2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_n = F.normalize(a, p=2, dim=-1)
        b_n = F.normalize(b, p=2, dim=-1)
        return (a_n * b_n).sum(dim=-1).pow(2).mean()

    def forward(self, v_sem: torch.Tensor, v_cf: torch.Tensor) -> dict:
        # ============================================================

        # ============================================================
        h_sem = self.proj_sem_to_hidden(v_sem)
        h_cf = self.proj_cf_to_hidden(v_cf)
        z_sh_in = self.enc_sh(torch.cat([h_sem, h_cf], dim=-1))
        z_sh, sh_indices, vq_loss_sh = self.rqvae_sh(z_sh_in)

        v_sem_sh = self.dec_sh_to_sem(z_sh)
        v_cf_sh = self.dec_sh_to_cf(z_sh)

        # ============================================================
        # Stage 2: private semantic on residual
        # ============================================================
        v_sem_priv_in = v_sem - v_sem_sh
        v_sem_priv_recon, sid_indices, vq_loss_sem = self.rqvae_sem(v_sem_priv_in)
        v_sem_recon = v_sem_sh + v_sem_priv_recon

        # ============================================================
        # Stage 3: private collaborative on residual
        # ============================================================
        v_cf_priv_in = v_cf - v_cf_sh
        z_cf_priv, cid_indices, vq_loss_cf = self.rqvae_cf(v_cf_priv_in)

        # ============================================================

        # ============================================================
        v_cf_hat = v_cf_sh + z_cf_priv

        # ============================================================
        # Losses
        # ============================================================
        loss_recon_sem = F.mse_loss(v_sem_recon, v_sem)
        loss_recon_cf = F.mse_loss(v_cf_hat, v_cf)

        v_sem_priv_in_cf = self.W1(v_sem_priv_recon)
        loss_orth_sh_sem = self._cos2(v_sem_sh, v_sem_priv_recon)
        loss_orth_sh_cf = self._cos2(v_cf_sh, z_cf_priv)
        loss_orth_priv = self._cos2(v_sem_priv_in_cf, z_cf_priv)
        loss_orth = (
            self.lambda_orth_sh_sem * loss_orth_sh_sem
            + self.lambda_orth_sh_cf * loss_orth_sh_cf
            + self.lambda_orth_priv * loss_orth_priv
        )

        loss_vq = vq_loss_sh + vq_loss_sem + vq_loss_cf
        loss_total = (
            loss_recon_sem
            + loss_recon_cf
            + self.lambda_orth * loss_orth
            + self.lambda_vq_sh * vq_loss_sh
            + self.lambda_vq_sem * vq_loss_sem
            + self.lambda_vq_cf * vq_loss_cf
        )

        with torch.no_grad():
            denom_sem = v_sem.norm(dim=-1).clamp(min=1e-6)
            denom_cf = v_cf.norm(dim=-1).clamp(min=1e-6)
            energy_sh_sem = (v_sem_sh.norm(dim=-1) / denom_sem).mean()
            energy_sh_cf = (v_cf_sh.norm(dim=-1) / denom_cf).mean()
        return {
            "z_sh": z_sh,
            "v_sem_sh": v_sem_sh,
            "v_cf_sh": v_cf_sh,
            "v_sem_priv_recon": v_sem_priv_recon,
            "z_cf_priv": z_cf_priv,
            "v_sem_recon": v_sem_recon,
            "v_cf_hat": v_cf_hat,
            "shid": sh_indices,
            "sid": sid_indices,
            "cid": cid_indices,
            "loss_recon_sem": loss_recon_sem,
            "loss_recon_cf": loss_recon_cf,
            "loss_orth": loss_orth,
            "loss_orth_sh_sem": loss_orth_sh_sem,
            "loss_orth_sh_cf": loss_orth_sh_cf,
            "loss_orth_priv": loss_orth_priv,
            "loss_vq": loss_vq,
            "loss_vq_sh": vq_loss_sh,
            "loss_vq_sem": vq_loss_sem,
            "loss_vq_cf": vq_loss_cf,
            "loss_total": loss_total,

            "energy_sh_sem": energy_sh_sem,
            "energy_sh_cf": energy_sh_cf,
        }
