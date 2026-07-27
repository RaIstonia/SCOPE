import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _build_quantizer(d_sem: int, d_cf: int, args):
    from tokenizer_triple.triple_rqvae import TripleQuantizer  # noqa: E402
    return TripleQuantizer(
        d_sem=d_sem, d_cf=d_cf,
        n_codes_sh=args.n_codes_sh, n_codes_sem=args.n_codes_sem, n_codes_cf=args.n_codes_cf,
        n_levels_sh=args.n_levels_sh, n_levels_sem=args.n_levels_sem, n_levels_cf=args.n_levels_cf,
        hidden_sh=args.hidden_sh,
        d_latent=args.d_latent,
        lambda_orth=args.lambda_orth,
    )

def main():
    parser = argparse.ArgumentParser(description="Export TripleQuantizer codes as tokenizer_state.")
    parser.add_argument("--category", type=str, default="Sports_and_Outdoors")
    parser.add_argument("--dataset", type=str, default="AmazonReviews2014")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with v_sem.npy / v_cf.npy / meta.json / id2item.json")
    parser.add_argument("--v_sem_name", type=str, default="v_sem.npy")
    parser.add_argument("--v_cf_name", type=str, default="v_cf.npy")
    parser.add_argument("--meta_name", type=str, default="meta.json")
    parser.add_argument("--cf_meta_key", type=str, default="d_cf",
                        help="meta.json key for CF dim; use d_cf_128 for SASRec128d")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to TripleQuantizer checkpoint")
    parser.add_argument("--n_codes_sh", type=int, default=256)
    parser.add_argument("--n_codes_sem", type=int, default=256)
    parser.add_argument("--n_codes_cf", type=int, default=256)
    parser.add_argument("--n_levels_sh", type=int, default=8)
    parser.add_argument("--n_levels_sem", type=int, default=16)
    parser.add_argument("--n_levels_cf", type=int, default=8)
    parser.add_argument("--hidden_sh", type=int, default=256)
    parser.add_argument("--d_latent", type=int, default=128)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--copy_splits_from", type=str, default=None,
                        help="Copy train/val/test and max_token_seq_len from this tokenized dir")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.n_codes_sh == args.n_codes_sem == args.n_codes_cf, (
        "DiscoGR shares codebook_size per digit; keep n_codes_sh == n_codes_sem == n_codes_cf"
    )
    codebook_size = args.n_codes_sh

    data_dir = args.data_dir or os.path.join(
        ROOT, "tokenizer_triple", "data_v3_te3l", args.dataset, args.category
    )
    ckpt_path = args.ckpt or os.path.join(
        ROOT, "tokenizer_triple", "ckpt", args.dataset, args.category,
        f"sh{args.n_levels_sh}_sem{args.n_levels_sem}_cf{args.n_levels_cf}",
        "baseline", "triple_rqvae.pt",
    )
    out_dir = args.out_dir or os.path.join(
        ROOT, "tokenizer_triple", "tokenized", args.dataset, args.category,
        f"triple_sh{args.n_levels_sh}_sem{args.n_levels_sem}_cf{args.n_levels_cf}",
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, args.meta_name), "r") as f:
        meta = json.load(f)
    with open(os.path.join(data_dir, "id2item.json"), "r") as f:
        id2item_raw = json.load(f)
    if isinstance(id2item_raw, list):
        id2item = {i: id2item_raw[i] for i in range(len(id2item_raw))}
    else:
        id2item = dict(id2item_raw)
    v_sem = np.load(os.path.join(data_dir, args.v_sem_name)).astype(np.float32)
    v_cf = np.load(os.path.join(data_dir, args.v_cf_name)).astype(np.float32)
    if args.cf_meta_key not in meta:
        raise KeyError(
            f"meta missing field '{args.cf_meta_key}'; available: "
            f"{[k for k in meta if k.startswith('d_cf') or k == 'd_sem']}"
        )
    n_items, d_sem, d_cf = meta["n_items"], meta["d_sem"], meta[args.cf_meta_key]
    assert v_sem.shape == (n_items, d_sem) and v_cf.shape == (n_items, d_cf)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_quantizer(d_sem, d_cf, args).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_legacy_compatible_state_dict(state_dict)
    model.eval()

    n_digit = args.n_levels_sh + args.n_levels_sem + args.n_levels_cf

    item2tokens = {}
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            x_sem = torch.from_numpy(v_sem[start:end]).to(device)
            x_cf = torch.from_numpy(v_cf[start:end]).to(device)
            out = model(x_sem, x_cf)
            sh_list = [t.cpu().numpy() for t in out["shid"]]
            sid_list = [t.cpu().numpy() for t in out["sid"]]
            cid_list = [t.cpu().numpy() for t in out["cid"]]
            for b in range(end - start):
                item_id = start + b
                codes = []
                for lv in range(args.n_levels_sh):
                    codes.append(int(sh_list[lv][b]))
                for lv in range(args.n_levels_sem):
                    codes.append(int(sid_list[lv][b]))
                for lv in range(args.n_levels_cf):
                    codes.append(int(cid_list[lv][b]))
                tokens = [codes[p] + p * codebook_size + 1 for p in range(n_digit)]
                item_str = id2item.get(item_id, str(item_id))
                item2tokens[item_str] = tokens

    eos_token = n_digit * codebook_size + 1
    vocab_size = eos_token + 1
    max_token_seq_len = 50
    if args.copy_splits_from:
        ts_path = os.path.join(args.copy_splits_from, "tokenizer_state.json")
        if os.path.isfile(ts_path):
            try:
                with open(ts_path, "r") as f:
                    ts_orig = json.load(f)
                max_token_seq_len = int(ts_orig.get("max_token_seq_len", max_token_seq_len))
            except Exception:
                pass

    tokenizer_state = {
        "item2tokens": {str(k): list(v) for k, v in item2tokens.items()},
        "eos_token": int(eos_token),
        "n_digit": int(n_digit),
        "codebook_size": int(codebook_size),
        "vocab_size": int(vocab_size),
        "max_token_seq_len": int(max_token_seq_len),
        "ignored_label": -100,
        # Triple-stream specific
        "n_levels_sh": int(args.n_levels_sh),
        "n_levels_sem": int(args.n_levels_sem),
        "n_levels_cf": int(args.n_levels_cf),
        "triple_rqvae_variant": "baseline",
        "triple_rqvae_ckpt": os.path.relpath(ckpt_path, ROOT),
    }
    with open(os.path.join(out_dir, "tokenizer_state.json"), "w") as f:
        json.dump(tokenizer_state, f, indent=2)
    print(
        f"Saved tokenizer_state.json -> {out_dir} "
        f"(n_digit={n_digit}, vocab_size={vocab_size}, sh:sem:cf={args.n_levels_sh}:{args.n_levels_sem}:{args.n_levels_cf})"
    )

    if args.copy_splits_from and os.path.isdir(args.copy_splits_from):
        for split in ("train", "val", "test"):
            src = os.path.join(args.copy_splits_from, split)
            dst = os.path.join(out_dir, split)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"Copied {split} from {args.copy_splits_from} -> {out_dir}")
        print(f"\nReady: --tokenized_dir={out_dir} --n_codebook={n_digit}")
    else:
        print(
            f"\nCopy train/val/test into {out_dir}, "
            f"or rerun with --copy_splits_from=<tokenized_dir>"
        )

if __name__ == "__main__":
    main()
