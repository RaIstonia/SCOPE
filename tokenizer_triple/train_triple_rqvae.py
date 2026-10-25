import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tokenizer_triple.triple_rqvae import TripleQuantizer  # noqa: E402

def _build_model(d_sem: int, d_cf: int, args):
    return TripleQuantizer(
        d_sem=d_sem, d_cf=d_cf,
        n_codes_sh=args.n_codes_sh, n_codes_sem=args.n_codes_sem, n_codes_cf=args.n_codes_cf,
        n_levels_sh=args.n_levels_sh, n_levels_sem=args.n_levels_sem, n_levels_cf=args.n_levels_cf,
        hidden_sh=args.hidden_sh,
        d_latent=args.d_latent,
        lambda_orth=args.lambda_orth,
        lambda_orth_sh_sem=args.lambda_orth_sh_sem,
        lambda_orth_sh_cf=args.lambda_orth_sh_cf,
        lambda_orth_priv=args.lambda_orth_priv,
        lambda_vq_sh=args.lambda_vq_sh,
        lambda_vq_sem=args.lambda_vq_sem,
        lambda_vq_cf=args.lambda_vq_cf,
    )

def _print_banner(args, extra: dict | None = None):

    print("=" * 88)
    print("[TripleQuantizer Training Run]")
    print("-" * 88)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:>20s} = {v}")
    if extra:
        print("-" * 88)
        for k, v in sorted(extra.items()):
            print(f"  {k:>20s} = {v}")
    print("=" * 88, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="Sports_and_Outdoors")
    parser.add_argument("--dataset", type=str, default="AmazonReviews2014")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with v_sem / v_cf / meta")
    parser.add_argument("--v_sem_name", type=str, default="v_sem.npy")
    parser.add_argument("--v_cf_name", type=str, default="v_cf.npy")
    parser.add_argument("--meta_name", type=str, default="meta.json")
    parser.add_argument("--cf_meta_key", type=str, default="d_cf",
                        help="meta.json key for CF dim; use d_cf_128 for SASRec128d")
    parser.add_argument("--n_codes_sh", type=int, default=256)
    parser.add_argument("--n_codes_sem", type=int, default=256)
    parser.add_argument("--n_codes_cf", type=int, default=256)
    parser.add_argument("--n_levels_sh", type=int, default=8)
    parser.add_argument("--n_levels_sem", type=int, default=16)
    parser.add_argument("--n_levels_cf", type=int, default=8)
    parser.add_argument("--hidden_sh", type=int, default=256,
                        help="Hidden width of the shared fusion encoder")
    parser.add_argument("--d_latent", type=int, default=128)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--lambda_orth_sh_sem", type=float, default=1.0)
    parser.add_argument("--lambda_orth_sh_cf", type=float, default=1.0)
    parser.add_argument("--lambda_orth_priv", type=float, default=1.0)
    parser.add_argument("--lambda_vq_sh", type=float, default=1.0)
    parser.add_argument("--lambda_vq_sem", type=float, default=1.0)
    parser.add_argument("--lambda_vq_cf", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, default="triple_rqvae.pt",
                        help="Checkpoint filename inside save_dir")
    args = parser.parse_args()
    assert args.n_codes_sh == args.n_codes_sem == args.n_codes_cf, (
        "SCOPE shares codebook_size per digit; keep n_codes_sh == n_codes_sem == n_codes_cf"
    )
    codebook_size = args.n_codes_sh

    data_dir = args.data_dir or os.path.join(
        ROOT, "tokenizer_triple", "data_v3_te3l", args.dataset, args.category
    )
    save_dir = args.save_dir or os.path.join(
        ROOT, "tokenizer_triple", "ckpt", args.dataset, args.category,
        f"sh{args.n_levels_sh}_sem{args.n_levels_sem}_cf{args.n_levels_cf}",
        "baseline",
    )
    os.makedirs(save_dir, exist_ok=True)

    v_sem = np.load(os.path.join(data_dir, args.v_sem_name)).astype(np.float32)
    v_cf = np.load(os.path.join(data_dir, args.v_cf_name)).astype(np.float32)
    with open(os.path.join(data_dir, args.meta_name), "r") as f:
        meta = json.load(f)
    if args.cf_meta_key not in meta:
        raise KeyError(
            f"meta missing field '{args.cf_meta_key}'; available: "
            f"{[k for k in meta if k.startswith('d_cf') or k == 'd_sem']}"
        )
    n_items, d_sem, d_cf = meta["n_items"], meta["d_sem"], meta[args.cf_meta_key]
    assert v_sem.shape == (n_items, d_sem) and v_cf.shape == (n_items, d_cf), (
        "v_sem / v_cf must share shape (n_items, d_*); row i is the same item"
    )

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    _print_banner(args, extra={
        "data_dir": data_dir,
        "save_dir": save_dir,
        "n_items": n_items,
        "d_sem": d_sem,
        "d_cf": d_cf,
        "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    })

    model = _build_model(d_sem, d_cf, args).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    idx = np.arange(1, n_items)
    n_lv_sh, n_lv_sem, n_lv_cf = args.n_levels_sh, args.n_levels_sem, args.n_levels_cf

    for ep in range(args.epochs):
        np.random.shuffle(idx)
        sums = {
            "recon_sem": 0.0, "recon_cf": 0.0,
            "orth": 0.0, "orth_sh_sem": 0.0, "orth_sh_cf": 0.0, "orth_priv": 0.0,
            "vq": 0.0, "vq_sh": 0.0, "vq_sem": 0.0, "vq_cf": 0.0,
            "energy_sh_sem": 0.0, "energy_sh_cf": 0.0,
        }
        n_b = 0

        sh_all = [[] for _ in range(n_lv_sh)]
        sid_all = [[] for _ in range(n_lv_sem)]
        cid_all = [[] for _ in range(n_lv_cf)]

        for start in range(0, len(idx), args.batch_size):
            bi = idx[start:start + args.batch_size]
            x_sem = torch.from_numpy(v_sem[bi]).to(device)
            x_cf = torch.from_numpy(v_cf[bi]).to(device)

            out = model(x_sem, x_cf)
            opt.zero_grad()
            out["loss_total"].backward()
            opt.step()

            sums["recon_sem"] += out["loss_recon_sem"].item()
            sums["recon_cf"] += out["loss_recon_cf"].item()
            sums["orth"] += out["loss_orth"].item()
            sums["orth_sh_sem"] += out["loss_orth_sh_sem"].item()
            sums["orth_sh_cf"] += out["loss_orth_sh_cf"].item()
            sums["orth_priv"] += out["loss_orth_priv"].item()
            sums["vq"] += out["loss_vq"].item()
            sums["vq_sh"] += out["loss_vq_sh"].item()
            sums["vq_sem"] += out["loss_vq_sem"].item()
            sums["vq_cf"] += out["loss_vq_cf"].item()
            sums["energy_sh_sem"] += out["energy_sh_sem"].item()
            sums["energy_sh_cf"] += out["energy_sh_cf"].item()
            n_b += 1

            for lv, t in enumerate(out["shid"]):
                sh_all[lv].append(t.detach())
            for lv, t in enumerate(out["sid"]):
                sid_all[lv].append(t.detach())
            for lv, t in enumerate(out["cid"]):
                cid_all[lv].append(t.detach())

        scheduler.step()

        util_sh = [torch.cat(sh_all[lv], dim=0).unique().numel() / codebook_size for lv in range(n_lv_sh)]
        util_sem = [torch.cat(sid_all[lv], dim=0).unique().numel() / codebook_size for lv in range(n_lv_sem)]
        util_cf = [torch.cat(cid_all[lv], dim=0).unique().numel() / codebook_size for lv in range(n_lv_cf)]
        all_util = util_sh + util_sem + util_cf
        u_min, u_max, u_mean = min(all_util), max(all_util), sum(all_util) / len(all_util)

        sh_cols = [torch.cat(sh_all[lv], dim=0).long().cpu() for lv in range(n_lv_sh)]
        sid_cols = [torch.cat(sid_all[lv], dim=0).long().cpu() for lv in range(n_lv_sem)]
        cid_cols = [torch.cat(cid_all[lv], dim=0).long().cpu() for lv in range(n_lv_cf)]
        code_mat = torch.stack(sh_cols + sid_cols + cid_cols, dim=1)  # [N, n_total]
        n_items_epoch = int(code_mat.shape[0])
        uniq_codes, counts = torch.unique(code_mat, dim=0, return_counts=True)
        n_unique = int(uniq_codes.shape[0])
        item_collision = (n_items_epoch - n_unique) / max(n_items_epoch, 1)
        pair_total = n_items_epoch * (n_items_epoch - 1) / 2.0
        pair_coll = float(((counts * (counts - 1)) // 2).sum().item())
        pair_collision = pair_coll / pair_total if pair_total > 0 else 0.0

        sh_str = " ".join([f"L{i}:util={util_sh[i]:.2%}" for i in range(n_lv_sh)])
        sem_str = " ".join([f"L{i}:util={util_sem[i]:.2%}" for i in range(n_lv_sem)])
        cf_str = " ".join([f"L{i}:util={util_cf[i]:.2%}" for i in range(n_lv_cf)])

        line = (
            f"Epoch {ep+1} "
            f"recon_sem={sums['recon_sem']/n_b:.4f} recon_cf={sums['recon_cf']/n_b:.4f} "
            f"orth={sums['orth']/n_b:.4f} (sh-sem={sums['orth_sh_sem']/n_b:.4f} "
            f"sh-cf={sums['orth_sh_cf']/n_b:.4f} priv={sums['orth_priv']/n_b:.4f}) "
        )
        line += (
            f"vq={sums['vq']/n_b:.4f} (sh={sums['vq_sh']/n_b:.4f} sem={sums['vq_sem']/n_b:.4f} cf={sums['vq_cf']/n_b:.4f}) "
            f"| E_sh(sem)={sums['energy_sh_sem']/n_b:.3f} E_sh(cf)={sums['energy_sh_cf']/n_b:.3f} "
        )
        line += (
            f"| util min={u_min:.2%} max={u_max:.2%} mean={u_mean:.2%} "
            f"| item_collision={item_collision:.2%} (unique={n_unique}/{n_items_epoch}) "
            f"| pair_collision={pair_collision:.2%} "
            f"| ShID [{sh_str}] | SID [{sem_str}] | CID [{cf_str}]"
        )
        print(line, flush=True)

    out_path = os.path.join(save_dir, args.ckpt_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
