"""Launch the original DiscoGR pipeline for one star config.

Steps:
  1) train TripleQuantizer if the checkpoint is missing
  2) export tokenizer_state + copy HF splits if missing
  3) run main_triple.py with the original trainer / CEA generator
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

def _append_opt(cmd: list[str], flag: str, value) -> None:
    if value is None or value == "":
        return
    cmd.append(f"{flag}={value}")

def quantizer_data_dir(cfg: dict) -> Path:
    quantizer_meta = (cfg.get("quantizer_meta") or "").strip()
    if quantizer_meta:
        return (ROOT / quantizer_meta).parent
    return ROOT / "tokenizer_triple" / "data_v3_te3l" / cfg["dataset"] / cfg["category"]

def build_quantizer_train_cmd(cfg: dict) -> list[str]:
    ckpt_path = ROOT / cfg["quantizer_ckpt"]
    return [
        sys.executable,
        str(ROOT / "tokenizer_triple" / "train_triple_rqvae.py"),
        f"--category={cfg['category']}",
        f"--dataset={cfg['dataset']}",
        f"--data_dir={quantizer_data_dir(cfg)}",
        f"--v_cf_name={cfg.get('v_cf_name', 'v_cf_128.npy')}",
        "--meta_name=meta.json",
        f"--cf_meta_key={cfg.get('cf_meta_key', 'd_cf_128')}",
        f"--n_levels_sh={cfg['n_digit_sh']}",
        f"--n_levels_sem={cfg['n_digit_sem']}",
        f"--n_levels_cf={cfg['n_digit_cf']}",
        f"--n_codes_sh={cfg.get('codebook_size', 256)}",
        f"--n_codes_sem={cfg.get('codebook_size', 256)}",
        f"--n_codes_cf={cfg.get('codebook_size', 256)}",
        f"--d_latent={cfg['d_latent']}",
        f"--hidden_sh={cfg.get('hidden_sh', 512)}",
        f"--lambda_orth={cfg.get('quantizer_lambda_orth', 0.1)}",
        f"--epochs={cfg.get('quantizer_epochs', 250)}",
        f"--batch_size={cfg.get('quantizer_batch_size', 4096)}",
        f"--lr={cfg.get('quantizer_lr', 0.001)}",
        f"--save_dir={ckpt_path.parent}",
        f"--ckpt_name={ckpt_path.name}",
    ]

def build_export_cmd(cfg: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tokenizer_triple" / "export_triple_rqvae_tokens.py"),
        f"--category={cfg['category']}",
        f"--dataset={cfg['dataset']}",
        f"--data_dir={quantizer_data_dir(cfg)}",
        f"--v_cf_name={cfg.get('v_cf_name', 'v_cf_128.npy')}",
        "--meta_name=meta.json",
        f"--cf_meta_key={cfg.get('cf_meta_key', 'd_cf_128')}",
        f"--ckpt={ROOT / cfg['quantizer_ckpt']}",
        f"--n_levels_sh={cfg['n_digit_sh']}",
        f"--n_levels_sem={cfg['n_digit_sem']}",
        f"--n_levels_cf={cfg['n_digit_cf']}",
        f"--n_codes_sh={cfg.get('codebook_size', 256)}",
        f"--n_codes_sem={cfg.get('codebook_size', 256)}",
        f"--n_codes_cf={cfg.get('codebook_size', 256)}",
        f"--d_latent={cfg['d_latent']}",
        f"--hidden_sh={cfg.get('hidden_sh', 512)}",
        f"--lambda_orth={cfg.get('quantizer_lambda_orth', 0.1)}",
        f"--out_dir={ROOT / cfg['tokenized_dir']}",
    ]
    copy_from = cfg.get("copy_splits_from")
    if copy_from:
        cmd.append(f"--copy_splits_from={ROOT / copy_from}")
    return cmd

def build_generator_cmd(cfg: dict, gpu: str | None) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "main_triple.py"),
        f"--model={cfg.get('model', 'DiscoGR')}",
        f"--dataset={cfg['dataset']}",
        f"--category={cfg['category']}",
        f"--tokenized_dir={ROOT / cfg['tokenized_dir']}",
        f"--n_digit_sh={cfg['n_digit_sh']}",
        f"--n_digit_sem={cfg['n_digit_sem']}",
        f"--n_digit_cf={cfg['n_digit_cf']}",
        f"--n_codebook={cfg['n_codebook']}",
        f"--hidden_sh={cfg.get('hidden_sh', 512)}",
        f"--d_latent={cfg.get('d_latent', 512)}",
        f"--cf_meta_key={cfg.get('cf_meta_key', 'd_cf_128')}",
        f"--triple_attn_type={cfg.get('triple_attn_type', 'shared_route')}",
        f"--lambda_orth_triple={cfg.get('lambda_orth_triple', 0.0)}",
        f"--downstream_orth_streams={cfg.get('downstream_orth_streams', 'all3')}",
        f"--lr={cfg['lr']}",
        f"--temperature={cfg.get('temperature', 0.03)}",
        f"--train_batch_size={cfg.get('train_batch_size', 256)}",
        f"--eval_batch_size={cfg.get('eval_batch_size', 256)}",
        f"--epochs={cfg.get('epochs', 100)}",
        f"--n_layer={cfg.get('n_layer', 2)}",
        f"--n_head={cfg.get('n_head', 4)}",
        f"--n_embd={cfg.get('n_embd', 448)}",
        f"--n_inner={cfg.get('n_inner', 1024)}",
        f"--generate_w_decoding_graph={str(bool(cfg.get('generate_w_decoding_graph', False))).lower()}",
        f"--run_id={cfg.get('run_id', cfg['category'])}",
        "--use_wandb=False",
        f"--ckpt_dir={ROOT / cfg.get('ckpt_dir', 'outputs/ckpts')}",
        f"--log_dir={ROOT / cfg.get('log_dir', 'outputs/logs')}",
        f"--tensorboard_log_dir={ROOT / cfg.get('tensorboard_log_dir', 'outputs/tensorboard')}",
        f"--cache_dir={ROOT / cfg.get('cache_dir', 'cache')}",
    ]
    for key in (
        "num_beams",
        "n_edges",
        "propagation_steps",
        "rand_seed",
        "patience",
        "warmup_steps",
        "freeze_sh",
        "freeze_sem",
    ):
        _append_opt(cmd, f"--{key}", cfg.get(key))
    _append_opt(cmd, "--quantizer_ckpt", str(ROOT / cfg["quantizer_ckpt"]) if cfg.get("quantizer_ckpt") else None)
    _append_opt(cmd, "--quantizer_meta", str(ROOT / cfg["quantizer_meta"]) if cfg.get("quantizer_meta") else None)
    env_hint = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    print(f"[run_one] generator cmd: {env_hint}{' '.join(cmd)}")
    return cmd

def ensure_quantizer(cfg: dict, dry_run: bool, force: bool = False) -> None:
    ckpt = ROOT / cfg["quantizer_ckpt"]
    if ckpt.exists() and not force:
        print(f"[run_one] quantizer exists: {ckpt}")
        return
    cmd = build_quantizer_train_cmd(cfg)
    print("[run_one] train quantizer:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)

def ensure_tokenized(cfg: dict, dry_run: bool, force: bool = False) -> None:
    tokenized = ROOT / cfg["tokenized_dir"]
    state = tokenized / "tokenizer_state.json"
    splits_ok = all((tokenized / split).is_dir() for split in ("train", "val", "test"))
    if state.exists() and splits_ok and not force:
        print(f"[run_one] tokenized_dir ready: {tokenized}")
        return
    cmd = build_export_cmd(cfg)
    print("[run_one] export tokens:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def tokenized_ready(cfg: dict) -> bool:
    tokenized = ROOT / cfg["tokenized_dir"]
    state = tokenized / "tokenizer_state.json"
    splits_ok = all((tokenized / split).is_dir() for split in ("train", "val", "test"))
    return state.exists() and splits_ok

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a DiscoGR star YAML")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_quantizer", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_generator", action="store_true")
    parser.add_argument("--force_quantizer", action="store_true")
    parser.add_argument("--force_export", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ready = tokenized_ready(cfg)

    if not args.skip_quantizer and not (ready and not args.force_quantizer and not args.force_export):
        ensure_quantizer(cfg, args.dry_run, force=args.force_quantizer)
    if not args.skip_export:
        ensure_tokenized(cfg, args.dry_run, force=args.force_export)
    if not args.skip_generator:
        cmd = build_generator_cmd(cfg, args.gpu)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True, env=env)

if __name__ == "__main__":
    main()
