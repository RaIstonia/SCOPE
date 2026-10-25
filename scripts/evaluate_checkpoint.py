"""Evaluate a generator checkpoint with the original full-catalog protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from genrec.pipeline import Pipeline

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    config_dict = {
        "category": cfg["category"],
        "tokenized_dir": str(ROOT / cfg["tokenized_dir"]),
        "n_digit_sh": cfg["n_digit_sh"],
        "n_digit_sem": cfg["n_digit_sem"],
        "n_digit_cf": cfg["n_digit_cf"],
        "n_codebook": cfg["n_codebook"],
        "hidden_sh": cfg.get("hidden_sh", 512),
        "d_latent": cfg.get("d_latent", 512),
        "cf_meta_key": cfg.get("cf_meta_key", "d_cf_128"),
        "triple_attn_type": cfg.get("triple_attn_type", "shared_route"),
        "lambda_orth_triple": cfg.get("lambda_orth_triple", 0.0),
        "downstream_orth_streams": cfg.get("downstream_orth_streams", "all3"),
        "generate_w_decoding_graph": False,
        "train_batch_size": cfg.get("train_batch_size", 256),
        "eval_batch_size": cfg.get("eval_batch_size", 256),
        "epochs": 1,
        "use_wandb": False,
        "cache_dir": str(ROOT / cfg.get("cache_dir", "cache")),
        "ckpt_dir": str(ROOT / cfg.get("ckpt_dir", "outputs/ckpts")),
        "log_dir": str(ROOT / cfg.get("log_dir", "outputs/logs")),
        "tensorboard_log_dir": str(
            ROOT / cfg.get("tensorboard_log_dir", "outputs/tensorboard")
        ),
        "run_id": cfg.get("run_id", cfg["category"]),
        "lr": cfg["lr"],
        "temperature": cfg.get("temperature", 0.03),
        "n_layer": cfg.get("n_layer", 2),
        "n_head": cfg.get("n_head", 4),
        "n_embd": cfg.get("n_embd", 448),
        "n_inner": cfg.get("n_inner", 1024),
        "rand_seed": cfg.get("rand_seed", 42),
        "freeze_sh": cfg.get("freeze_sh", False),
        "freeze_sem": cfg.get("freeze_sem", False),
    }
    if cfg.get("quantizer_ckpt"):
        config_dict["quantizer_ckpt"] = str(ROOT / cfg["quantizer_ckpt"])
    if cfg.get("quantizer_meta"):
        config_dict["quantizer_meta"] = str(ROOT / cfg["quantizer_meta"])

    pipe = Pipeline(
        model_name=cfg.get("model", "SCOPE"),
        dataset_name=cfg["dataset"],
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        config_dict=config_dict,
    )
    test_dataloader = DataLoader(
        pipe.tokenized_datasets["test"],
        batch_size=pipe.config["eval_batch_size"],
        shuffle=False,
        collate_fn=pipe.tokenizer.collate_fn["test"],
    )
    pipe.model, test_dataloader = pipe.accelerator.prepare(pipe.model, test_dataloader)
    pipe.trainer.model.generate_w_decoding_graph = False
    results = pipe.trainer.evaluate(test_dataloader, split="test")
    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
    pipe.trainer.end()

if __name__ == "__main__":
    main()
