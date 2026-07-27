# DiscoGR

Minimal release for running `DiscoGR`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Required files

- `configs/`
- `genrec/`
- `tokenizer_triple/` (code + `data_v3_te3l/` embeddings + `tokenized/` splits)
- `scripts/`
- `main_triple.py`
- `cache/AmazonReviews2014/<category>/processed/`

`run_one.py` trains the quantizer when `quantizer_ckpt` is missing, re-exports
item codes, then trains the generator with checkpoint-based codebook init.

## Train

Default (quantizer → export → generator):

Beauty:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_one.py configs/beauty.yaml --gpu 0
```

Sports:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_one.py configs/sports.yaml --gpu 0
```

Toys:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_one.py configs/toys.yaml --gpu 0
```

Yelp:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_one.py configs/yelp.yaml --gpu 0
```

Force a fresh quantizer + re-export:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_one.py configs/beauty.yaml \
  --gpu 0 --force_quantizer --force_export
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_checkpoint.py \
  --config configs/beauty.yaml \
  --checkpoint path/to/generator.pth \
  --output outputs/beauty_eval.json \
  --gpu 0
```

## Note

Quantizer `.pt` weights are produced locally under `tokenizer_triple/ckpt_v3_te3l/`
and are not shipped. Embedding inputs live in `tokenizer_triple/data_v3_te3l/`.
Shipped `tokenized/` splits are reused after re-export; only `tokenizer_state.json`
is rewritten to match the newly trained quantizer.
