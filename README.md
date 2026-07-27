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

Keep these paths:

- `configs/`
- `genrec/`
- `tokenizer_triple/`
- `scripts/`
- `main_triple.py`
- `cache/AmazonReviews2014/<category>/processed/`
- `tokenizer_triple/tokenized/AmazonReviews2014/<category>/<run>/`
  - `tokenizer_state.json`
  - `train/val/test/data-00000-of-00001.arrow`
  - `train/val/test/dataset_info.json`
  - `train/val/test/state.json`

## Train

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

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_checkpoint.py \
  --config configs/beauty.yaml \
  --checkpoint path/to/generator.pth \
  --output outputs/beauty_eval.json \
  --gpu 0
```

## Note

Quantizer `.pt` weights are not shipped. DiscoGR falls back to random codebook
initialization when `quantizer_ckpt` is absent. Pre-exported item codes under
`tokenizer_triple/tokenized/` are sufficient for generator training.
`quantizer_meta` is optional; if a local checkpoint is provided later, dimensions
can be inferred from the weights.
