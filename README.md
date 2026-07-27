# DiscoGR

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

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

