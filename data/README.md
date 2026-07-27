# Data layout

## Quantizer inputs (shipped)

`tokenizer_triple/data_v3_te3l/AmazonReviews2014/<category>/`

- `v_sem.npy`, `v_cf_128.npy`
- `meta.json`, `id2item.json`, `item2id.json`
- optional `pca_*.npy`

## Generator inputs

- `tokenizer_triple/tokenized/AmazonReviews2014/<category>/<run>/`
  - `tokenizer_state.json` (rewritten after quantizer training)
  - HuggingFace `train` / `val` / `test` splits
- `cache/AmazonReviews2014/<category>/processed/`
  - `all_item_seqs.json`, `id_mapping.json`, `metadata.sentence.json`

## Local-only

- `tokenizer_triple/ckpt_v3_te3l/**/*.pt` — trained by `scripts/run_one.py`
- `cache/**/raw/`, `cache/**/tokenized_standalone_v3_te3l/`
