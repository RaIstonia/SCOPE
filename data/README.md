# Data layout

Generator training needs:

- `tokenizer_triple/tokenized/AmazonReviews2014/<run>/`
  - `tokenizer_state.json`
  - HuggingFace `train` / `val` / `test`
- `cache/AmazonReviews2014/<category>/processed/`
  - `all_item_seqs.json`, `id_mapping.json`, `metadata.sentence.json`

Raw OpenAI embedding caches, legacy retrain assets, and API secrets are
intentionally excluded.
