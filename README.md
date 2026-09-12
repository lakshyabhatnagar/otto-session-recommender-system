# OTTO Session Recommender

An offline, CPU-friendly multi-objective e-commerce recommender. Given an anonymous session history, it retrieves a small set of product candidates and ranks them separately for clicks, carts, and orders.

## System design

```text
session events
    -> co-visitation retrieval
    -> session, item, and retrieval features
    -> three XGBoost scoring models
    -> Top-20 products per objective
```

The system does not score every product in the catalog. It first retrieves candidates from:

- products already seen in the session;
- general co-visitation;
- event-type-weighted co-visitation;
- time-weighted co-visitation; and
- cart/order-only co-visitation.

It then adds candidate-level features such as retrieval strength, session repetition, recency, item popularity, cart/order rates, and session context. Three separate XGBoost classifiers rank the same candidate set for the three objectives.

## Development evaluation

The development dataset contains 100,000 randomly sampled complete sessions. Sessions are split 80/20 into training and validation groups; each group is then split by position into a visible 70% history and a hidden 30% future. Global retrieval and item statistics are built from training-visible events only.

Final validated development results:

| Metric | Clicks | Carts | Orders | Weighted |
|---|---:|---:|---:|---:|
| Candidate target coverage | 0.42018 | 0.38209 | 0.69968 | — |
| Candidate oracle Recall@20 | 0.42018 | 0.38245 | 0.70037 | 0.57697 |
| XGBoost Recall@20 | 0.38548 | 0.30062 | 0.52849 | **0.44583** |

These are development-split results, not a claim about future-time performance. A chronological validation protocol is the next evaluation improvement.

## Run

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.check_validation
.venv/bin/python -m scripts.run_validation --save-dir artifacts/dev_baseline
```

The final command saves the three model files, four retrieval matrices, item statistics, feature order, candidate budgets, and metrics. The generated artifacts are local and ignored by Git.

## Reuse the trained system

```python
import polars as pl

from src.inference import recommend

session_events = pl.read_parquet("data/processed/test.parquet")
predictions = recommend(session_events, "artifacts/dev_baseline")
print(predictions["orders"])
```

`predictions` contains one `session, predictions` table for each objective. It does not access labels or retrain the model.

## Current limitations

- Validation uses randomly held-out sessions with a positional history/future split; it is not yet a chronological future-time evaluation.
- The current retrieval layer is heuristic co-visitation. Learned retrieval, embeddings, and sequence models are deliberately deferred until a measured retrieval limitation justifies them.
- The current models are pointwise classifiers. A session-grouped learning-to-rank comparison is a future controlled experiment, not an assumed improvement.
