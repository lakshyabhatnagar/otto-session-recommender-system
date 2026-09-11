# OTTO Session Recommender

CPU-friendly, session-based retrieval and XGBoost reranking for OTTO clicks, carts, and orders.

The current development baseline uses complete random sessions from `data/processed/train_dev.parquet`. It first splits sessions, derives co-visitation and item statistics only from the training-observed fold, then evaluates held-out sessions without retrieval or popularity leakage.

Its classical candidate-retrieval and separate-target XGBoost structure follows [Theo Viel's 3rd-place OTTO solution](https://github.com/TheoViel/kaggle_otto_rs), while keeping the first version small enough to run on CPU.

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.check_validation
.venv/bin/python -m scripts.run_validation
```
