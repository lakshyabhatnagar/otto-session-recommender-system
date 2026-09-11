"""Run a small leak-free validation check: python -m scripts.check_validation."""

import polars as pl

from src.training.evaluation import candidate_lists, evaluate_predictions, recall_at_k
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


def main():
    events = pl.DataFrame(
        {
            "session": [1, 1, 2, 2, 3, 3, 4, 4],
            "aid": [10, 11, 20, 21, 30, 31, 40, 41],
            "ts": list(range(8)),
            "type": ["clicks"] * 8,
        }
    )
    train_events, validation_events = split_by_session(events, val_size=0.5, seed=42)
    assert set(train_events["session"]) & set(validation_events["session"]) == set()
    repeated_train, repeated_validation = split_by_session(events, val_size=0.5, seed=42)
    assert train_events.equals(repeated_train)
    assert validation_events.equals(repeated_validation)

    observed, hidden = split_observed_hidden(train_events)
    ground_truth = build_ground_truth(hidden)
    assert observed.height == hidden.height == train_events.height // 2

    candidates = hidden.select("session", pl.col("aid").alias("candidate"))
    predictions = candidate_lists(candidates)
    scores = evaluate_predictions(predictions, ground_truth, k=None)
    assert scores == {"clicks": 1.0, "carts": 0.0, "orders": 0.0, "weighted": 0.1}

    long_targets = pl.DataFrame({"session": [1], "cart_targets": [list(range(21))]})
    top_twenty = pl.DataFrame({"session": [1], "predictions": [list(range(20))]})
    assert recall_at_k(top_twenty, long_targets, "cart_targets") == 1.0
    print("validation checks passed")


if __name__ == "__main__":
    main()
