"""Explain which hidden targets are absent from the candidate union.

Run from the repository root:
    .venv/bin/python -m scripts.analyze_candidate_misses
"""

import polars as pl

from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


TARGET_COLUMNS = {
    "clicks": "click_target",
    "carts": "cart_targets",
    "orders": "order_targets",
}


def main():
    events = pl.read_parquet("data/processed/train_dev.parquet")
    train_events, validation_events = split_by_session(events)
    train_observed, _ = split_observed_hidden(train_events)
    validation_observed, validation_hidden = split_observed_hidden(validation_events)
    validation_ground_truth = build_ground_truth(validation_hidden)

    matrices = [
        build_covisitation_matrix(train_observed, weighting=weighting)
        for weighting in ["general", "type", "time", "buy_to_buy"]
    ]
    candidates = generate_candidates(validation_observed, *matrices)

    observed_items = {
        session: set(items)
        for session, items in validation_observed.group_by("session").agg(pl.col("aid").unique()).iter_rows()
    }
    candidate_items = {
        session: set(items)
        for session, items in candidates.group_by("session").agg(pl.col("candidate").unique()).iter_rows()
    }

    results = []
    for objective, target_column in TARGET_COLUMNS.items():
        counts = {"seen_and_retained": 0, "new_and_retrieved": 0, "new_and_missing": 0}

        for session, target in validation_ground_truth.select("session", target_column).iter_rows():
            targets = [] if target is None else target if isinstance(target, list) else [target]
            for aid in set(targets):
                if aid in observed_items[session]:
                    assert aid in candidate_items[session]
                    counts["seen_and_retained"] += 1
                elif aid in candidate_items[session]:
                    counts["new_and_retrieved"] += 1
                else:
                    counts["new_and_missing"] += 1

        total = sum(counts.values())
        results.append({"objective": objective, "targets": total, **counts, "missing_rate": counts["new_and_missing"] / total})

    print(pl.DataFrame(results))


if __name__ == "__main__":
    main()
