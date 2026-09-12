"""Measure what each candidate source contributes without training XGBoost.

Run from the repository root:
    .venv/bin/python -m scripts.analyze_candidate_sources
"""

import polars as pl

from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.evaluation import candidate_lists, evaluate_predictions
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


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

    source_candidates = {
        "session_items": validation_observed.select("session", pl.col("aid").alias("candidate")).unique(),
        "general": candidates.filter(pl.col("general_score") > 0).select("session", "candidate"),
        "type_weighted": candidates.filter(pl.col("type_score") > 0).select("session", "candidate"),
        "temporal": candidates.filter(pl.col("time_score") > 0).select("session", "candidate"),
        "buy_to_buy": candidates.filter(pl.col("buy_score") > 0).select("session", "candidate"),
        "all_sources": candidates.select("session", "candidate"),
    }

    results = []
    for source, source_df in source_candidates.items():
        coverage = evaluate_predictions(candidate_lists(source_df), validation_ground_truth, k=None)
        results.append(
            {
                "source": source,
                "candidates_per_session": source_df.height / source_df["session"].n_unique(),
                "click_coverage": coverage["clicks"],
                "cart_coverage": coverage["carts"],
                "order_coverage": coverage["orders"],
                "weighted_coverage": coverage["weighted"],
            }
        )

    print(pl.DataFrame(results).sort("weighted_coverage", descending=True))


if __name__ == "__main__":
    main()
