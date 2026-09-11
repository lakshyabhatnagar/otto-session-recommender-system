"""Compare candidate budgets without training a reranker.

Run from the repository root:
    .venv/bin/python -m scripts.ablate_candidate_budgets
"""

import polars as pl

from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.evaluation import candidate_lists, evaluate_predictions
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


# Each experiment changes only one retrieval source from the current baseline.
BUDGETS = {
    "baseline": {"general_k": 30, "type_k": 20, "time_k": 20, "buy_k": 20},
    "more_general": {"general_k": 50, "type_k": 20, "time_k": 20, "buy_k": 20}, #on running script, this result is useful
    "more_type": {"general_k": 30, "type_k": 40, "time_k": 20, "buy_k": 20},
    "more_time": {"general_k": 30, "type_k": 20, "time_k": 40, "buy_k": 20},
    "more_buy": {"general_k": 30, "type_k": 20, "time_k": 20, "buy_k": 40},
}


def main():
    events = pl.read_parquet("data/processed/train_dev.parquet")
    train_events, validation_events = split_by_session(events)
    train_observed, _ = split_observed_hidden(train_events)
    validation_observed, validation_hidden = split_observed_hidden(validation_events)
    validation_ground_truth = build_ground_truth(validation_hidden)

    # These global relationships are learned from training sessions only.
    matrices = [
        build_covisitation_matrix(train_observed, weighting=weighting)
        for weighting in ["general", "type", "time", "buy_to_buy"]
    ]

    results = []
    for name, budgets in BUDGETS.items():
        candidates = generate_candidates(validation_observed, *matrices, **budgets)
        coverage = evaluate_predictions(candidate_lists(candidates), validation_ground_truth, k=None)

        results.append(
            {
                "experiment": name,
                "candidates_per_session": candidates.height / candidates["session"].n_unique(),
                "click_coverage": coverage["clicks"],
                "cart_coverage": coverage["carts"],
                "order_coverage": coverage["orders"],
                "weighted_coverage": coverage["weighted"],
            }
        )

    print(pl.DataFrame(results).sort("weighted_coverage", descending=True))


if __name__ == "__main__":
    main()
