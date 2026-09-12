"""Run the leak-free multi-objective baseline: python -m scripts.run_validation."""

import polars as pl

from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.dataset import build_training_dataset
from src.training.evaluation import candidate_lists, evaluate_predictions, recall_at_k
from src.training.train_xgb import prepare_training_data, train_binary_model
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


OBJECTIVES = {
    "clicks": ("click_label", "click_target"),
    "carts": ("cart_label", "cart_targets"),
    "orders": ("order_label", "order_targets"),
}


def main():
    events = pl.read_parquet("data/processed/train_dev.parquet")
    print("Splitting sessions into training and validation data...")
    train_events, validation_events = split_by_session(events)
    train_observed, train_hidden = split_observed_hidden(train_events)
    validation_observed, validation_hidden = split_observed_hidden(validation_events)
    train_ground_truth = build_ground_truth(train_hidden)
    validation_ground_truth = build_ground_truth(validation_hidden)

    matrices = []
    for weighting in ["general", "type", "time", "buy_to_buy"]:
        print(f"Building {weighting} co-visitation matrix...")
        matrices.append(build_covisitation_matrix(train_observed, weighting=weighting))

    print("Generating candidates...")
    train_candidates = generate_candidates(train_observed, *matrices)
    validation_candidates = generate_candidates(validation_observed, *matrices)
    print("Validation candidates per session:", validation_candidates.height / validation_candidates["session"].n_unique())

    print("Building candidate features and labels...")
    training_df = build_training_dataset(train_candidates, train_observed, train_ground_truth)
    validation_df = build_training_dataset(
        validation_candidates,
        validation_observed,
        validation_ground_truth,
        item_events=train_observed,
    )

    print("Candidate coverage:", evaluate_predictions(candidate_lists(validation_candidates), validation_ground_truth, k=None))
    recalls = {}
    for objective, (label_column, target_column) in OBJECTIVES.items():
        print(f"Training {objective} XGBoost model...")
        (X_train, y_train), (X_val, y_val) = prepare_training_data(training_df, validation_df, label_column)
        model = train_binary_model(X_train, y_train, X_val, y_val)
        scored_candidates = validation_df.select("session", "candidate").with_columns(
            pl.Series("score", model.predict_proba(X_val)[:, 1])
        )
        predictions = candidate_lists(scored_candidates, score_column="score", top_k=20)
        recalls[objective] = recall_at_k(predictions, validation_ground_truth, target_column)
        print(f"XGBoost {objective} Recall@20:", recalls[objective])

    recalls["weighted"] = 0.10 * recalls["clicks"] + 0.30 * recalls["carts"] + 0.60 * recalls["orders"]
    print("Weighted OTTO score:", recalls["weighted"])


if __name__ == "__main__":
    main()
