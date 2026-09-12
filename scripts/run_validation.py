"""Train and evaluate the development recommender."""

import argparse

import polars as pl

from src.artifacts import save_artifacts
from src.features.item_features import build_item_features
from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.dataset import build_training_dataset
from src.training.evaluation import candidate_target_coverage, candidate_lists, oracle_recall_at_k, recall_at_k
from src.training.train_xgb import prepare_training_data, train_binary_model
from src.training.validation import build_ground_truth, split_by_session, split_observed_hidden


OBJECTIVES = {
    "clicks": ("click_label", "click_target"),
    "carts": ("cart_label", "cart_targets"),
    "orders": ("order_label", "order_targets"),
}
CANDIDATE_CONFIG = {"general_k": 50, "type_k": 20, "time_k": 20, "buy_k": 20}


def main(save_dir: str | None = None):
    events = pl.read_parquet("data/processed/train_dev.parquet")
    print("Splitting sessions into training and validation data...")
    train_events, validation_events = split_by_session(events)
    train_observed, train_hidden = split_observed_hidden(train_events)
    validation_observed, validation_hidden = split_observed_hidden(validation_events)
    train_ground_truth = build_ground_truth(train_hidden)
    validation_ground_truth = build_ground_truth(validation_hidden)
    item_features = build_item_features(train_observed)

    matrices = []
    for weighting in ["general", "type", "time", "buy_to_buy"]:
        print(f"Building {weighting} co-visitation matrix...")
        matrices.append(build_covisitation_matrix(train_observed, weighting=weighting))

    print("Generating candidates...")
    train_candidates = generate_candidates(train_observed, *matrices, **CANDIDATE_CONFIG)
    validation_candidates = generate_candidates(validation_observed, *matrices, **CANDIDATE_CONFIG)
    print("Validation candidates per session:", validation_candidates.height / validation_candidates["session"].n_unique())

    print("Building candidate features and labels...")
    training_df = build_training_dataset(train_candidates, train_observed, train_ground_truth, item_features=item_features)
    validation_df = build_training_dataset(
        validation_candidates,
        validation_observed,
        validation_ground_truth,
        item_features=item_features,
    )

    coverage = candidate_target_coverage(validation_candidates, validation_ground_truth)
    oracle = oracle_recall_at_k(validation_candidates, validation_ground_truth)
    print("Candidate target coverage:", coverage)
    print("Candidate oracle Recall@20:", oracle)
    recalls = {}
    models = {}
    for objective, (label_column, target_column) in OBJECTIVES.items():
        print(f"Training {objective} XGBoost model...")
        (X_train, y_train), (X_val, y_val) = prepare_training_data(training_df, validation_df, label_column)
        model = train_binary_model(X_train, y_train, X_val, y_val)
        models[objective] = model
        scored_candidates = validation_df.select("session", "candidate").with_columns(
            pl.Series("score", model.predict_proba(X_val)[:, 1])
        )
        predictions = candidate_lists(scored_candidates, score_column="score", top_k=20)
        recalls[objective] = recall_at_k(predictions, validation_ground_truth, target_column)
        print(f"XGBoost {objective} Recall@20:", recalls[objective])

    recalls["weighted"] = 0.10 * recalls["clicks"] + 0.30 * recalls["carts"] + 0.60 * recalls["orders"]
    print("Weighted OTTO score:", recalls["weighted"])
    if save_dir is not None:
        metrics = {"candidate_coverage": coverage, "candidate_oracle": oracle, "ranking": recalls}
        save_artifacts(save_dir, models, matrices, item_features, CANDIDATE_CONFIG, metrics)
        print(f"Saved models and retrieval artifacts to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir")
    args = parser.parse_args()
    main(args.save_dir)
