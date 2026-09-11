import polars as pl


TARGET_COLUMNS = {
    "clicks": "click_target",
    "carts": "cart_targets",
    "orders": "order_targets",
}


def candidate_lists(candidates: pl.DataFrame, score_column: str | None = None, top_k: int | None = None) -> pl.DataFrame:
    if score_column is None:
        return candidates.group_by("session").agg(pl.col("candidate").unique().alias("predictions"))

    ranked = candidates.sort(["session", score_column, "candidate"], descending=[False, True, False])
    predictions = ranked.group_by("session", maintain_order=True).agg(pl.col("candidate").alias("predictions"))
    return predictions.with_columns(pl.col("predictions").list.head(top_k) if top_k is not None else pl.col("predictions"))


def recall_at_k(predictions: pl.DataFrame, ground_truth: pl.DataFrame, target_column: str, k: int | None = 20) -> float:
    total_hits = 0
    total_targets = 0
    evaluation = ground_truth.join(predictions, on="session", how="left")

    for target, predicted in evaluation.select(target_column, "predictions").iter_rows():
        if target is None:
            continue

        targets = target if isinstance(target, list) else [target]
        targets = set(targets)
        if not targets:
            continue

        predicted = [] if predicted is None else predicted
        if k is not None:
            predicted = predicted[:k]
        total_hits += len(set(predicted) & targets)
        total_targets += min(20, len(targets))

    return total_hits / total_targets if total_targets else 0.0


def evaluate_predictions(predictions: pl.DataFrame, ground_truth: pl.DataFrame, k: int | None = 20) -> dict[str, float]:
    scores = {objective: recall_at_k(predictions, ground_truth, target_column, k) for objective, target_column in TARGET_COLUMNS.items()}
    scores["weighted"] = 0.10 * scores["clicks"] + 0.30 * scores["carts"] + 0.60 * scores["orders"]
    return scores
