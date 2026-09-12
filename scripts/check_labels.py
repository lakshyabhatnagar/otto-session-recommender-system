"""Check prototype labels on real development data: python -m scripts.check_labels."""

import polars as pl

from src.retrieval.candidates import generate_candidates
from src.retrieval.covisitation import build_covisitation_matrix
from src.training.labels import add_labels
from src.training.validation import split_by_session


def main():
    events = pl.read_parquet("data/processed/train_dev.parquet").sort(["session", "ts"], maintain_order=True)
    events = events.with_columns(
        pl.int_range(pl.len()).over("session").alias("event_pos"),
        pl.len().over("session").alias("session_len"),
    )
    observed = events.filter(pl.col("event_pos") < (pl.col("session_len") * 0.7).floor())
    hidden = events.filter(pl.col("event_pos") >= (pl.col("session_len") * 0.7).floor())
    ground_truth = hidden.group_by("session").agg(
        pl.col("aid").filter(pl.col("type") == "clicks").first().alias("click_target"),
        pl.col("aid").filter(pl.col("type") == "carts").unique().alias("cart_targets"),
        pl.col("aid").filter(pl.col("type") == "orders").unique().alias("order_targets"),
    )

    # Reproduce prototype retrieval for label checks only; this is not model validation.
    matrices = []
    for weighting in ["general", "type", "time", "buy_to_buy"]:
        print("Building matrix:", weighting, flush=True)
        matrices.append(build_covisitation_matrix(observed, weighting=weighting))
    print("Generating candidates", flush=True)
    candidates = generate_candidates(observed, *matrices)
    labeled = add_labels(candidates, ground_truth)
    labels = ["click_label", "cart_label", "order_label"]
    null_counts = labeled.select(labels).null_count()
    print("Label null counts:", null_counts.to_dicts()[0])
    assert null_counts.row(0) == (0, 0, 0)
    assert labeled.height == candidates.height
    for label in labels:
        assert labeled[label].is_in([0, 1]).all()
    no_click = ground_truth.filter(pl.col("click_target").is_null()).select("session")
    no_click_rows = labeled.join(no_click, on="session", how="inner")
    assert no_click_rows.height > 0, "Need real sessions without a hidden click to exercise the bug"
    assert no_click_rows["click_label"].sum() == 0
    print("Rows without a hidden click:", no_click_rows.height)
    train, validation = split_by_session(labeled)
    print("Total candidate rows:", labeled.height)
    print("Train rows:", train.height)
    print("Validation rows:", validation.height)
    print("Positive clicks:", train["click_label"].sum())
    print("Positive click ratio:", train["click_label"].mean())


if __name__ == "__main__":
    main()
