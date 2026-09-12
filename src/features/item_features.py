import polars as pl


def build_item_features(events: pl.DataFrame) -> pl.DataFrame:
    return (
        events
        .group_by("aid")
        .agg(
            pl.len().alias("global_event_count"),
            (pl.col("type") == "clicks").sum().alias("global_click_count"),
            (pl.col("type") == "carts").sum().alias("global_cart_count"),
            (pl.col("type") == "orders").sum().alias("global_order_count"),
            pl.col("session").n_unique().alias("unique_sessions")
        )
        .rename({"aid": "candidate"})
    )


def add_item_features(candidates: pl.DataFrame, events: pl.DataFrame | None = None, item_features: pl.DataFrame | None = None) -> pl.DataFrame:
    if item_features is None:
        if events is None:
            raise ValueError("events or item_features is required")
        item_features = build_item_features(events)
    return candidates.join(item_features, on="candidate", how="left")
