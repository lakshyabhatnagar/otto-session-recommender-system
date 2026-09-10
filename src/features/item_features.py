import polars as pl

def add_item_features(candidates: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    item_features = (
        events
        .group_by("aid")
        .agg(
            pl.len().alias("global_event_count"),
            (pl.col("type") == "clicks").sum().alias("global_click_count"),
            (pl.col("type") == "carts").sum().alias("global_cart_count"),
            (pl.col("type") == "orders").sum().alias("global_order_count"),
            pl.col("session").n_unique().alias("unique_sessions")
        )
    )
    item_features = item_features.rename({"aid": "candidate"})
    df=candidates.join(item_features, on="candidate", how="left")
    return df