import polars as pl

def add_session_aggregate_features(candidates: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    session_features = (
        events
        .group_by("session")
        .agg(
            pl.len().alias("session_event_count"),
            pl.col("aid").n_unique().alias("session_unique_items"),
            (pl.col("type") == "clicks").sum().alias("session_click_count"),
            (pl.col("type") == "carts").sum().alias("session_cart_count"),
            (pl.col("type") == "orders").sum().alias("session_order_count")
        )
    )

    return candidates.join(session_features, on="session", how="left")