import polars as pl

def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("session_event_count") == 1)
        .then(1.0)
        .otherwise(pl.col("last_position") / (pl.col("session_event_count") - 1))
        .alias("last_position_ratio"),

        pl.when(pl.col("global_event_count") > 0)
        .then(pl.col("global_cart_count") / pl.col("global_event_count"))
        .otherwise(0.0)
        .alias("item_cart_rate"),

        pl.when(pl.col("global_event_count") > 0)
        .then(pl.col("global_order_count") / pl.col("global_event_count"))
        .otherwise(0.0)
        .alias("item_order_rate"),

        pl.when(pl.col("session_event_count") > 0)
        .then(1 - pl.col("session_unique_items") / pl.col("session_event_count"))
        .otherwise(0.0)
        .alias("session_repeat_ratio")
    )

    return df