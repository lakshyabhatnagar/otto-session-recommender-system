import polars as pl


def add_session_features(candidates: pl.DataFrame, session_events: pl.DataFrame) -> pl.DataFrame:
    events = (
        session_events
        .sort(["session", "ts"])
        .with_columns(
            pl.int_range(0, pl.len()).over("session").alias("position")
        )
    )

    features = (
        events
        .group_by(["session", "aid"])
        .agg(
            pl.len().alias("event_count"),
            (pl.col("type") == "clicks").sum().alias("click_count"),
            (pl.col("type") == "carts").sum().alias("cart_count"),
            (pl.col("type") == "orders").sum().alias("order_count"),
            pl.col("position").max().alias("last_position"),
            pl.col("ts").max().alias("last_ts")
        )
        .rename({"aid": "candidate"})
    )

    df = candidates.join(features, on=["session", "candidate"], how="left")

    count_cols = ["event_count", "click_count", "cart_count", "order_count"]

    df = df.with_columns(
        [pl.col(col).fill_null(0) for col in count_cols]
    )

    df = df.with_columns(
        (pl.col("click_count") > 0).cast(pl.Int8).alias("was_clicked"),
        (pl.col("cart_count") > 0).cast(pl.Int8).alias("was_carted"),
        (pl.col("order_count") > 0).cast(pl.Int8).alias("was_ordered")
    )
    session_last_ts = (
        events
        .group_by("session")
        .agg(pl.col("ts").max().alias("session_last_ts"))
    )

    df = df.join(session_last_ts, on="session", how="left")

    df = df.with_columns(
        (pl.col("session_last_ts") - pl.col("last_ts")).alias("time_since_last_interaction")
    )

    return df