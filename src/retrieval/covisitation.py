import polars as pl

DAY_MS = 24 * 60 * 60 * 1000
TYPE_WEIGHTS = {"clicks": 1.0, "carts": 6.0, "orders": 3.0}
VALID_WEIGHTINGS = {"general", "type", "time", "buy_to_buy"}


def build_covisitation_matrix(events: pl.DataFrame, weighting: str = "general", top_k: int = 50) -> pl.DataFrame:
    if weighting not in VALID_WEIGHTINGS:
        raise ValueError(f"Unknown weighting: {weighting}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    df = events

    if weighting == "buy_to_buy":
        df = df.filter(pl.col("type").is_in(["carts", "orders"]))

    df = (
        df.sort(["session", "ts"], descending=[False, True])
        .group_by("session", maintain_order=True)
        .head(30)
    )

    left = df.select(
        "session",
        pl.col("aid").alias("aid_x"),
        pl.col("ts").alias("ts_x"),
        pl.col("type").alias("type_x")
    )

    right = df.select(
        "session",
        pl.col("aid").alias("aid_y"),
        pl.col("ts").alias("ts_y"),
        pl.col("type").alias("type_y")
    )

    pairs = left.join(right, on="session", how="inner")

    pairs = pairs.filter(
        (pl.col("aid_x") != pl.col("aid_y")) &
        ((pl.col("ts_x") - pl.col("ts_y")).abs() < DAY_MS)
    )

    if weighting == "type":
        pairs = pairs.with_columns(
            pl.col("type_y").replace_strict(TYPE_WEIGHTS).alias("weight")
        )
        pairs = pairs.group_by(["session", "aid_x", "aid_y"]).agg(pl.col("weight").max().alias("weight"))

    elif weighting == "time":
        min_ts = events["ts"].min()
        max_ts = events["ts"].max()
        pairs = pairs.with_columns(pl.max_horizontal("ts_x", "ts_y").alias("pair_ts"))
        pairs = pairs.group_by(["session", "aid_x", "aid_y"]).agg(pl.col("pair_ts").max().alias("pair_ts"))
        if min_ts == max_ts:
            pairs = pairs.with_columns(pl.lit(1.0).alias("weight"))
        else:
            pairs = pairs.with_columns(
                (1 + 3 * (pl.col("pair_ts") - min_ts) / (max_ts - min_ts)).alias("weight")
            )

    else:
        pairs = pairs.unique(["session", "aid_x", "aid_y"], keep="first", maintain_order=True)
        pairs = pairs.with_columns(pl.lit(1.0).alias("weight"))

    matrix = (
        pairs.group_by(["aid_x", "aid_y"])
        .agg(pl.col("weight").sum().alias("score"))
        .sort(["aid_x", "score", "aid_y"], descending=[False, True, False])
        .group_by("aid_x", maintain_order=True)
        .head(top_k)
    )

    return matrix
