import polars as pl


def add_labels(candidates: pl.DataFrame, ground_truth: pl.DataFrame) -> pl.DataFrame:
    df = candidates.join(ground_truth, on="session", how="left")

    df = df.with_columns(
        (pl.col("candidate") == pl.col("click_target")).cast(pl.Int8).alias("click_label"),

        pl.col("cart_targets")
        .list.contains(pl.col("candidate"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("cart_label"),

        pl.col("order_targets")
        .list.contains(pl.col("candidate"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("order_label")
    )

    return df.drop("click_target", "cart_targets", "order_targets")