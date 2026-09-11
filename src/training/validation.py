import numpy as np
import polars as pl


def split_by_session(events: pl.DataFrame, val_size: float = 0.2, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    sessions = events.select("session").unique().sort("session").to_series().to_numpy()
    sessions = np.random.default_rng(seed).permutation(sessions)
    split_idx = int(len(sessions) * (1 - val_size))

    train_sessions = sessions[:split_idx]
    validation_sessions = sessions[split_idx:]
    return (
        events.filter(pl.col("session").is_in(train_sessions)),
        events.filter(pl.col("session").is_in(validation_sessions)),
    )


def split_observed_hidden(events: pl.DataFrame, observed_fraction: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0 < observed_fraction < 1:
        raise ValueError("observed_fraction must be between 0 and 1")

    positioned = (
        events.sort(["session", "ts"])
        .with_columns(
            pl.int_range(pl.len()).over("session").alias("event_pos"),
            pl.len().over("session").alias("session_len"),
        )
        .with_columns((pl.col("event_pos") < (pl.col("session_len") * observed_fraction).floor()).alias("is_observed"))
    )
    return positioned.filter(pl.col("is_observed")).drop("event_pos", "session_len", "is_observed"), positioned.filter(~pl.col("is_observed")).drop("event_pos", "session_len", "is_observed")


def build_ground_truth(hidden_events: pl.DataFrame) -> pl.DataFrame:
    return hidden_events.group_by("session").agg(
        pl.col("aid").filter(pl.col("type") == "clicks").first().alias("click_target"),
        pl.col("aid").filter(pl.col("type") == "carts").unique().alias("cart_targets"),
        pl.col("aid").filter(pl.col("type") == "orders").unique().alias("order_targets"),
    )
