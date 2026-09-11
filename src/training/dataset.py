import polars as pl
from src.features.pipeline import build_features
from src.training.labels import add_labels


def build_training_dataset(candidates: pl.DataFrame, events: pl.DataFrame, ground_truth: pl.DataFrame, item_events: pl.DataFrame | None = None) -> pl.DataFrame:
    df = build_features(candidates, events, item_events)
    df = add_labels(df, ground_truth)
    return df
