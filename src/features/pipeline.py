import polars as pl

from src.features.session_features import add_session_features
from src.features.item_features import add_item_features
from src.features.session_aggregate_features import add_session_aggregate_features
from src.features.derived_features import add_derived_features

def build_features(candidates: pl.DataFrame, events: pl.DataFrame, item_events: pl.DataFrame | None = None, item_features: pl.DataFrame | None = None) -> pl.DataFrame:
    item_events = events if item_events is None else item_events
    df = add_session_features(candidates, events)
    df = add_item_features(df, item_events, item_features)
    df = add_session_aggregate_features(df, events)
    df = add_derived_features(df)
    return df
