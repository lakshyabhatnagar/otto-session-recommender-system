from pathlib import Path

import polars as pl

from src.artifacts import load_artifacts
from src.features.pipeline import build_features
from src.retrieval.candidates import generate_candidates
from src.training.evaluation import candidate_lists


def recommend(session_events: pl.DataFrame, artifact_dir: str | Path, top_k: int = 20) -> dict[str, pl.DataFrame]:
    models, matrices, item_features, metadata = load_artifacts(artifact_dir)
    candidates = generate_candidates(session_events, *matrices, **metadata["candidate_config"])
    if candidates.is_empty():
        return {objective: pl.DataFrame({"session": [], "predictions": []}) for objective in models}

    features = build_features(candidates, session_events, item_features=item_features)
    feature_frame = features.select(metadata["features"]).to_pandas()
    predictions = {}

    for objective, model in models.items():
        scored = features.select("session", "candidate").with_columns(
            pl.Series("score", model.predict_proba(feature_frame)[:, 1])
        )
        predictions[objective] = candidate_lists(scored, score_column="score", top_k=top_k)

    return predictions
