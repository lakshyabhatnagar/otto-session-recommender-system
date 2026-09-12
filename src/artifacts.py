import json
from pathlib import Path

import polars as pl
from xgboost import XGBClassifier

from src.training.train_xgb import FEATURES


MATRIX_NAMES = ["general", "type", "time", "buy_to_buy"]


def save_artifacts(output_dir: str | Path, models: dict[str, XGBClassifier], matrices: list[pl.DataFrame], item_features: pl.DataFrame, candidate_config: dict[str, int], metrics: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for objective, model in models.items():
        model.save_model(str(output_dir / f"{objective}_model.json"))
    for name, matrix in zip(MATRIX_NAMES, matrices, strict=True):
        matrix.write_parquet(output_dir / f"{name}_matrix.parquet")
    item_features.write_parquet(output_dir / "item_features.parquet")
    metadata = {"features": FEATURES, "candidate_config": candidate_config, "metrics": metrics}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_artifacts(output_dir: str | Path) -> tuple[dict[str, XGBClassifier], list[pl.DataFrame], pl.DataFrame, dict]:
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    models = {}
    for objective in ["clicks", "carts", "orders"]:
        model = XGBClassifier()
        model.load_model(str(output_dir / f"{objective}_model.json"))
        models[objective] = model
    matrices = [pl.read_parquet(output_dir / f"{name}_matrix.parquet") for name in MATRIX_NAMES]
    return models, matrices, pl.read_parquet(output_dir / "item_features.parquet"), metadata
