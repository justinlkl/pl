from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessArtifacts:
    feature_columns: list[str]
    pipeline: Pipeline

    def save(self, path: str) -> None:
        dump({"feature_columns": self.feature_columns, "pipeline": self.pipeline}, path)

    @staticmethod
    def load(path: str) -> "PreprocessArtifacts":
        obj = load(path)
        return PreprocessArtifacts(feature_columns=obj["feature_columns"], pipeline=obj["pipeline"])


def select_and_coerce_numeric(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    missing = [c for c in [*feature_columns, target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[["player_id", "gw", "web_name", *feature_columns, target_column]].copy()

    for col in feature_columns + [target_column]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def make_preprocess_pipeline() -> Pipeline:
    # Median imputation + standard scaling.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def fit_preprocessor_on_timesteps(
    timesteps_2d: np.ndarray,
) -> Pipeline:
    """Fit preprocessing on 2D array (n_timesteps, n_features)."""
    pipeline = make_preprocess_pipeline()
    pipeline.fit(timesteps_2d)
    return pipeline


def transform_sequences(pipeline: Pipeline, X_3d: np.ndarray) -> np.ndarray:
    """Apply a 2D preprocessing pipeline to 3D sequence data."""
    n_samples, seq_len, n_features = X_3d.shape
    flat = X_3d.reshape(n_samples * seq_len, n_features)
    flat_t = pipeline.transform(flat)
    return flat_t.reshape(n_samples, seq_len, n_features)
