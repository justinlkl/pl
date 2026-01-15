from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    X: np.ndarray  # (n_samples, seq_len, n_features)
    y: np.ndarray  # (n_samples, horizon)
    player_id: np.ndarray  # (n_samples,)
    end_gw: np.ndarray  # (n_samples,) gw of the last timestep in the input window


def build_sequences(
    *,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seq_length: int,
    horizon: int,
) -> SequenceDataset:
    """Build (X, y) sequences per player.

    For each player and each time index t, use features from [t-seq_length+1..t]
    to predict targets [t+1..t+horizon].
    """

    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    pid_list: list[int] = []
    end_gw_list: list[int] = []

    for player_id, g in df.groupby("player_id", sort=False):
        g = g.sort_values("gw")
        feature_matrix = g[feature_columns].to_numpy(dtype=float)
        target_vector = g[target_column].to_numpy(dtype=float)
        gws = g["gw"].to_numpy(dtype=int)

        n = len(g)
        # t is the index of the last input timestep
        for t in range(seq_length - 1, n - horizon):
            X_window = feature_matrix[t - seq_length + 1 : t + 1]
            y_window = target_vector[t + 1 : t + 1 + horizon]

            if X_window.shape != (seq_length, len(feature_columns)):
                continue
            if y_window.shape != (horizon,):
                continue

            X_list.append(X_window)
            y_list.append(y_window)
            pid_list.append(int(player_id))
            end_gw_list.append(int(gws[t]))

    if not X_list:
        raise ValueError("No sequences could be built. Check seq_length/horizon and data coverage.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    return SequenceDataset(
        X=X,
        y=y,
        player_id=np.asarray(pid_list, dtype=int),
        end_gw=np.asarray(end_gw_list, dtype=int),
    )


def split_by_end_gw(
    dataset: SequenceDataset,
    *,
    train_max_end_gw: int,
    val_max_end_gw: int,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Time-based split by the gw of the last timestep in each input sequence."""

    train_mask = dataset.end_gw <= train_max_end_gw
    val_mask = (dataset.end_gw > train_max_end_gw) & (dataset.end_gw <= val_max_end_gw)
    test_mask = dataset.end_gw > val_max_end_gw

    def _subset(mask: np.ndarray) -> SequenceDataset:
        return SequenceDataset(
            X=dataset.X[mask],
            y=dataset.y[mask],
            player_id=dataset.player_id[mask],
            end_gw=dataset.end_gw[mask],
        )

    return _subset(train_mask), _subset(val_mask), _subset(test_mask)
