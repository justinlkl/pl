from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_bins(pred: np.ndarray, actual: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Equal-frequency calibration bins for regression.

    Returns a DataFrame with predicted/actual means per bin. Used for an ECE-style
    calibration error estimate (weighted mean absolute bin gap).
    """

    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[mask]
    actual = actual[mask]
    if pred.size == 0:
        return pd.DataFrame(columns=["bin", "count", "pred_mean", "actual_mean", "pred_min", "pred_max"])

    q = np.linspace(0, 1, int(n_bins) + 1)
    edges = np.quantile(pred, q)
    edges = np.unique(edges)
    if edges.size < 3:
        return pd.DataFrame(
            {
                "bin": [0],
                "count": [int(pred.size)],
                "pred_mean": [float(np.mean(pred))],
                "actual_mean": [float(np.mean(actual))],
                "pred_min": [float(np.min(pred))],
                "pred_max": [float(np.max(pred))],
            }
        )

    bin_ids = np.digitize(pred, edges[1:-1], right=False)
    rows: list[dict] = []
    for b in range(int(np.max(bin_ids)) + 1):
        m = bin_ids == b
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": int(b),
                "count": int(np.sum(m)),
                "pred_mean": float(np.mean(pred[m])),
                "actual_mean": float(np.mean(actual[m])),
                "pred_min": float(np.min(pred[m])),
                "pred_max": float(np.max(pred[m])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fpl_model(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    player_id: np.ndarray,
    roles: np.ndarray,
    top_n: int = 50,
    n_calibration_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute ranking-centric diagnostics for FPL.

    Metrics are computed at player-level by aggregating sequence-level targets and
    predictions per player.

    Returns:
        metrics: Flat dict with overall metrics and per-role MAE entries.
        calib_bins: Calibration bins on player-level totals.
        per_role: Per-role MAE table on player-level totals.
    """

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        mask = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(mask)) < 3:
            return float("nan")
        ra = pd.Series(a[mask]).rank(method="average").to_numpy(dtype=float)
        rb = pd.Series(b[mask]).rank(method="average").to_numpy(dtype=float)
        ra = ra - float(np.mean(ra))
        rb = rb - float(np.mean(rb))
        denom = float(np.sqrt(np.sum(ra**2)) * np.sqrt(np.sum(rb**2)))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(ra * rb) / denom)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 2:
        true_total = np.sum(y_true, axis=1)
    else:
        true_total = y_true.reshape(y_true.shape[0], -1).sum(axis=1)
    if y_pred.ndim == 2:
        pred_total = np.sum(y_pred, axis=1)
    else:
        pred_total = y_pred.reshape(y_pred.shape[0], -1).sum(axis=1)

    df = pd.DataFrame(
        {
            "player_id": np.asarray(player_id).astype(int),
            "role": np.asarray(roles, dtype=object).astype(str),
            "true_total": true_total,
            "pred_total": pred_total,
        }
    )

    def _mode(series: pd.Series) -> str:
        vc = series.value_counts(dropna=False)
        return str(vc.index[0]) if not vc.empty else ""

    by_player = (
        df.groupby("player_id", as_index=False)
        .agg(
            true_total=("true_total", "mean"),
            pred_total=("pred_total", "mean"),
            role=("role", _mode),
        )
        .reset_index(drop=True)
    )

    overall_mae = float(np.mean(np.abs(by_player["pred_total"] - by_player["true_total"])))
    rank_corr = _spearman(by_player["pred_total"].to_numpy(), by_player["true_total"].to_numpy())

    k = int(min(max(int(top_n), 1), len(by_player)))
    top_true = set(by_player.nlargest(k, "true_total")["player_id"].astype(int).tolist())
    top_pred = set(by_player.nlargest(k, "pred_total")["player_id"].astype(int).tolist())
    top_k_recall = float(len(top_true & top_pred) / float(k)) if k > 0 else float("nan")

    calib = calibration_bins(
        by_player["pred_total"].to_numpy(),
        by_player["true_total"].to_numpy(),
        n_bins=int(n_calibration_bins),
    )
    if calib.empty:
        calib_error = float("nan")
        calib_error_rel = float("nan")
    else:
        weights = calib["count"].to_numpy(dtype=float)
        diffs = np.abs(calib["pred_mean"].to_numpy(dtype=float) - calib["actual_mean"].to_numpy(dtype=float))
        calib_error = float(np.sum(weights * diffs) / np.sum(weights))
        denom = float(np.mean(by_player["true_total"]))
        calib_error_rel = float(calib_error / denom) if denom > 0 else float("nan")

    per_role_rows: list[dict] = []
    for role, g in by_player.groupby("role"):
        per_role_rows.append(
            {
                "role": str(role),
                "count": int(len(g)),
                "mae": float(np.mean(np.abs(g["pred_total"] - g["true_total"]))),
            }
        )
    per_role = pd.DataFrame(per_role_rows).sort_values(["role"]).reset_index(drop=True)

    metrics: dict[str, float] = {
        "rank_correlation": float(rank_corr),
        f"top_{k}_recall": float(top_k_recall),
        "calibration_error": float(calib_error),
        "calibration_error_rel": float(calib_error_rel),
        "mae": float(overall_mae),
    }
    for _, r in per_role.iterrows():
        metrics[f"mae_{r['role']}"] = float(r["mae"])

    return metrics, calib, per_role
