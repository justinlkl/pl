#!/usr/bin/env python3
"""Full prediction pipeline with fixture difficulty, ensemble stacking, and uncertainty.

Usage:
    python scripts/predict_with_enhancements.py --season 2025-2026 --horizon 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpl_projection.config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.preprocessing import PreprocessArtifacts, select_and_coerce_numeric, transform_sequences
from src.fpl_projection.fixture_features import integrate_fixture_features
from src.fpl_projection.form_features import integrate_form_features
from src.fpl_projection.ensemble_stacker import EnsembleStacker
from src.fpl_projection.uncertainty_estimation import predict_with_uncertainty
from src.fpl_projection.role_modeling import (
    build_feature_weight_vector,
    infer_role_from_window,
    position_to_role,
    load_role_scaling,
    scale_projection_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate projections with fixture difficulty, ensemble stacking, and uncertainty"
    )
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mid-split", action="store_true", help="Split MID into MID_DM/MID_AM")
    parser.add_argument("--no-role-scaling", action="store_true", help="Disable role scaling")
    parser.add_argument("--use-ensemble", action="store_true", help="Use ensemble stacking if available")
    parser.add_argument("--uncertainty-simulations", type=int, default=50, help="Monte Carlo simulations for uncertainty")
    parser.add_argument("--with-fixture-features", action="store_true", help="Include fixture difficulty features")
    parser.add_argument("--with-form-features", action="store_true", help="Include form trend features")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (repo_root / "artifacts")
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (outputs_dir / "projections_enhanced.csv")
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED PROJECTION PIPELINE")
    print("=" * 80)

    # Load data
    print(f"\n[1/6] Loading data for {args.season}...")
    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season)
    print(f"  ✅ Loaded {len(raw)} records")

    # Add fixture features
    if args.with_fixture_features:
        print(f"\n[2a/6] Building fixture difficulty features...")
        try:
            raw = integrate_fixture_features(
                df=raw,
                repo_root=repo_root,
                season=args.season,
                graceful_fallback=True,
            )
            print(f"  ✅ Fixture features integrated")
        except Exception as e:
            print(f"  ⚠️  Fixture features failed (continuing without): {e}")
    else:
        print(f"\n[2a/6] Skipping fixture features (use --with-fixture-features to enable)")

    # Add form features
    if args.with_form_features:
        print(f"\n[2b/6] Building form trend features...")
        try:
            raw = integrate_form_features(
                df=raw,
                rolling_windows=[3, 5],
                metrics=['total_points'],
            )
            print(f"  ✅ Form features integrated (rolling windows: 3, 5)")
        except Exception as e:
            print(f"  ⚠️  Form features failed (continuing without): {e}")
    else:
        print(f"\n[2b/6] Skipping form features (use --with-form-features to enable)")

    # Load model and preprocessor
    print(f"\n[3/6] Loading LSTM model and preprocessor...")
    model = tf.keras.models.load_model(str(artifacts_dir / "model.keras"), compile=False)
    prep = PreprocessArtifacts.load(str(artifacts_dir / "preprocess.joblib"))
    print(f"  ✅ Model: {model.count_params():,} parameters")

    # Select features
    df = select_and_coerce_numeric(raw, prep.feature_columns, TARGET_COLUMN)
    last_gw = int(pd.to_numeric(df["gw"], errors="coerce").dropna().max())
    next_gws = list(range(last_gw + 1, last_gw + 1 + args.horizon))

    # Build sequences per player
    print(f"\n[4/6] Building sequences (seq_length={args.seq_length}, horizon={args.horizon})...")
    rows: list[dict] = []
    X_list: list[np.ndarray] = []
    roles: list[str] = []

    pid_to_pos: dict[int, object] = {}
    if "player_id" in raw.columns and "position" in raw.columns:
        _meta = (
            raw.sort_values(["player_id", "gw"])
            .groupby("player_id", sort=False, as_index=False)
            .tail(1)[["player_id", "position"]]
        )
        pid_to_pos = {int(r["player_id"]): r.get("position") for _, r in _meta.iterrows()}

    for player_id, g in df.sort_values(["player_id", "gw"]).groupby("player_id", sort=False):
        g = g.sort_values("gw")
        if len(g) < args.seq_length:
            continue

        window = g.iloc[-args.seq_length :]
        X_window = window[prep.feature_columns].to_numpy(dtype=float)
        if X_window.shape != (args.seq_length, len(prep.feature_columns)):
            continue

        X_list.append(X_window)
        rows.append({"player_id": int(player_id)})

        pos = pid_to_pos.get(int(player_id))
        role = infer_role_from_window(pos, window[prep.feature_columns], mid_split=args.mid_split)
        if role is None:
            role = position_to_role(pos)
        roles.append(str(role))

    if not X_list:
        raise ValueError("No players had enough history to build sequences.")

    X = np.stack(X_list, axis=0)
    X = transform_sequences(prep.pipeline, X)

    # Apply per-sample role weights
    uniq = sorted(set(roles))
    role_to_w = {r: build_feature_weight_vector(prep.feature_columns, r) for r in uniq}
    W = np.stack([role_to_w.get(r, np.ones(len(prep.feature_columns))) for r in roles], axis=0)
    X = X * W[:, None, :]

    print(f"  ✅ Built {len(X)} sequences")

    # Predict with LSTM
    print(f"\n[5/6] Generating predictions...")
    lstm_preds = model.predict(X, verbose=0)

    # Option: Ensemble stacking (if saved)
    if args.use_ensemble:
        ensemble_dir = artifacts_dir / "ensemble"
        if ensemble_dir.exists() and (ensemble_dir / "meta_model.joblib").exists():
            print(f"     Loading ensemble stacker...")
            try:
                stacker = EnsembleStacker.load(ensemble_dir)
                ensemble_preds = stacker.predict(X, model)
                print(f"  ✅ Ensemble predictions generated")
                preds = ensemble_preds
            except Exception as e:
                print(f"  ⚠️  Ensemble failed (falling back to LSTM): {e}")
                preds = lstm_preds
        else:
            print(f"  ℹ️  Ensemble not found; using LSTM only")
            preds = lstm_preds
    else:
        preds = lstm_preds

    # Uncertainty estimation
    print(f"     Computing uncertainty (n_simulations={args.uncertainty_simulations})...")
    uncertainty = predict_with_uncertainty(model, X, n_simulations=args.uncertainty_simulations)
    print(f"  ✅ Uncertainty bounds computed")

    # Attach metadata
    meta_cols = [c for c in ["player_id", "web_name", "team_code", "position"] if c in raw.columns]
    meta = (
        raw.sort_values(["player_id", "gw"])
        .groupby("player_id", sort=False, as_index=False)
        .tail(1)[meta_cols]
        .copy()
    )
    out = pd.DataFrame(rows).merge(meta, on="player_id", how="left")
    out["role"] = roles

    # Role scaling
    if not args.no_role_scaling:
        overrides = load_role_scaling(artifacts_dir / "role_scaling.json")
        preds = scale_projection_matrix(preds, out["role"].to_numpy(dtype=object), overrides=overrides)

    # Write output with uncertainty
    print(f"\n[6/6] Writing outputs...")
    for i, gw in enumerate(next_gws):
        col_base = f"GW{gw}"
        out[f"{col_base}_proj"] = preds[:, i]
        out[f"{col_base}_std"] = uncertainty["std"][:, i]
        out[f"{col_base}_lower_95"] = uncertainty["lower_95"][:, i]
        out[f"{col_base}_upper_95"] = uncertainty["upper_95"][:, i]

    out.to_csv(output_path, index=False)
    print(f"  ✅ Projections written to {output_path.relative_to(repo_root)}")

    # Summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Players projected: {len(out)}")
    print(f"Gameweeks: {next_gws}")
    print(f"\nTop 10 by GW{next_gws[0]}_proj:")
    top_10 = out.nlargest(10, f"GW{next_gws[0]}_proj")[
        ["player_id", "web_name", "position", "role", f"GW{next_gws[0]}_proj", f"GW{next_gws[0]}_std"]
    ]
    print(top_10.to_string(index=False))

    print(f"\n✅ Run complete at {datetime.now().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
