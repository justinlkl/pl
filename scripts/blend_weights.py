"""Compute blended predictions using LSTM, HGB (proxy for XGBoost), and LightGBM.

Usage:
  python scripts/blend_weights.py --artifacts artifacts/ensemble_run_lgbm --top 20
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.sequences import build_sequences, split_by_end_gw
from src.fpl_projection.preprocessing import fit_preprocessor_on_timesteps, transform_sequences
from src.fpl_projection.role_modeling import infer_role_from_window, position_to_role, build_feature_weight_vector
from src.fpl_projection.modeling import build_lstm_model
from src.fpl_projection.config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


def _apply_role_weights(X: np.ndarray, roles: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    uniq = sorted(set(str(r) for r in np.unique(roles)))
    role_to_w = {r: build_feature_weight_vector(feature_columns, r) for r in uniq}
    W = np.stack([role_to_w.get(str(r), np.ones(X.shape[-1])) for r in roles], axis=0)
    return X * W[:, None, :]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts/ensemble_run_lgbm"))
    p.add_argument("--season", type=str, default="2025-2026")
    p.add_argument("--repo-root", type=Path, default=Path('.'))
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root)
    artifacts = Path(args.artifacts)

    print('Loading raw data and building sequences...')
    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)
    feature_columns = [f for f in DEFAULT_FEATURE_COLUMNS if f in raw.columns]
    df_num = raw.copy()

    # Build sequences using defaults
    dataset = build_sequences(df=df_num, feature_columns=feature_columns, target_column='total_points', seq_length=5, horizon=DEFAULT_HORIZON)

    # infer roles per sequence
    pid_to_pos = {}
    if 'player_id' in raw.columns and 'position' in raw.columns:
        meta = raw.sort_values(['player_id','gw']).groupby('player_id', sort=False, as_index=False).tail(1)[['player_id','position']]
        pid_to_pos = {int(r['player_id']): r['position'] for _, r in meta.iterrows()}

    role_labels = []
    for i in range(dataset.X.shape[0]):
        pid = int(dataset.player_id[i])
        pos = pid_to_pos.get(pid)
        role = infer_role_from_window(pos, pd.DataFrame(dataset.X[i], columns=feature_columns), mid_split=False)
        if role is None:
            role = position_to_role(pos)
        role_labels.append(role)
    roles_for_seq = np.asarray(role_labels, dtype=object)

    # Fit preprocessor on train timesteps
    n_features = dataset.X.shape[-1]
    train_timesteps_2d = dataset.X.reshape(dataset.X.shape[0] * 5, n_features)
    pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

    X_all = transform_sequences(pipeline, dataset.X)
    X_all = _apply_role_weights(X_all, roles_for_seq, feature_columns)

    # load LSTM
    lstm_path = artifacts / 'lstm_model.keras'
    if lstm_path.exists():
        lstm = tf.keras.models.load_model(str(lstm_path), compile=False)
    else:
        # try weights fallback
        weights = artifacts / 'lstm_model_weights.h5'
        if not weights.exists():
            raise FileNotFoundError('No LSTM artifact found in artifacts dir')
        lstm = build_lstm_model(seq_length=5, num_features=n_features, horizon=DEFAULT_HORIZON)
        lstm.load_weights(str(weights))

    print('Computing LSTM predictions...')
    lstm_preds = lstm.predict(X_all, verbose=0)

    # Recreate tab (last-timestep tabular features)
    # Build base tab similar to ensemble_train
    keep = ['player_id', 'gw', 'web_name'] + feature_columns + ['total_points']
    base = df_num[keep].copy()
    base = base.sort_values(['gw','player_id']).reset_index(drop=True)
    base = base.rename(columns={'gw':'end_gw'})

    # Apply role weights to base features
    roles_row = []
    for _, r in base.iterrows():
        pos = r.get('position') if 'position' in r.index else None
        window = pd.DataFrame([r])
        role = infer_role_from_window(pos, window, mid_split=False)
        if role is None:
            role = position_to_role(pos)
        roles_row.append(role)
    roles_row_arr = np.asarray(roles_row, dtype=object)
    uniq = sorted(set(str(v) for v in np.unique(roles_row_arr)))
    role_to_w = {r: build_feature_weight_vector(feature_columns, r) for r in uniq}
    W = np.stack([role_to_w.get(str(r), np.ones(len(feature_columns))) for r in roles_row_arr], axis=0)
    base.loc[:, feature_columns] = base[feature_columns].to_numpy(dtype=float) * W

    seq_df = pd.DataFrame({
        'player_id': dataset.player_id,
        'end_gw': dataset.end_gw,
        'role': roles_for_seq,
    })
    for k in range(DEFAULT_HORIZON):
        seq_df[f'lstm_pred_h{k+1}'] = lstm_preds[:, k]

    tab = seq_df.merge(base, on=['player_id','end_gw'], how='left')
    tab = tab.sort_values(['end_gw','player_id']).reset_index(drop=True)
    tab = tab.dropna(subset=feature_columns).copy()

    # identify latest end_gw
    latest_gw = int(tab['end_gw'].max())
    print('Latest end_gw in tab:', latest_gw)

    # For each horizon, load stacking model and extract base preds
    blends = {
        'conservative': (0.50, 0.30, 0.20),
        'balanced': (0.40, 0.35, 0.25),
        'aggressive': (0.30, 0.40, 0.30),
    }

    results = {}
    for h in range(1, DEFAULT_HORIZON+1):
        model_path = artifacts / f'stack_h{h}.joblib'
        if not model_path.exists():
            print(f'stack model for horizon {h} not found: {model_path}')
            continue
        model = joblib.load(model_path)
        # Determine feature columns used by stacking
        # feature columns used by stacking: feature_columns + lstm_pred_h{h}
        feature_cols = [c for c in feature_columns if c in tab.columns]
        feature_cols.append(f'lstm_pred_h{h}')
        Xall = tab[feature_cols].to_numpy(dtype=float)

        # compute base estimator preds
        base_preds = {}
        # If sklearn StackingRegressor: model.estimators_ is list of fitted estimators
        ests = getattr(model, 'estimators_', None)
        if ests is None:
            # fallback: try named_estimators_
            ests = []
            for name, est in getattr(model, 'named_estimators_', {}).items():
                ests.append(est)

        for est in ests:
            try:
                name = est.__class__.__name__
                pred = est.predict(Xall)
                base_preds[name] = pred
            except Exception:
                # skip
                pass

        # Identify HGB and LGBM preds
        hgb_pred = None
        lgb_pred = None
        for kname, pred in base_preds.items():
            if 'HistGradient' in kname or 'Gradient' in kname:
                hgb_pred = pred
            if 'LGBM' in kname or 'LightGBM' in kname:
                lgb_pred = pred
        # Fall back: if only one base present, use it for both
        if hgb_pred is None and len(base_preds) > 0:
            # pick first
            hgb_pred = list(base_preds.values())[0]
        if lgb_pred is None and len(base_preds) > 1:
            lgb_pred = list(base_preds.values())[-1]
        if lgb_pred is None:
            lgb_pred = np.zeros(len(Xall))
        if hgb_pred is None:
            hgb_pred = np.zeros(len(Xall))

        lstm_col = tab[f'lstm_pred_h{h}'].to_numpy(dtype=float)

        # compute blends
        blends_out = {}
        for key, (w_lstm, w_xgb, w_lgb) in blends.items():
            final = w_lstm * lstm_col + w_xgb * hgb_pred + w_lgb * lgb_pred
            blends_out[key] = final
        results[h] = blends_out

    # For horizon=1, show top N for latest end_gw
    h = 1
    latest_idx = tab['end_gw'] == latest_gw
    out_rows = []
    for key in blends.keys():
        arr = results[h][key]
        subset = tab[latest_idx].copy()
        subset['blend_pred'] = arr[latest_idx]
        top = subset.sort_values('blend_pred', ascending=False).head(args.top)[['player_id','web_name','end_gw','blend_pred']]
        out_rows.append((key, top))

    for key, df in out_rows:
        print('\n=== Blend:', key, '===')
        print(df.to_string(index=False))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
