#!/usr/bin/env python3
"""
R1.1 Conditions-of-Detection Experiments (Tables 13-14)

Controlled experiments answering "under what conditions does FI-Divergence-based
detection succeed or fail?" on UCI Adult.

E1  Feature-redundancy ablation (Table 13):
    Add a synthetic twin of the most-important numeric feature at controlled
    Pearson correlation rho in {none, 0.70, 0.90, 0.99}; inject a FIXED outlier
    corruption (20% severity) into the ORIGINAL target feature only; measure how
    FI Divergence (SHAP, JS) responds.

E2  Model-robustness / regularization sweep (Table 14):
    Vary regularization strength (XGBoost reg_lambda; RandomForest max_depth) under
    a FIXED outlier corruption; measure FI Divergence.

E3  Redundancy quantification (descriptive):
    Report the inter-feature Pearson correlation structure of the UCI Adult numeric
    features.

Design: outlier 20%, seed=42, 10-fold stratified CV, 95% CI (t-distribution).
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, MODEL_CONFIGS
from data_loader import prepare_dataset, NUMERICAL_COLS
from quality_simulator import inject_outliers
from xai_analyzer import analyze_xai
from metrics import calculate_js_divergence, calculate_rank_correlation

# --- experiment constants ---
RHO_LEVELS = [None, 0.70, 0.90, 0.99]     # None = no twin (baseline redundancy)
SEVERITY = 0.20                            # fixed outlier severity
N_FOLDS = 10
SHAP_N = 100                               # SHAP subsample (same as main experiments)
XGB_LAMBDA = [0.0, 1.0, 10.0, 100.0]       # L2 regularization sweep
RF_DEPTH = [3, 6, 10, None]                # shallower tree = more regularized/robust


def ci95(values):
    """Mean and 95% CI half-width via t-distribution."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return (float(a.mean()) if n else float('nan'), float('nan'))
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return float(m), float(h)


def make_twin(target_col, rho, rng):
    """Synthetic feature with Pearson correlation ~rho to target_col."""
    x = target_col.values.astype(float)
    z = (x - x.mean()) / (x.std() + 1e-12)
    noise = rng.standard_normal(len(z))
    twin = rho * z + np.sqrt(max(0.0, 1.0 - rho ** 2)) * noise
    return pd.Series(twin, index=target_col.index)


def build_model(model_type, **overrides):
    cfg = dict(MODEL_CONFIGS[model_type])
    cfg.update(overrides)
    if model_type == 'random_forest':
        return RandomForestClassifier(**cfg)
    return XGBClassifier(**cfg)


def fi_divergence_one_fold(model_type, X_tr, y_tr, X_va, target_feature,
                           model_overrides=None):
    """Train clean & corrupted models on one fold, return (js, spearman)."""
    model_overrides = model_overrides or {}

    m_clean = build_model(model_type, **model_overrides)
    m_clean.fit(X_tr, y_tr)
    fi_clean = analyze_xai(m_clean, X_tr, X_va, method='shap', n_samples=SHAP_N)

    X_tr_c = inject_outliers(X_tr, [target_feature], SEVERITY, random_seed=RANDOM_SEED)
    X_va_c = inject_outliers(X_va, [target_feature], SEVERITY, random_seed=RANDOM_SEED)
    m_corr = build_model(model_type, **model_overrides)
    m_corr.fit(X_tr_c, y_tr)
    fi_corr = analyze_xai(m_corr, X_tr_c, X_va_c, method='shap', n_samples=SHAP_N)

    js = calculate_js_divergence(fi_clean, fi_corr)
    rho_s, _ = calculate_rank_correlation(fi_clean, fi_corr, method='spearman')
    return float(js), float(rho_s)


def pick_target_feature(X_train, y_train):
    """Most important numeric feature on a clean RandomForest baseline (SHAP)."""
    m = build_model('random_forest')
    m.fit(X_train, y_train)
    fi = analyze_xai(m, X_train, X_train, method='shap', n_samples=SHAP_N)
    numeric_fi = fi[[c for c in NUMERICAL_COLS if c in fi.index]]
    return numeric_fi.sort_values(ascending=False).index[0]


def run_e1_redundancy(data, target):
    """E1: FI Divergence vs feature-redundancy level."""
    print(f"\n[E1] Redundancy ablation (target feature = '{target}')")
    rng = np.random.default_rng(RANDOM_SEED)
    X = data['X_train']
    y = data['y_train']
    rows = []
    for rho in RHO_LEVELS:
        if rho is None:
            X_aug = X.copy()
        else:
            X_aug = X.copy()
            X_aug[f'{target}__twin'] = make_twin(X[target], rho, rng)
        for model_type in ('random_forest', 'xgboost'):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            js_folds, sp_folds = [], []
            for tr_idx, va_idx in skf.split(X_aug, y):
                X_tr, X_va = X_aug.iloc[tr_idx], X_aug.iloc[va_idx]
                y_tr = y.iloc[tr_idx]
                js, sp = fi_divergence_one_fold(model_type, X_tr, y_tr, X_va, target)
                js_folds.append(js)
                sp_folds.append(sp)
            m, h = ci95(js_folds)
            sm, _ = ci95(sp_folds)
            label = 'baseline(no twin)' if rho is None else f'{rho:.2f}'
            rows.append({'experiment': 'E1_redundancy', 'rho': (None if rho is None else rho),
                         'rho_label': label, 'model': model_type,
                         'js_mean': m, 'js_ci95': h, 'spearman_mean': sm,
                         'n_folds': N_FOLDS, 'target_feature': target,
                         'corruption': 'outlier', 'severity': SEVERITY})
            print(f"    rho={label:18s} {model_type:14s} JS={m:.4f} ± {h:.4f}")
    return rows


def run_e2_robustness(data, target):
    """E2: FI Divergence vs model regularization strength."""
    print("\n[E2] Regularization / robustness sweep")
    X, y = data['X_train'], data['y_train']
    rows = []
    sweeps = [('xgboost', 'reg_lambda', XGB_LAMBDA),
              ('random_forest', 'max_depth', RF_DEPTH)]
    for model_type, param, levels in sweeps:
        for lv in levels:
            overrides = {param: lv}
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            js_folds = []
            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr = y.iloc[tr_idx]
                js, _ = fi_divergence_one_fold(model_type, X_tr, y_tr, X_va, target,
                                               model_overrides=overrides)
                js_folds.append(js)
            m, h = ci95(js_folds)
            rows.append({'experiment': 'E2_robustness', 'model': model_type,
                         'param': param, 'level': (None if lv is None else lv),
                         'js_mean': m, 'js_ci95': h, 'n_folds': N_FOLDS,
                         'target_feature': target, 'corruption': 'outlier',
                         'severity': SEVERITY})
            print(f"    {model_type:14s} {param}={str(lv):6s} JS={m:.4f} ± {h:.4f}")
    return rows


def run_e3_correlation(data):
    """E3: numeric-feature redundancy quantification (descriptive)."""
    print("\n[E3] Numeric-feature correlation structure (UCI Adult)")
    num = [c for c in NUMERICAL_COLS if c in data['X_train'].columns]
    corr = data['X_train'][num].corr().abs()
    off = corr.where(~np.eye(len(corr), dtype=bool))
    max_abs = float(np.nanmax(off.values))
    mean_abs = float(np.nanmean(off.values))
    pair = off.stack().idxmax()
    print(f"    numeric features: {num}")
    print(f"    max |corr| = {max_abs:.3f} ({pair[0]} ~ {pair[1]}), mean |corr| = {mean_abs:.3f}")
    return {'experiment': 'E3_correlation', 'numeric_features': num,
            'max_abs_corr': max_abs, 'mean_abs_corr': mean_abs,
            'max_pair': list(pair), 'corr_matrix': corr.round(4).to_dict()}


def main():
    print("=" * 64)
    print("R1.1 Conditions-of-Detection Experiments (UCI Adult)")
    print("=" * 64)
    np.random.seed(RANDOM_SEED)

    data = prepare_dataset()
    print(f"X_train: {data['X_train'].shape} | features: {len(data['feature_names'])}")

    target = pick_target_feature(data['X_train'], data['y_train'])
    e1 = run_e1_redundancy(data, target)
    e2 = run_e2_robustness(data, target)
    e3 = run_e3_correlation(data)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(e1 + e2)
    csv_path = RESULTS_DIR / f'redundancy_robustness_results_{ts}.csv'
    json_path = RESULTS_DIR / f'redundancy_robustness_results_{ts}.json'
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump({'meta': {'dataset': 'UCI Adult', 'seed': RANDOM_SEED,
                            'severity': SEVERITY, 'n_folds': N_FOLDS,
                            'target_feature': target, 'timestamp': ts},
                   'E1_redundancy': e1, 'E2_robustness': e2, 'E3_correlation': e3},
                  f, indent=2)

    # figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        e1df = pd.DataFrame(e1)
        for mt, g in e1df.groupby('model'):
            xs = list(range(len(g)))
            ax[0].errorbar(xs, g['js_mean'], yerr=g['js_ci95'], marker='o', capsize=3, label=mt)
            ax[0].set_xticks(xs)
            ax[0].set_xticklabels(g['rho_label'], rotation=20)
        ax[0].set_title('E1: FI Divergence vs feature redundancy')
        ax[0].set_xlabel('twin correlation rho'); ax[0].set_ylabel('JS Divergence'); ax[0].legend()
        e2df = pd.DataFrame(e2)
        for mt, g in e2df.groupby('model'):
            xs = list(range(len(g)))
            ax[1].errorbar(xs, g['js_mean'], yerr=g['js_ci95'], marker='s', capsize=3,
                           label=f"{mt} ({g['param'].iloc[0]})")
            ax[1].set_xticks(xs)
            ax[1].set_xticklabels([str(v) for v in g['level']], rotation=20)
        ax[1].set_title('E2: FI Divergence vs regularization')
        ax[1].set_xlabel('regularization level'); ax[1].set_ylabel('JS Divergence'); ax[1].legend()
        fig.tight_layout()
        fig_path = FIGURES_DIR / f'redundancy_robustness_{ts}.png'
        fig.savefig(fig_path, dpi=150)
        print(f"\nFigure: {fig_path}")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("=" * 64)
    print("DONE")


if __name__ == '__main__':
    main()
