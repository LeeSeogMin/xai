#!/usr/bin/env python3
"""
MLP Integrated-Gradients FI Divergence Experiment (Table 9 / H6)

Reproduces Table 9 (deep-learning FI Divergence via Integrated Gradients):

  - Train the MLP once on clean data (per CV fold), no retraining per corruption.
  - Compute baseline IG importance on a clean test-fold sample.
  - Inject each quality issue (missing/MCAR, outlier, distribution_shift) at each
    severity into the same test-fold sample, recompute IG importance, and measure
    FI Divergence (Jensen-Shannon) against the clean baseline.

Design: UCI Adult, MLP (Integrated Gradients, n_steps=50), 5-fold stratified CV,
3 issue types x 4 severities x 5 folds = 60 conditions, seed=42. Output saved to
results/mlp_ig_divergence_results_<timestamp>.json.
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from config import (RANDOM_SEED, RESULTS_DIR, SEVERITY_LEVELS,
                    QUALITY_ISSUE_TYPES, DL_EPOCHS, DL_BATCH_SIZE,
                    DL_LEARNING_RATE, IG_N_STEPS)
from data_loader import prepare_dataset, NUMERICAL_COLS
from quality_simulator import inject_missing_values, inject_outliers, inject_distribution_shift
from metrics import calculate_fi_divergence_metrics
from dl_models import TabularMLP, DLModelWrapper
from dl_xai import DLXAIAnalyzer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS_DL = 5
N_SAMPLE = 100      # IG attribution sample size

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def ig_importance(wrapper, df_sample, feature_names):
    """Integrated-Gradients global importance as a feature-indexed Series."""
    X_scaled = wrapper.scaler.transform(df_sample.values)
    analyzer = DLXAIAnalyzer(wrapper.model, device=DEVICE)
    imp = analyzer.get_feature_importance(X_scaled, method='integrated_gradients', n_steps=IG_N_STEPS)
    return pd.Series(np.asarray(imp, dtype=float), index=list(feature_names))


def corrupt(df, num_cols, issue_type, severity):
    if issue_type == 'missing':
        deg = inject_missing_values(df.copy(), num_cols, severity)
        return deg.fillna(deg.median())
    if issue_type == 'outlier':
        return inject_outliers(df.copy(), num_cols, severity)
    if issue_type == 'distribution_shift':
        return inject_distribution_shift(df.copy(), num_cols, severity, shift_type='covariate_shift')
    raise ValueError(issue_type)


def main():
    print("=" * 70)
    print("MLP Integrated-Gradients FI Divergence (Table 9 / H6) — UCI Adult, 5-fold CV")
    print(f"Device={DEVICE}  seed={RANDOM_SEED}  IG n_steps={IG_N_STEPS}")
    print("=" * 70)

    data = prepare_dataset(scale=False)  # corruption injected on raw numerical cols
    X_full = pd.concat([data['X_train'], data['X_test']], axis=0).reset_index(drop=True)
    y_full = pd.concat([pd.Series(data['y_train']), pd.Series(data['y_test'])], axis=0).reset_index(drop=True)
    feature_names = list(X_full.columns)
    num_cols = [c for c in NUMERICAL_COLS if c in X_full.columns]
    print(f"Data: X={X_full.shape}, numerical cols corrupted={num_cols}")

    skf = StratifiedKFold(n_splits=N_FOLDS_DL, shuffle=True, random_state=RANDOM_SEED)
    experiments = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_full, y_full)):
        X_tr, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx].values
        X_te = X_full.iloc[te_idx].reset_index(drop=True)
        print(f"\n[Fold {fold+1}/{N_FOLDS_DL}] train={len(X_tr)} test={len(X_te)}")

        wrapper = DLModelWrapper(
            model_class=TabularMLP,
            model_params={'hidden_dims': [128, 64, 32], 'dropout': 0.3},
            epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, learning_rate=DL_LEARNING_RATE,
            device=DEVICE, random_state=RANDOM_SEED, verbose=False,
        )
        wrapper.fit(X_tr, y_tr)

        y_te = y_full.iloc[te_idx].values
        sample = X_te.iloc[:N_SAMPLE]
        fi_base = ig_importance(wrapper, sample, feature_names)
        base_acc = accuracy_score(y_te, wrapper.predict(X_te.values))

        for issue in QUALITY_ISSUE_TYPES:
            for sev in SEVERITY_LEVELS:
                deg_full = corrupt(X_te, num_cols, issue, sev)
                deg_sample = deg_full.iloc[:N_SAMPLE]
                fi_corr = ig_importance(wrapper, deg_sample, feature_names)
                m = calculate_fi_divergence_metrics(fi_base, fi_corr)
                corr_acc = accuracy_score(y_te, wrapper.predict(deg_full.values))
                experiments.append({
                    'fold': fold, 'issue_type': issue, 'severity': sev,
                    'js_divergence': float(m['js_divergence']),
                    'spearman_correlation': float(m['spearman_correlation']),
                    'rank_change_ratio_top5': float(m['rank_change_ratio_top5']),
                    'accuracy_drop': float(base_acc - corr_acc),
                })
                print(f"  {issue:18s} {int(sev*100):>2d}%: JS={m['js_divergence']:.4f} accdrop={base_acc-corr_acc:+.4f}")

    # Aggregate per issue type over (4 severities x 5 folds = 20)
    df = pd.DataFrame(experiments)
    summary = {}
    for issue in QUALITY_ISSUE_TYPES:
        sdf = df[df.issue_type == issue]
        s = sdf['js_divergence']
        summary[issue] = {'mean': float(s.mean()), 'std': float(s.std()), 'n': int(s.size),
                          'accuracy_drop_mean': float(sdf['accuracy_drop'].mean())}
    overall = {'js_min': float(df.js_divergence.min()), 'js_max': float(df.js_divergence.max())}

    out = {
        'meta': {
            'timestamp': datetime.now().isoformat(), 'device': DEVICE, 'random_seed': RANDOM_SEED,
            'dataset': 'uci_adult', 'model': 'MLP', 'xai': 'integrated_gradients',
            'n_steps': IG_N_STEPS, 'n_folds': N_FOLDS_DL, 'sample_size': N_SAMPLE,
            'n_conditions': len(experiments),
            'methodology': 'train clean per fold; IG on clean vs corrupted test sample; no retraining',
        },
        'summary_table9': summary,
        'overall': overall,
        'experiments': experiments,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(RESULTS_DIR) / f'mlp_ig_divergence_results_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TABLE 9 (regenerated) — MLP FI Divergence via Integrated Gradients")
    for issue in QUALITY_ISSUE_TYPES:
        s = summary[issue]
        print(f"  {issue:18s}: {s['mean']:.4f} ± {s['std']:.4f}  acc_drop={s['accuracy_drop_mean']*100:+.2f}%  (n={s['n']})")
    print(f"  JS range: [{overall['js_min']:.4f}, {overall['js_max']:.4f}]")
    print(f"Saved -> {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
