#!/usr/bin/env python3
'Full experiment runner that executes model, XAI, statistical, and baseline comparisons.'

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths
CODE_DIR = Path(__file__).parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from config import (
    RANDOM_SEED, RESULTS_DIR, MODELS_DIR, FIGURES_DIR,
    QUALITY_ISSUE_TYPES, SEVERITY_LEVELS,
    N_FOLDS, CONFIDENCE_LEVEL,
    DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE, DEVICE
)

# Local data and quality simulation imports
from data_loader import prepare_dataset, NUMERICAL_COLS, CATEGORICAL_COLS
from quality_simulator import inject_missing_values, inject_outliers, inject_distribution_shift

# Model and experiment imports
from models import create_model, train_model, evaluate_model
from xai_analyzer import analyze_xai, SHAPAnalyzer
from metrics import calculate_fi_divergence_metrics
from statistical_validation import StatisticalValidator
from traditional_baselines import TraditionalQualityDetector, BaselineComparator

# Deep Learning imports
import torch
from dl_models import TabularMLP, TabularAttentionNet, DLModelWrapper
from dl_xai import DLXAIAnalyzer

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def run_baseline_experiments():
    'Run baseline experiments.'
    print("\n" + "=" * 70)
    print("Step 1: Baseline Experiments (RF, XGBoost + SHAP)")
    print("=" * 70)


    print("\n[Loading data...]")
    data = prepare_dataset(scale=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")

    results = {
        'dataset': 'uci_adult',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'models': {}
    }


    for model_name in ['random_forest', 'xgboost']:
        print(f"\n[{model_name.upper()}]")


        model = create_model(model_name)
        model = train_model(model, X_train, y_train)


        metrics = evaluate_model(model, X_test, y_test)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['roc_auc']:.4f}")


        print("  Running SHAP analysis...")
        shap_importance = analyze_xai(model, X_train, X_test, method='shap', n_samples=100)
        fi_dict = shap_importance.to_dict()

        results['models'][model_name] = {
            'metrics': metrics,
            'feature_importance': fi_dict,
            'top_5_features': list(shap_importance.nlargest(5).index)
        }

    return results


def run_cross_validation_experiments():
    'Run cross validation experiments.'
    print("\n" + "=" * 70)
    print("Step 2: 10-Fold Cross-Validation")
    print("=" * 70)


    data = prepare_dataset(scale=True)
    X = pd.concat([data['X_train'], data['X_test']], axis=0).reset_index(drop=True)
    y = pd.concat([data['y_train'], data['y_test']], axis=0).reset_index(drop=True)

    print(f"\n[Data] X: {X.shape}, y: {y.shape}")

    validator = StatisticalValidator(n_folds=N_FOLDS, random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    for model_name in ['random_forest', 'xgboost']:
        print(f"\n[{model_name.upper()} - {N_FOLDS}-fold CV]")

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = create_model(model_name)
            model = train_model(model, X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            fold_metrics.append(metrics)

            if (fold_idx + 1) % 5 == 0:
                print(f"  Fold {fold_idx+1}/{N_FOLDS} complete")


        metrics_df = pd.DataFrame(fold_metrics)
        stats = {}

        for col in metrics_df.columns:
            values = metrics_df[col].values
            cv_result = validator.compute_confidence_intervals(values, confidence=CONFIDENCE_LEVEL)
            stats[col] = {
                'mean': float(cv_result.mean),
                'std': float(cv_result.std),
                'ci_lower': float(cv_result.ci_lower),
                'ci_upper': float(cv_result.ci_upper),
                'fold_values': values.tolist()
            }

        results[model_name] = stats
        print(f"  Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        print(f"  95% CI: [{stats['accuracy']['ci_lower']:.4f}, {stats['accuracy']['ci_upper']:.4f}]")

    # Paired t-test: RF vs XGBoost
    rf_f1 = np.array(results['random_forest']['f1']['fold_values'])
    xgb_f1 = np.array(results['xgboost']['f1']['fold_values'])
    ttest_result = validator.paired_t_test(rf_f1, xgb_f1)

    results['statistical_test'] = {
        'comparison': 'rf_vs_xgboost',
        'metric': 'f1',
        't_statistic': float(ttest_result['t_statistic']),
        'p_value': float(ttest_result['p_value']),
        'effect_size_cohens_d': float(ttest_result['effect_size']),
        'significant': bool(ttest_result['significant'])
    }

    print(f"\n[Statistical Test: RF vs XGBoost]")
    print(f"  t-statistic: {ttest_result['t_statistic']:.4f}")
    print(f"  p-value: {ttest_result['p_value']:.4f}")
    print(f"  Cohen's d: {ttest_result['effect_size']:.4f}")
    print(f"  Significant: {ttest_result['significant']}")

    return results


def run_quality_degradation_experiments():
    'Run quality degradation experiments.'
    print("\n" + "=" * 70)
    print("Step 3: Quality Degradation Experiments")
    print("=" * 70)

    data = prepare_dataset(scale=False)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']


    num_cols = [c for c in NUMERICAL_COLS if c in X_test.columns]

    results = {'issue_types': QUALITY_ISSUE_TYPES, 'severity_levels': SEVERITY_LEVELS, 'experiments': []}

    for model_name in ['random_forest', 'xgboost']:
        print(f"\n[{model_name.upper()}]")


        model = create_model(model_name)
        model = train_model(model, X_train, y_train)


        fi_baseline = analyze_xai(model, X_train, X_test, method='shap', n_samples=100)

        for issue_type in QUALITY_ISSUE_TYPES:
            for severity in SEVERITY_LEVELS:

                if issue_type == 'missing':
                    X_degraded, affected = inject_missing_values(
                        X_test.copy(), num_cols, severity, return_affected=True
                    )
                    X_degraded = X_degraded.fillna(X_degraded.median())
                elif issue_type == 'outlier':
                    X_degraded, affected = inject_outliers(
                        X_test.copy(), num_cols, severity, return_affected=True
                    )
                elif issue_type == 'distribution_shift':
                    X_degraded, affected = inject_distribution_shift(
                        X_test.copy(), num_cols, severity,
                        shift_type='covariate_shift', return_affected=True
                    )


                fi_degraded = analyze_xai(model, X_train, X_degraded, method='shap', n_samples=100)


                div_metrics = calculate_fi_divergence_metrics(fi_baseline, fi_degraded)

                experiment = {
                    'model': model_name,
                    'issue_type': issue_type,
                    'severity': severity,
                    'affected_columns': affected,
                    'js_divergence': float(div_metrics['js_divergence']),
                    'spearman_correlation': float(div_metrics['spearman_correlation']),
                    'rank_change_ratio_top5': float(div_metrics['rank_change_ratio_top5']),
                    'rank_change_ratio_top10': float(div_metrics['rank_change_ratio_top10'])
                }
                results['experiments'].append(experiment)

                print(f"  {issue_type} {severity*100:.0f}%: JS={div_metrics['js_divergence']:.4f}, Spearman={div_metrics['spearman_correlation']:.4f}")

    return results


def run_deep_learning_experiments():
    'Run deep learning experiments.'
    print("\n" + "=" * 70)
    print("Step 4: Deep Learning Experiments (MLP, AttentionNet)")
    print("=" * 70)

    data = prepare_dataset(scale=True)
    X_train = data['X_train'].values
    y_train = data['y_train'].values
    X_test = data['X_test'].values
    y_test = data['y_test'].values
    feature_names = data['feature_names']

    print(f"\n[Data] X_train: {X_train.shape}, Device: {DEVICE}")

    results = {}

    # MLP
    print(f"\n[MLP]")
    mlp_wrapper = DLModelWrapper(
        model_class=TabularMLP,
        model_params={'hidden_dims': [128, 64, 32], 'dropout': 0.3},
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        learning_rate=DL_LEARNING_RATE,
        device=DEVICE,
        random_state=RANDOM_SEED,
        verbose=True
    )
    mlp_wrapper.fit(X_train, y_train)

    y_pred = mlp_wrapper.predict(X_test)
    y_proba = mlp_wrapper.predict_proba(X_test)[:, 1]

    mlp_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba))
    }

    print(f"  Accuracy: {mlp_metrics['accuracy']:.4f}")
    print(f"  F1: {mlp_metrics['f1']:.4f}")
    print(f"  AUC: {mlp_metrics['roc_auc']:.4f}")

    # Integrated Gradients
    print("  Running Integrated Gradients analysis...")
    dl_analyzer = DLXAIAnalyzer(mlp_wrapper.model, device=DEVICE)
    X_sample = mlp_wrapper.scaler.transform(X_test[:100])
    ig_importance = dl_analyzer.get_feature_importance(X_sample, method='integrated_gradients')

    results['MLP'] = {
        'metrics': mlp_metrics,
        'feature_importance': {feature_names[i]: float(ig_importance[i]) for i in range(len(feature_names))},
        'training_loss': mlp_wrapper.training_history['loss']
    }

    # AttentionNet
    print(f"\n[AttentionNet]")
    attn_wrapper = DLModelWrapper(
        model_class=TabularAttentionNet,
        model_params={'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128, 'num_layers': 2, 'dropout': 0.2},
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        learning_rate=DL_LEARNING_RATE * 0.5,  # Lower LR for attention
        device=DEVICE,
        random_state=RANDOM_SEED,
        verbose=True
    )
    attn_wrapper.fit(X_train, y_train)

    y_pred = attn_wrapper.predict(X_test)
    y_proba = attn_wrapper.predict_proba(X_test)[:, 1]

    attn_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba))
    }

    print(f"  Accuracy: {attn_metrics['accuracy']:.4f}")
    print(f"  F1: {attn_metrics['f1']:.4f}")
    print(f"  AUC: {attn_metrics['roc_auc']:.4f}")

    # Attention weights
    print("  Extracting attention weights...")
    attn_wrapper.model.eval()
    with torch.no_grad():
        X_sample_tensor = torch.FloatTensor(X_sample).to(DEVICE)
        _ = attn_wrapper.model(X_sample_tensor)
        attn_importance = attn_wrapper.model.get_attention_weights()

    results['AttentionNet'] = {
        'metrics': attn_metrics,
        'feature_importance': {feature_names[i]: float(attn_importance[i]) for i in range(len(feature_names))},
        'training_loss': attn_wrapper.training_history['loss']
    }

    return results


def run_traditional_baselines():
    'Run traditional baselines.'
    print("\n" + "=" * 70)
    print("Step 5: Traditional Baselines Comparison")
    print("=" * 70)

    data = prepare_dataset(scale=False)
    X_test = data['X_test']
    num_cols = [c for c in NUMERICAL_COLS if c in X_test.columns]


    X_degraded, affected_cols = inject_outliers(X_test.copy(), num_cols, 0.1, return_affected=True)

    print(f"\n[Outlier Injection: 10%, Affected: {affected_cols}]")

    detector = TraditionalQualityDetector()
    results = {}

    # Z-score
    print("\n[Z-score Detection]")
    zscore_result = detector.detect_outliers_zscore(X_degraded)
    results['zscore'] = {
        'detected_features': zscore_result['affected_features'],
        'n_detected': len(zscore_result['affected_features']),
        'total_outliers': int(zscore_result['total_outliers'])
    }
    print(f"  Detected: {zscore_result['affected_features']}")

    # IQR
    print("\n[IQR Detection]")
    iqr_result = detector.detect_outliers_iqr(X_degraded)
    results['iqr'] = {
        'detected_features': iqr_result['affected_features'],
        'n_detected': len(iqr_result['affected_features']),
        'total_outliers': int(iqr_result['total_outliers'])
    }
    print(f"  Detected: {iqr_result['affected_features']}")

    # KS Test
    print("\n[KS Test Detection]")
    ks_result = detector.detect_distribution_shift_ks(X_test, X_degraded)
    results['ks_test'] = {
        'detected_features': ks_result['affected_features'],
        'n_detected': ks_result['n_affected']
    }
    print(f"  Detected: {ks_result['affected_features']}")

    # PSI
    print("\n[PSI Detection]")
    psi_result = detector.detect_distribution_shift_psi(X_test, X_degraded)
    results['psi'] = {
        'detected_features': psi_result['affected_features'],
        'n_detected': psi_result['n_affected']
    }
    print(f"  Detected: {psi_result['affected_features']}")


    print("\n[XAI (SHAP) Detection]")
    model = create_model('random_forest')
    model = train_model(model, data['X_train'], data['y_train'])
    fi_baseline = analyze_xai(model, data['X_train'], X_test, method='shap', n_samples=100)
    fi_degraded = analyze_xai(model, data['X_train'], X_degraded, method='shap', n_samples=100)


    fi_change = abs(fi_baseline - fi_degraded) / (fi_baseline + 1e-10)
    top_changed = fi_change.nlargest(5).index.tolist()

    results['xai_shap'] = {
        'top_changed_features': top_changed,
        'fi_change': fi_change.to_dict()
    }
    print(f"  Top Changed: {top_changed}")


    true_affected = set(affected_cols)

    for method in ['zscore', 'iqr', 'ks_test', 'psi']:
        detected = set(results[method]['detected_features'])
        tp = len(true_affected & detected)
        precision = tp / len(detected) if len(detected) > 0 else 0
        recall = tp / len(true_affected) if len(true_affected) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[method]['precision'] = precision
        results[method]['recall'] = recall
        results[method]['f1'] = f1


    detected_xai = set(top_changed[:len(affected_cols)])
    tp_xai = len(true_affected & detected_xai)
    results['xai_shap']['precision'] = tp_xai / len(detected_xai) if len(detected_xai) > 0 else 0
    results['xai_shap']['recall'] = tp_xai / len(true_affected) if len(true_affected) > 0 else 0

    print("\n[Detection Accuracy Summary]")
    for method in ['zscore', 'iqr', 'ks_test', 'psi', 'xai_shap']:
        if 'precision' in results[method]:
            print(f"  {method}: Precision={results[method]['precision']:.2f}, Recall={results[method]['recall']:.2f}")

    results['true_affected'] = affected_cols

    return results


def run_all_experiments():
    'Run the full experiment suite.'
    print("=" * 70)
    print("Full Experiment Runner - Phase 5")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)

    all_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
            'n_folds': N_FOLDS,
            'confidence_level': CONFIDENCE_LEVEL
        }
    }

    # Step 1: Baseline
    baseline_results = run_baseline_experiments()
    all_results['baseline'] = baseline_results

    # Step 2: Cross-validation
    cv_results = run_cross_validation_experiments()
    all_results['cross_validation'] = cv_results

    # Step 3: Quality degradation
    degradation_results = run_quality_degradation_experiments()
    all_results['quality_degradation'] = degradation_results

    # Step 4: Deep Learning
    dl_results = run_deep_learning_experiments()
    all_results['deep_learning'] = dl_results

    # Step 5: Traditional baselines
    traditional_results = run_traditional_baselines()
    all_results['traditional_baselines'] = traditional_results


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f'full_experiment_results_{timestamp}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print('Full experiment run complete!')
    print(f"Saved results: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return all_results, output_file


if __name__ == '__main__':
    results, output_path = run_all_experiments()
    print(f"\nResult file: {output_path}")
