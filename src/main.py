'Main experiment runner for the XAI-based data quality diagnosis workflow.'

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    prepare_dataset, NUMERICAL_COLS, CATEGORICAL_COLS,
    encode_features, scale_features
)
from quality_simulator import (
    inject_missing_values, inject_outliers,
    inject_distribution_shift, SEVERITY_LEVELS
)
from config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
    QUALITY_ISSUE_TYPES, MODEL_CONFIGS
)
from models import create_model, train_model, evaluate_model, get_feature_importance
from xai_analyzer import analyze_xai
from metrics import calculate_fi_divergence_metrics

warnings.filterwarnings('ignore')


EXPERIMENT_CONFIG = {
    'models': ['random_forest', 'xgboost'],
    'xai_methods': ['shap', 'lime'],
    'quality_issues': ['missing', 'outlier', 'distribution_shift'],
    'severity_levels': SEVERITY_LEVELS,
    'n_xai_samples': 100,
    'random_seed': RANDOM_SEED
}


def run_baseline_experiment(data: dict) -> dict:
    'Run baseline machine-learning and SHAP experiments.'
    print("\n" + "=" * 60)
    print('Baseline experiments')
    print("=" * 60)

    baseline_results = {}

    for model_type in EXPERIMENT_CONFIG['models']:
        print(f"\n[{model_type.upper()}]")


        model = create_model(model_type)
        model = train_model(
            model, data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )


        metrics = evaluate_model(model, data['X_test'], data['y_test'])
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")


        model_fi = get_feature_importance(model, data['feature_names'])


        xai_results = {}
        for xai_method in EXPERIMENT_CONFIG['xai_methods']:
            print(f"  XAI ({xai_method})...", end=' ')
            fi = analyze_xai(
                model, data['X_train'], data['X_test'],
                method=xai_method,
                n_samples=EXPERIMENT_CONFIG['n_xai_samples']
            )
            xai_results[xai_method] = fi
            print("Done")

        baseline_results[model_type] = {
            'model': model,
            'metrics': metrics,
            'model_fi': model_fi,
            'xai_fi': xai_results
        }

    return baseline_results


def run_quality_experiment(
    data: dict,
    baseline_results: dict,
    quality_issue: str,
    severity: float
) -> dict:
    'Main experiment runner for the XAI-based data quality diagnosis workflow.'
    results = {}


    train_raw = data['train_raw'].copy()
    encoders = data['encoders']
    scaler = data['scaler']


    if quality_issue == 'missing':

        corrupted = inject_missing_values(
            train_raw, NUMERICAL_COLS[:3], severity, RANDOM_SEED
        )

        for col in NUMERICAL_COLS[:3]:
            median_val = train_raw[col].median()
            corrupted[col] = corrupted[col].fillna(median_val)
        target_cols = NUMERICAL_COLS[:3]

    elif quality_issue == 'outlier':
        corrupted = inject_outliers(
            train_raw, NUMERICAL_COLS[:3], severity,
            method='iqr', random_seed=RANDOM_SEED
        )
        target_cols = NUMERICAL_COLS[:3]

    elif quality_issue == 'distribution_shift':
        corrupted = inject_distribution_shift(
            train_raw, NUMERICAL_COLS[:3], severity,
            shift_type='covariate_shift', random_seed=RANDOM_SEED
        )
        target_cols = NUMERICAL_COLS[:3]

    else:
        raise ValueError(f"Unknown quality issue: {quality_issue}")


    corrupted_encoded, _ = encode_features(corrupted, encoders=encoders, fit=False)
    corrupted_scaled, _ = scale_features(corrupted_encoded, scaler=scaler, fit=False)

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X_corrupted = corrupted_scaled[feature_cols]
    y_corrupted = corrupted_scaled['income']


    for model_type in EXPERIMENT_CONFIG['models']:
        baseline = baseline_results[model_type]


        model = create_model(model_type)
        model = train_model(model, X_corrupted, y_corrupted)


        metrics = evaluate_model(model, data['X_test'], data['y_test'])


        perf_degradation = {
            metric: baseline['metrics'][metric] - metrics[metric]
            for metric in metrics
        }


        model_fi = get_feature_importance(model, data['feature_names'])


        xai_results = {}
        divergence_results = {}

        for xai_method in EXPERIMENT_CONFIG['xai_methods']:
            fi = analyze_xai(
                model, X_corrupted, data['X_test'],
                method=xai_method,
                n_samples=EXPERIMENT_CONFIG['n_xai_samples']
            )
            xai_results[xai_method] = fi


            baseline_fi = baseline['xai_fi'][xai_method]
            divergence = calculate_fi_divergence_metrics(baseline_fi, fi)
            divergence_results[xai_method] = divergence

        results[model_type] = {
            'metrics': metrics,
            'perf_degradation': perf_degradation,
            'model_fi': model_fi,
            'xai_fi': xai_results,
            'divergence': divergence_results,
            'target_columns': target_cols
        }

    return results


def run_full_experiment() -> dict:
    'Run full experiment.'
    print("\n" + "=" * 60)
    print('Full XAI data quality experiment')
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


    print("\n[Data loading]")
    data = prepare_dataset()
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_val: {data['X_val'].shape}")
    print(f"  X_test: {data['X_test'].shape}")


    all_results = {
        'config': EXPERIMENT_CONFIG,
        'baseline': None,
        'experiments': {}
    }


    baseline_results = run_baseline_experiment(data)
    all_results['baseline'] = {
        model_type: {
            'metrics': result['metrics'],
            'model_fi': result['model_fi'].to_dict(),
            'xai_fi': {
                method: fi.to_dict()
                for method, fi in result['xai_fi'].items()
            }
        }
        for model_type, result in baseline_results.items()
    }


    print("\n" + "=" * 60)
    print('Quality degradation experiments')
    print("=" * 60)

    total_experiments = (
        len(EXPERIMENT_CONFIG['quality_issues']) *
        len(EXPERIMENT_CONFIG['severity_levels'])
    )

    with tqdm(total=total_experiments, desc='Experiment progress') as pbar:
        for quality_issue in EXPERIMENT_CONFIG['quality_issues']:
            all_results['experiments'][quality_issue] = {}

            for severity in EXPERIMENT_CONFIG['severity_levels']:
                pbar.set_description(f"{quality_issue} ({severity*100:.0f}%)")

                exp_results = run_quality_experiment(
                    data, baseline_results, quality_issue, severity
                )


                all_results['experiments'][quality_issue][severity] = {
                    model_type: {
                        'metrics': result['metrics'],
                        'perf_degradation': result['perf_degradation'],
                        'divergence': result['divergence'],
                        'target_columns': result['target_columns']
                    }
                    for model_type, result in exp_results.items()
                }

                pbar.update(1)

    print("\n" + "=" * 60)
    print('Experiment progress')
    print("=" * 60)

    return all_results


def save_results(results: dict, filename: str = None):
    'Save results.'
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"experiment_results_{timestamp}.json"

    filepath = RESULTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results: {filepath}")
    return filepath


def print_summary(results: dict):
    'Print summary.'
    print("\n" + "=" * 60)
    print('Print summary.')
    print("=" * 60)


    print('Results saved')
    for model_type, result in results['baseline'].items():
        metrics = result['metrics']
        print(f"  {model_type}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")


    print('Summary')
    for quality_issue in results['experiments']:
        print(f"\n  {quality_issue.upper()}:")
        for model_type in ['random_forest', 'xgboost']:
            max_degradation = 0
            max_severity = 0
            for severity, exp in results['experiments'][quality_issue].items():
                degradation = exp[model_type]['perf_degradation']['accuracy']
                if degradation > max_degradation:
                    max_degradation = degradation
                    max_severity = severity
            print(f"    {model_type}: -{max_degradation:.4f} (at {float(max_severity)*100:.0f}%)")


    print('Quality issue summary')
    for quality_issue in results['experiments']:
        print(f"\n  {quality_issue.upper()}:")
        exp = results['experiments'][quality_issue][0.30]
        for model_type in ['random_forest', 'xgboost']:
            print(f"    {model_type}:")
            for xai_method in ['shap', 'lime']:
                div = exp[model_type]['divergence'][xai_method]
                print(f"      {xai_method}: JS={div['js_divergence']:.4f}, "
                      f"Spearman={div['spearman_correlation']:.4f}")


if __name__ == '__main__':

    results = run_full_experiment()


    save_results(results)


    print_summary(results)
