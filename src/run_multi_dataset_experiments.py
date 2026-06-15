'Multi-dataset experiment runner for tree-based FI-divergence analysis.'

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import warnings
from typing import Dict, List, Tuple, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


from config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
    QUALITY_ISSUE_TYPES, SEVERITY_LEVELS, MODEL_CONFIGS,
    DATASETS, DATASET_SAMPLING, N_FOLDS, CONFIDENCE_LEVEL
)


def load_dataset_with_sampling(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    'Load dataset with sampling.'
    from dataset_registry import create_default_registry

    print(f"\n[Loading {dataset_name}...]")
    registry = create_default_registry()

    X, y = registry.get_dataset(dataset_name)
    config = registry.get_config(dataset_name)


    sampling_config = DATASET_SAMPLING.get(dataset_name, {})
    max_samples = sampling_config.get('max_samples')

    if max_samples and len(X) > max_samples:
        print(f"  Sampling {max_samples:,} from {len(X):,} samples (stratified)")
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y if sampling_config.get('stratify', True) else None,
            random_state=RANDOM_SEED
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)


    if sampling_config.get('apply_smote', False):
        print("  Applying SMOTE for class imbalance...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=RANDOM_SEED)
            X_arr, y_arr = smote.fit_resample(X.values, y.values)
            X = pd.DataFrame(X_arr, columns=X.columns)
            y = pd.Series(y_arr)
            print(f"  After SMOTE: {len(X):,} samples")
        except ImportError:
            print("  Warning: imbalanced-learn not installed. Skipping SMOTE.")

    print(f"  Final: {len(X):,} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(y.value_counts())}")

    return X, y, config.numerical_features, config.categorical_features


def inject_quality_issue(
    X: pd.DataFrame,
    issue_type: str,
    severity: float,
    numerical_features: List[str],
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    'Multi-dataset experiment runner for tree-based FI-divergence analysis.'
    X_corrupted = X.copy()
    np.random.seed(seed)


    target_features = [f for f in numerical_features if f in X.columns][:3]

    if not target_features:
        print(f"  Warning: No numerical features found for corruption")
        return X_corrupted

    if issue_type == 'missing':

        for col in target_features:
            mask = np.random.random(len(X)) < severity
            X_corrupted.loc[mask, col] = np.nan

        for col in target_features:
            median_val = X[col].median()
            X_corrupted[col] = X_corrupted[col].fillna(median_val)

    elif issue_type == 'outlier':

        for col in target_features:
            n_outliers = int(len(X) * severity)
            outlier_idx = np.random.choice(X_corrupted.index, n_outliers, replace=False)
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1

            outlier_values = np.where(
                np.random.random(n_outliers) > 0.5,
                q3 + 3 * iqr * (1 + np.random.random(n_outliers)),
                q1 - 3 * iqr * (1 + np.random.random(n_outliers))
            )
            X_corrupted.loc[outlier_idx, col] = outlier_values

    elif issue_type == 'distribution_shift':

        shift_amount = severity * 3
        for col in target_features:
            std = X[col].std()
            X_corrupted[col] = X_corrupted[col] + shift_amount * std

    return X_corrupted


def calculate_fi_divergence(fi_baseline: Dict, fi_corrupted: Dict) -> Dict:
    'Calculate feature-importance divergence.'
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr

    results = {}

    for method in fi_baseline.keys():
        if method not in fi_corrupted:
            continue


        base_val = fi_baseline[method]
        corr_val = fi_corrupted[method]

        if hasattr(base_val, 'values'):
            base = np.array(base_val.values)
        elif isinstance(base_val, dict):
            base = np.array(list(base_val.values()))
        else:
            base = np.array(base_val)

        if hasattr(corr_val, 'values'):
            corr = np.array(corr_val.values)
        elif isinstance(corr_val, dict):
            corr = np.array(list(corr_val.values()))
        else:
            corr = np.array(corr_val)


        base = base / (base.sum() + 1e-10)
        corr = corr / (corr.sum() + 1e-10)

        # Jensen-Shannon Divergence
        js_div = jensenshannon(base, corr) ** 2  # squared for proper metric


        spearman_corr, _ = spearmanr(base, corr)


        rank_base = np.argsort(np.argsort(-base))
        rank_corr = np.argsort(np.argsort(-corr))
        rank_change = np.mean(np.abs(rank_base - rank_corr))

        results[method] = {
            'js_divergence': float(js_div),
            'spearman_correlation': float(spearman_corr),
            'rank_change': float(rank_change)
        }

    return results


def run_single_fold_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: List[str],
    model_type: str,
    issue_type: str,
    severity: float,
    fold_idx: int
) -> Dict:
    'Multi-dataset experiment runner for tree-based FI-divergence analysis.'
    from models import create_model, train_model, evaluate_model
    from xai_analyzer import analyze_xai

    results = {
        'fold': fold_idx,
        'model': model_type,
        'issue_type': issue_type,
        'severity': severity
    }


    model_baseline = create_model(model_type)
    model_baseline = train_model(model_baseline, X_train, y_train)


    baseline_metrics = evaluate_model(model_baseline, X_test, y_test)
    results['baseline_accuracy'] = baseline_metrics['accuracy']
    results['baseline_f1'] = baseline_metrics['f1']
    results['baseline_auc'] = baseline_metrics['roc_auc']


    try:
        fi_baseline = analyze_xai(
            model_baseline, X_train, X_test,
            method='shap', n_samples=min(100, len(X_test))
        )
        fi_baseline = {'shap': fi_baseline}
    except Exception as e:
        print(f"    SHAP baseline failed: {e}")
        fi_baseline = {'shap': {col: 1.0/len(X_train.columns) for col in X_train.columns}}


    X_train_corrupted = inject_quality_issue(
        X_train, issue_type, severity, numerical_features
    )


    model_corrupted = create_model(model_type)
    model_corrupted = train_model(model_corrupted, X_train_corrupted, y_train)


    corrupted_metrics = evaluate_model(model_corrupted, X_test, y_test)
    results['corrupted_accuracy'] = corrupted_metrics['accuracy']
    results['corrupted_f1'] = corrupted_metrics['f1']
    results['corrupted_auc'] = corrupted_metrics['roc_auc']


    results['accuracy_drop'] = baseline_metrics['accuracy'] - corrupted_metrics['accuracy']
    results['f1_drop'] = baseline_metrics['f1'] - corrupted_metrics['f1']
    results['auc_drop'] = baseline_metrics['roc_auc'] - corrupted_metrics['roc_auc']


    try:
        fi_corrupted = analyze_xai(
            model_corrupted, X_train_corrupted, X_test,
            method='shap', n_samples=min(100, len(X_test))
        )
        fi_corrupted = {'shap': fi_corrupted}
    except Exception as e:
        print(f"    SHAP corrupted failed: {e}")
        fi_corrupted = {'shap': {col: 1.0/len(X_train.columns) for col in X_train.columns}}


    fi_divergence = calculate_fi_divergence(fi_baseline, fi_corrupted)
    if 'shap' in fi_divergence:
        results['js_divergence'] = fi_divergence['shap']['js_divergence']
        results['spearman_correlation'] = fi_divergence['shap']['spearman_correlation']
        results['rank_change'] = fi_divergence['shap']['rank_change']

    return results


def run_dataset_experiments(dataset_name: str) -> List[Dict]:
    'Run all experiments for one dataset.'
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")


    X, y, numerical_features, categorical_features = load_dataset_with_sampling(dataset_name)

    all_results = []


    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    total_experiments = N_FOLDS * len(['random_forest', 'xgboost']) * len(QUALITY_ISSUE_TYPES) * len(SEVERITY_LEVELS)

    with tqdm(total=total_experiments, desc=f"Experiments ({dataset_name})") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_type in ['random_forest', 'xgboost']:
                for issue_type in QUALITY_ISSUE_TYPES:
                    for severity in SEVERITY_LEVELS:
                        try:
                            result = run_single_fold_experiment(
                                X_train, X_test, y_train, y_test,
                                numerical_features, model_type,
                                issue_type, severity, fold_idx
                            )
                            result['dataset'] = dataset_name
                            all_results.append(result)
                        except Exception as e:
                            print(f"\n  Error in {dataset_name}/{model_type}/{issue_type}/{severity}: {e}")

                        pbar.update(1)

    return all_results


def compute_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    'Compute statistics.'
    from scipy import stats

    grouped = results_df.groupby(['dataset', 'model', 'issue_type', 'severity'])

    stats_list = []
    for name, group in grouped:
        dataset, model, issue_type, severity = name

        n = len(group)

        stat = {
            'dataset': dataset,
            'model': model,
            'issue_type': issue_type,
            'severity': severity,
            'n_folds': n,


            'accuracy_drop_mean': group['accuracy_drop'].mean(),
            'accuracy_drop_std': group['accuracy_drop'].std(),
            'f1_drop_mean': group['f1_drop'].mean(),
            'f1_drop_std': group['f1_drop'].std(),

            # FI Divergence
            'js_divergence_mean': group['js_divergence'].mean(),
            'js_divergence_std': group['js_divergence'].std(),
            'spearman_mean': group['spearman_correlation'].mean(),
            'spearman_std': group['spearman_correlation'].std(),
        }


        ci_mult = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1) if n > 1 else 0
        stat['js_divergence_ci'] = ci_mult * stat['js_divergence_std'] / np.sqrt(n) if n > 1 else 0

        stats_list.append(stat)

    return pd.DataFrame(stats_list)


def main():
    'Run the script entry point.'
    print("=" * 60)
    print("Multi-Dataset XAI Experiments")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print(f"\nDatasets: {DATASETS}")
    print(f"Quality Issues: {QUALITY_ISSUE_TYPES}")
    print(f"Severity Levels: {SEVERITY_LEVELS}")
    print(f"Cross-Validation Folds: {N_FOLDS}")

    all_results = []

    for dataset_name in DATASETS:
        try:
            results = run_dataset_experiments(dataset_name)
            all_results.extend(results)
            print(f"\n✓ {dataset_name}: {len(results)} experiments completed")
        except Exception as e:
            print(f"\n✗ {dataset_name} failed: {e}")
            import traceback
            traceback.print_exc()


    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'multi_dataset_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Raw results saved: {results_file}")


    stats_df = compute_statistics(results_df)
    stats_file = RESULTS_DIR / f'multi_dataset_stats_{timestamp}.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved: {stats_file}")


    json_file = RESULTS_DIR / f'multi_dataset_results_{timestamp}.json'
    results_df.to_json(json_file, orient='records', indent=2)
    print(f"✓ JSON results saved: {json_file}")


    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for dataset in DATASETS:
        if dataset in results_df['dataset'].values:
            ds_results = results_df[results_df['dataset'] == dataset]
            print(f"\n[{dataset.upper()}]")
            print(f"  Total experiments: {len(ds_results)}")
            print(f"  Mean JS Divergence: {ds_results['js_divergence'].mean():.4f}")
            print(f"  Mean Accuracy Drop: {ds_results['accuracy_drop'].mean():.4f}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results_df, stats_df


if __name__ == '__main__':
    results_df, stats_df = main()
