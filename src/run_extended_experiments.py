'Extended experiment runner for MAR and deep learning FI-divergence analyses.'

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import warnings
from typing import Dict, List, Tuple
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')


from config import RESULTS_DIR, RANDOM_SEED, N_FOLDS

# =============================================================================
# PART 1: MAR (Missing At Random) Simulation
# =============================================================================

def inject_missing_mcar(
    X: pd.DataFrame,
    target_features: List[str],
    severity: float,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, List[str]]:
    'Extended experiment runner for MAR and deep learning FI-divergence analyses.'
    np.random.seed(seed)
    X_corrupted = X.copy()

    for col in target_features:
        if col in X.columns:
            mask = np.random.random(len(X)) < severity
            X_corrupted.loc[mask, col] = np.nan

            median_val = X[col].median()
            X_corrupted[col] = X_corrupted[col].fillna(median_val)

    return X_corrupted, target_features


def inject_missing_mar(
    X: pd.DataFrame,
    y: pd.Series,
    target_features: List[str],
    severity: float,
    correlation_strength: float = 0.8,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, List[str]]:
    'Extended experiment runner for MAR and deep learning FI-divergence analyses.'
    np.random.seed(seed)
    X_corrupted = X.copy()
    y_values = y.values if hasattr(y, 'values') else y

    for col in target_features:
        if col in X.columns:
            # Label-dependent missing probability
            prob_missing = np.where(
                y_values == 1,
                severity * correlation_strength,
                severity * (1 - correlation_strength)
            )
            mask = np.random.random(len(X)) < prob_missing
            X_corrupted.loc[mask, col] = np.nan

            median_val = X[col].median()
            X_corrupted[col] = X_corrupted[col].fillna(median_val)

    return X_corrupted, target_features


def run_mar_comparison_experiments():
    'Run mar comparison experiments.'
    print("=" * 60)
    print("Experiment 1: MCAR vs MAR Comparison")
    print("=" * 60)

    from dataset_registry import create_default_registry
    from models import create_model, train_model, evaluate_model
    from xai_analyzer import analyze_xai


    print("\n[Loading UCI Adult dataset]")
    registry = create_default_registry()
    X, y = registry.get_dataset('uci_adult')
    config = registry.get_config('uci_adult')

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Numerical: {config.numerical_features[:3]}...")


    target_features = [f for f in config.numerical_features if f in X.columns][:3]
    print(f"  Target features for corruption: {target_features}")

    results = []
    severity_levels = [0.10, 0.20, 0.30]
    n_folds = 5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    total_experiments = n_folds * 2 * len(severity_levels) * 2  # 2 models * 2 missing types

    with tqdm(total=total_experiments, desc="MAR vs MCAR Experiments") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

            for model_type in ['random_forest', 'xgboost']:

                model_baseline = create_model(model_type)
                model_baseline = train_model(model_baseline, X_train, y_train)
                baseline_metrics = evaluate_model(model_baseline, X_test, y_test)

                # Baseline SHAP
                try:
                    fi_baseline = analyze_xai(
                        model_baseline, X_train, X_test,
                        method='shap', n_samples=min(100, len(X_test))
                    )
                except:
                    fi_baseline = pd.Series({col: 1.0/len(X_train.columns) for col in X_train.columns})

                for severity in severity_levels:
                    for missing_type in ['mcar', 'mar']:

                        if missing_type == 'mcar':
                            X_train_corrupted, _ = inject_missing_mcar(
                                X_train, target_features, severity
                            )
                        else:  # mar
                            X_train_corrupted, _ = inject_missing_mar(
                                X_train, y_train, target_features, severity
                            )


                        model_corrupted = create_model(model_type)
                        model_corrupted = train_model(model_corrupted, X_train_corrupted, y_train)
                        corrupted_metrics = evaluate_model(model_corrupted, X_test, y_test)

                        # Corrupted SHAP
                        try:
                            fi_corrupted = analyze_xai(
                                model_corrupted, X_train_corrupted, X_test,
                                method='shap', n_samples=min(100, len(X_test))
                            )
                        except:
                            fi_corrupted = pd.Series({col: 1.0/len(X_train.columns) for col in X_train.columns})


                        base_arr = np.array(list(fi_baseline.values))
                        corr_arr = np.array(list(fi_corrupted.values))
                        base_arr = base_arr / (base_arr.sum() + 1e-10)
                        corr_arr = corr_arr / (corr_arr.sum() + 1e-10)

                        js_div = jensenshannon(base_arr, corr_arr) ** 2
                        spearman_corr, _ = spearmanr(base_arr, corr_arr)

                        result = {
                            'fold': fold_idx,
                            'model': model_type,
                            'missing_type': missing_type,
                            'severity': severity,
                            'baseline_accuracy': baseline_metrics['accuracy'],
                            'corrupted_accuracy': corrupted_metrics['accuracy'],
                            'accuracy_drop': baseline_metrics['accuracy'] - corrupted_metrics['accuracy'],
                            'js_divergence': float(js_div),
                            'spearman_correlation': float(spearman_corr)
                        }
                        results.append(result)
                        pbar.update(1)


    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'mar_comparison_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")


    print("\n" + "=" * 60)
    print("MAR vs MCAR Summary")
    print("=" * 60)

    summary = results_df.groupby(['missing_type', 'severity']).agg({
        'js_divergence': ['mean', 'std'],
        'accuracy_drop': ['mean', 'std']
    }).round(4)
    print(summary)

    return results_df


# =============================================================================
# PART 2: DL Model FI Divergence Experiments
# =============================================================================

def run_dl_fi_divergence_experiments():
    'Run dl fi divergence experiments.'
    import torch
    from dl_models import TabularMLP, TabularAttentionNet, DLModelWrapper
    from dl_xai import DLXAIAnalyzer

    print("\n" + "=" * 60)
    print("Experiment 2: DL Model FI Divergence")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from dataset_registry import create_default_registry


    print("\n[Loading UCI Adult dataset]")
    registry = create_default_registry()
    X, y = registry.get_dataset('uci_adult')
    config = registry.get_config('uci_adult')


    max_samples = 10000
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=RANDOM_SEED
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    print(f"  Samples: {len(X):,}")

    target_features = [f for f in config.numerical_features if f in X.columns][:3]
    feature_names = list(X.columns)

    results = []
    severity_levels = [0.10, 0.30]
    quality_issues = ['missing', 'outlier', 'distribution_shift']
    n_folds = 5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)


    dl_models = {
        'mlp': {
            'class': TabularMLP,
            'params': {'hidden_dims': [128, 64, 32]},
            'epochs': 30,
            'lr': 0.001,
            'xai_method': 'integrated_gradients'
        },
        'attention': {
            'class': TabularAttentionNet,
            'params': {'embed_dim': 64, 'num_heads': 4, 'num_layers': 2},
            'epochs': 40,
            'lr': 0.0005,
            'xai_method': 'attention'
        }
    }

    total_experiments = n_folds * len(dl_models) * len(quality_issues) * len(severity_levels)

    with tqdm(total=total_experiments, desc="DL FI Divergence Experiments") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train = X.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            for model_name, model_config in dl_models.items():

                print(f"\n  Fold {fold_idx+1}, {model_name}: Training baseline...")

                wrapper_baseline = DLModelWrapper(
                    model_class=model_config['class'],
                    model_params=model_config['params'],
                    epochs=model_config['epochs'],
                    learning_rate=model_config['lr'],
                    device=device,
                    verbose=False
                )
                wrapper_baseline.fit(X_train, y_train)


                y_pred = wrapper_baseline.predict(X_test)
                baseline_accuracy = (y_pred == y_test).mean()

                # Baseline FI (Integrated Gradients or Attention)
                analyzer_baseline = DLXAIAnalyzer(wrapper_baseline.model, device)
                X_test_scaled = wrapper_baseline.scaler.transform(X_test[:100])

                try:
                    if model_config['xai_method'] == 'integrated_gradients':
                        fi_baseline = analyzer_baseline.get_feature_importance(
                            X_test_scaled, method='integrated_gradients', n_steps=30
                        )
                    else:  # attention
                        fi_baseline = analyzer_baseline.get_feature_importance(
                            X_test_scaled, method='attention'
                        )
                except Exception as e:
                    print(f"    Baseline XAI failed: {e}")
                    fi_baseline = np.ones(X_train.shape[1]) / X_train.shape[1]

                for issue_type in quality_issues:
                    for severity in severity_levels:

                        X_train_df = pd.DataFrame(X_train, columns=feature_names)

                        if issue_type == 'missing':
                            X_train_corrupted, _ = inject_missing_mcar(
                                X_train_df, target_features, severity
                            )
                        elif issue_type == 'outlier':
                            X_train_corrupted = inject_outlier(
                                X_train_df, target_features, severity
                            )
                        else:  # distribution_shift
                            X_train_corrupted = inject_distribution_shift(
                                X_train_df, target_features, severity
                            )

                        X_train_corrupted = X_train_corrupted.values


                        wrapper_corrupted = DLModelWrapper(
                            model_class=model_config['class'],
                            model_params=model_config['params'],
                            epochs=model_config['epochs'],
                            learning_rate=model_config['lr'],
                            device=device,
                            verbose=False
                        )
                        wrapper_corrupted.fit(X_train_corrupted, y_train)


                        y_pred_corr = wrapper_corrupted.predict(X_test)
                        corrupted_accuracy = (y_pred_corr == y_test).mean()

                        # Corrupted FI
                        analyzer_corrupted = DLXAIAnalyzer(wrapper_corrupted.model, device)
                        X_test_scaled_corr = wrapper_corrupted.scaler.transform(X_test[:100])

                        try:
                            if model_config['xai_method'] == 'integrated_gradients':
                                fi_corrupted = analyzer_corrupted.get_feature_importance(
                                    X_test_scaled_corr, method='integrated_gradients', n_steps=30
                                )
                            else:
                                fi_corrupted = analyzer_corrupted.get_feature_importance(
                                    X_test_scaled_corr, method='attention'
                                )
                        except Exception as e:
                            print(f"    Corrupted XAI failed: {e}")
                            fi_corrupted = np.ones(X_train.shape[1]) / X_train.shape[1]


                        base_arr = np.abs(fi_baseline) / (np.abs(fi_baseline).sum() + 1e-10)
                        corr_arr = np.abs(fi_corrupted) / (np.abs(fi_corrupted).sum() + 1e-10)

                        js_div = jensenshannon(base_arr, corr_arr) ** 2
                        spearman_corr, _ = spearmanr(base_arr, corr_arr)

                        result = {
                            'fold': fold_idx,
                            'model': model_name,
                            'xai_method': model_config['xai_method'],
                            'issue_type': issue_type,
                            'severity': severity,
                            'baseline_accuracy': float(baseline_accuracy),
                            'corrupted_accuracy': float(corrupted_accuracy),
                            'accuracy_drop': float(baseline_accuracy - corrupted_accuracy),
                            'js_divergence': float(js_div),
                            'spearman_correlation': float(spearman_corr)
                        }
                        results.append(result)
                        pbar.update(1)


    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'dl_fi_divergence_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")


    print("\n" + "=" * 60)
    print("DL FI Divergence Summary")
    print("=" * 60)

    summary = results_df.groupby(['model', 'issue_type']).agg({
        'js_divergence': ['mean', 'std'],
        'accuracy_drop': ['mean', 'std']
    }).round(4)
    print(summary)

    return results_df


def inject_outlier(
    X: pd.DataFrame,
    target_features: List[str],
    severity: float,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    'Extended experiment runner for MAR and deep learning FI-divergence analyses.'
    np.random.seed(seed)
    X_corrupted = X.copy()

    for col in target_features:
        if col in X.columns:
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

    return X_corrupted


def inject_distribution_shift(
    X: pd.DataFrame,
    target_features: List[str],
    severity: float,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    'Extended experiment runner for MAR and deep learning FI-divergence analyses.'
    np.random.seed(seed)
    X_corrupted = X.copy()

    shift_amount = severity * 3
    for col in target_features:
        if col in X.columns:
            std = X[col].std()
            X_corrupted[col] = X_corrupted[col] + shift_amount * std

    return X_corrupted


def main():
    'Run the script entry point.'
    print("=" * 60)
    print("Extended Experiments: MAR & DL FI Divergence")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # Experiment 1: MAR vs MCAR
    mar_results = run_mar_comparison_experiments()

    # Experiment 2: DL FI Divergence
    dl_results = run_dl_fi_divergence_experiments()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n[MAR vs MCAR Comparison]")
    mar_summary = mar_results.groupby('missing_type')['js_divergence'].mean()
    print(f"  MCAR Mean JS Divergence: {mar_summary.get('mcar', 0):.4f}")
    print(f"  MAR Mean JS Divergence: {mar_summary.get('mar', 0):.4f}")
    if 'mar' in mar_summary.index and 'mcar' in mar_summary.index:
        diff_pct = (mar_summary['mar'] - mar_summary['mcar']) / mar_summary['mcar'] * 100
        print(f"  MAR shows {diff_pct:.1f}% higher FI Divergence than MCAR")

    print("\n[DL Model FI Divergence]")
    dl_summary = dl_results.groupby('model')['js_divergence'].mean()
    for model, js in dl_summary.items():
        print(f"  {model.upper()} Mean JS Divergence: {js:.4f}")

    return mar_results, dl_results


if __name__ == '__main__':
    mar_results, dl_results = main()
