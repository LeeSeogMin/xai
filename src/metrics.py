'Metrics for feature-importance divergence and data quality diagnosis.'

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, kendalltau
from typing import Optional


def normalize_distribution(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    'Normalize distribution.'
    x = np.array(x, dtype=float)
    x = np.abs(x) + eps
    return x / x.sum()


def calculate_js_divergence(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series
) -> float:
    'Metrics for feature-importance divergence and data quality diagnosis.'

    common_features = fi_baseline.index.intersection(fi_corrupted.index)
    fi_base = fi_baseline[common_features].values
    fi_corr = fi_corrupted[common_features].values


    p = normalize_distribution(fi_base)
    q = normalize_distribution(fi_corr)

    return jensenshannon(p, q)


def calculate_rank_correlation(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    method: str = 'spearman'
) -> tuple[float, float]:
    'Metrics for feature-importance divergence and data quality diagnosis.'
    common_features = fi_baseline.index.intersection(fi_corrupted.index)
    fi_base = fi_baseline[common_features].values
    fi_corr = fi_corrupted[common_features].values

    if method == 'spearman':
        corr, p_value = spearmanr(fi_base, fi_corr)
    elif method == 'kendall':
        corr, p_value = kendalltau(fi_base, fi_corr)
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr, p_value


def calculate_rank_change_ratio(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    top_n: int = 10
) -> float:
    'Metrics for feature-importance divergence and data quality diagnosis.'

    top_baseline = set(fi_baseline.nlargest(top_n).index)
    top_corrupted = set(fi_corrupted.nlargest(top_n).index)


    common = len(top_baseline.intersection(top_corrupted))


    return 1 - (common / top_n)


def calculate_fi_divergence_metrics(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series
) -> dict:
    'Metrics for feature-importance divergence and data quality diagnosis.'
    js_div = calculate_js_divergence(fi_baseline, fi_corrupted)
    spearman_corr, spearman_p = calculate_rank_correlation(
        fi_baseline, fi_corrupted, 'spearman'
    )
    rank_change_5 = calculate_rank_change_ratio(fi_baseline, fi_corrupted, 5)
    rank_change_10 = calculate_rank_change_ratio(fi_baseline, fi_corrupted, 10)

    return {
        'js_divergence': js_div,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'rank_change_ratio_top5': rank_change_5,
        'rank_change_ratio_top10': rank_change_10
    }



def compute_fi_divergence(fi_baseline: pd.Series, fi_corrupted: pd.Series) -> dict:
    """Compatibility wrapper for README examples."""
    metrics = calculate_fi_divergence_metrics(fi_baseline, fi_corrupted)
    metrics['js'] = metrics['js_divergence']
    return metrics

def calculate_detection_accuracy(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    corrupted_columns: list[str],
    threshold_percentile: int = 90
) -> dict:
    'Metrics for feature-importance divergence and data quality diagnosis.'
    common_features = fi_baseline.index.intersection(fi_corrupted.index)


    fi_change = abs(fi_baseline[common_features] - fi_corrupted[common_features])
    fi_change_pct = fi_change / (fi_baseline[common_features] + 1e-10) * 100


    threshold = np.percentile(fi_change_pct, threshold_percentile)
    detected_features = set(fi_change_pct[fi_change_pct >= threshold].index)


    corrupted_set = set(corrupted_columns)
    true_positives = len(detected_features.intersection(corrupted_set))
    false_positives = len(detected_features - corrupted_set)
    false_negatives = len(corrupted_set - detected_features)

    precision = true_positives / max(len(detected_features), 1)
    recall = true_positives / max(len(corrupted_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1,
        'detected_features': list(detected_features),
        'corrupted_features': corrupted_columns
    }


def compare_xai_methods(
    fi_dict: dict[str, pd.Series],
    baseline_method: str = None
) -> pd.DataFrame:
    'Metrics for feature-importance divergence and data quality diagnosis.'
    methods = list(fi_dict.keys())
    n_methods = len(methods)

    correlation_matrix = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr, _ = calculate_rank_correlation(
                    fi_dict[method1], fi_dict[method2]
                )
                correlation_matrix[i, j] = corr

    return pd.DataFrame(
        correlation_matrix,
        index=methods,
        columns=methods
    )


# ====================================================================
# Improvement 2: Statistical Validation - Variance-Aware Metrics
# ====================================================================

def compute_fi_divergence_with_variance(
    baseline_fi: np.ndarray,
    degraded_fi: np.ndarray
) -> dict[str, float]:
    'Metrics for feature-importance divergence and data quality diagnosis.'
    assert baseline_fi.shape == degraded_fi.shape, \
        "Baseline and degraded FI must have same shape"

    n_folds, n_features = baseline_fi.shape

    js_divergences = []
    spearman_corrs = []
    rank_changes = []

    for fold_idx in range(n_folds):
        # Extract FI for this fold
        fi_base = pd.Series(baseline_fi[fold_idx, :], index=range(n_features))
        fi_deg = pd.Series(degraded_fi[fold_idx, :], index=range(n_features))

        # Compute metrics
        js_div = calculate_js_divergence(fi_base, fi_deg)
        spearman_corr, _ = calculate_rank_correlation(fi_base, fi_deg)
        rank_change = calculate_rank_change_ratio(fi_base, fi_deg, top_n=min(10, n_features))

        js_divergences.append(js_div)
        spearman_corrs.append(spearman_corr)
        rank_changes.append(rank_change)

    # Aggregate statistics
    return {
        'js_divergence_mean': np.mean(js_divergences),
        'js_divergence_std': np.std(js_divergences, ddof=1),
        'js_divergence_min': np.min(js_divergences),
        'js_divergence_max': np.max(js_divergences),
        'spearman_mean': np.mean(spearman_corrs),
        'spearman_std': np.std(spearman_corrs, ddof=1),
        'spearman_min': np.min(spearman_corrs),
        'spearman_max': np.max(spearman_corrs),
        'rank_change_mean': np.mean(rank_changes),
        'rank_change_std': np.std(rank_changes, ddof=1),
        'n_folds': n_folds
    }


if __name__ == '__main__':
    print("=" * 60)
    print('FI Divergence metrics test')
    print("=" * 60)


    np.random.seed(42)
    features = [f'feature_{i}' for i in range(10)]


    fi_baseline = pd.Series(
        np.random.rand(10) + 0.1,
        index=features
    )


    fi_corrupted = fi_baseline.copy()
    fi_corrupted['feature_0'] *= 2.0
    fi_corrupted['feature_1'] *= 0.5
    fi_corrupted['feature_2'] += 0.1

    print('Baseline feature importance')
    print(fi_baseline.round(3))

    print('Corrupted feature importance')
    print(fi_corrupted.round(3))


    print("\n[FI Divergence Metrics]")
    metrics = calculate_fi_divergence_metrics(fi_baseline, fi_corrupted)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


    print("\n[Detection Accuracy]")
    detection = calculate_detection_accuracy(
        fi_baseline, fi_corrupted,
        ['feature_0', 'feature_1']
    )
    for key, value in detection.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print('Metrics test complete!')
