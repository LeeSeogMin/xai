"""
평가 지표 계산 모듈
Phase 5: Feature Importance Divergence 및 품질 진단 지표

XAI 기반 데이터 품질 진단을 위한 핵심 지표:
1. FI Divergence Score (Jensen-Shannon Divergence)
2. Rank Correlation (Spearman)
3. Rank Change Ratio
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, kendalltau
from typing import Optional


def normalize_distribution(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Feature Importance를 확률 분포로 정규화

    Args:
        x: Feature Importance 배열
        eps: 0 방지를 위한 작은 값

    Returns:
        정규화된 확률 분포
    """
    x = np.array(x, dtype=float)
    x = np.abs(x) + eps  # 음수 방지 및 0 방지
    return x / x.sum()


def calculate_js_divergence(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series
) -> float:
    """
    Jensen-Shannon Divergence 계산

    Args:
        fi_baseline: 기준 Feature Importance
        fi_corrupted: 품질 문제 주입 후 Feature Importance

    Returns:
        JS Divergence (0~1, 클수록 분포 차이 큼)
    """
    # 같은 인덱스 순서로 정렬
    common_features = fi_baseline.index.intersection(fi_corrupted.index)
    fi_base = fi_baseline[common_features].values
    fi_corr = fi_corrupted[common_features].values

    # 확률 분포로 정규화
    p = normalize_distribution(fi_base)
    q = normalize_distribution(fi_corr)

    return jensenshannon(p, q)


def calculate_rank_correlation(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    method: str = 'spearman'
) -> tuple[float, float]:
    """
    순위 상관관계 계산

    Args:
        fi_baseline: 기준 Feature Importance
        fi_corrupted: 품질 문제 주입 후 Feature Importance
        method: 'spearman' 또는 'kendall'

    Returns:
        (상관계수, p-value)
    """
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
    """
    상위 N개 특성의 순위 변화 비율 계산

    Args:
        fi_baseline: 기준 Feature Importance
        fi_corrupted: 품질 문제 주입 후 Feature Importance
        top_n: 비교할 상위 특성 수

    Returns:
        순위 변화 비율 (0~1, 클수록 순위 변화 큼)
    """
    # 상위 N개 특성 추출
    top_baseline = set(fi_baseline.nlargest(top_n).index)
    top_corrupted = set(fi_corrupted.nlargest(top_n).index)

    # 공통 특성 수
    common = len(top_baseline.intersection(top_corrupted))

    # 변화 비율 (1 - 공통비율)
    return 1 - (common / top_n)


def calculate_fi_divergence_metrics(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series
) -> dict:
    """
    FI Divergence 관련 모든 지표 계산

    Args:
        fi_baseline: 기준 Feature Importance
        fi_corrupted: 품질 문제 주입 후 Feature Importance

    Returns:
        모든 지표를 포함한 딕셔너리
    """
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


def calculate_detection_accuracy(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    corrupted_columns: list[str],
    threshold_percentile: int = 90
) -> dict:
    """
    품질 문제 탐지 정확도 계산

    품질 문제가 주입된 컬럼의 Feature Importance 변화가
    상위 변화량에 포함되는지 확인

    Args:
        fi_baseline: 기준 Feature Importance
        fi_corrupted: 품질 문제 주입 후 Feature Importance
        corrupted_columns: 품질 문제가 주입된 컬럼 목록
        threshold_percentile: 상위 변화량 임계값 백분위

    Returns:
        탐지 정확도 지표
    """
    common_features = fi_baseline.index.intersection(fi_corrupted.index)

    # FI 변화량 계산
    fi_change = abs(fi_baseline[common_features] - fi_corrupted[common_features])
    fi_change_pct = fi_change / (fi_baseline[common_features] + 1e-10) * 100

    # 임계값 이상 변화한 특성
    threshold = np.percentile(fi_change_pct, threshold_percentile)
    detected_features = set(fi_change_pct[fi_change_pct >= threshold].index)

    # 실제 품질 문제 컬럼과 교집합
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
    """
    XAI 기법 간 일관성 비교

    Args:
        fi_dict: {method_name: fi_series} 형태의 딕셔너리
        baseline_method: 기준 방법 (None이면 첫 번째)

    Returns:
        방법 간 상관관계 매트릭스
    """
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
    """
    Cross-validation 결과에서 FI Divergence와 분산 계산

    Args:
        baseline_fi: Baseline FI from all folds, shape (n_folds, n_features)
        degraded_fi: Degraded FI from all folds, shape (n_folds, n_features)

    Returns:
        Dictionary with:
        - js_divergence_mean: Mean JS divergence across folds
        - js_divergence_std: Std of JS divergence
        - spearman_mean: Mean Spearman correlation
        - spearman_std: Std of Spearman correlation
        - rank_change_mean: Mean rank change ratio
        - rank_change_std: Std of rank change ratio

    Example:
        >>> baseline_fi = np.random.rand(10, 5)  # 10 folds, 5 features
        >>> degraded_fi = np.random.rand(10, 5)
        >>> metrics = compute_fi_divergence_with_variance(baseline_fi, degraded_fi)
        >>> print(metrics['js_divergence_mean'])
    """
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
    print("FI Divergence 지표 테스트")
    print("=" * 60)

    # 테스트용 데이터
    np.random.seed(42)
    features = [f'feature_{i}' for i in range(10)]

    # 기준 FI
    fi_baseline = pd.Series(
        np.random.rand(10) + 0.1,
        index=features
    )

    # 변화된 FI (일부 특성 변화)
    fi_corrupted = fi_baseline.copy()
    fi_corrupted['feature_0'] *= 2.0  # 큰 변화
    fi_corrupted['feature_1'] *= 0.5  # 큰 변화
    fi_corrupted['feature_2'] += 0.1  # 작은 변화

    print("\n[기준 FI]")
    print(fi_baseline.round(3))

    print("\n[변화된 FI]")
    print(fi_corrupted.round(3))

    # 지표 계산
    print("\n[FI Divergence Metrics]")
    metrics = calculate_fi_divergence_metrics(fi_baseline, fi_corrupted)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 탐지 정확도
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

    print("\n✅ 지표 테스트 완료!")
