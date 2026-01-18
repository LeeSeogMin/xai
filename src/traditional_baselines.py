"""
Traditional Quality Detection Baselines
Phase 5: Experiments and Analysis

Improvement 3: Comparison with Traditional Methods
- Z-score based outlier detection
- IQR based outlier detection
- Kolmogorov-Smirnov test for distribution shift
- Population Stability Index (PSI)
- Comparison metrics: Precision, Recall, NDCG
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')


class TraditionalQualityDetector:
    """
    Traditional statistical methods for data quality detection

    Methods:
    - Z-score outlier detection
    - IQR outlier detection
    - KS test for distribution shift
    - PSI for distribution shift
    - Missing value detection
    """

    def __init__(self, zscore_threshold: float = 3.0, psi_bins: int = 10):
        """
        Initialize detector

        Args:
            zscore_threshold: Z-score threshold for outlier detection (default: 3.0)
            psi_bins: Number of bins for PSI calculation (default: 10)
        """
        self.zscore_threshold = zscore_threshold
        self.psi_bins = psi_bins

    def detect_outliers_zscore(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Detect outliers using Z-score method

        Args:
            X: Feature matrix
            threshold: Z-score threshold (default: self.zscore_threshold)

        Returns:
            Dictionary with:
            - outlier_mask: Boolean mask (n_samples,) indicating outliers
            - outlier_scores_per_feature: Z-scores for each feature
            - affected_features: List of features with outliers
            - outlier_counts: Number of outliers per feature
        """
        if threshold is None:
            threshold = self.zscore_threshold

        # Compute Z-scores
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))

        # Detect outliers per feature
        outlier_mask_per_feature = z_scores > threshold

        # Overall outlier mask (any feature is outlier)
        outlier_mask = outlier_mask_per_feature.any(axis=1)

        # Count outliers per feature
        outlier_counts = outlier_mask_per_feature.sum(axis=0)

        # Identify affected features
        affected_features = X.columns[outlier_counts > 0].tolist()

        return {
            'outlier_mask': outlier_mask,
            'outlier_scores_per_feature': pd.DataFrame(
                z_scores, columns=X.columns, index=X.index
            ),
            'affected_features': affected_features,
            'outlier_counts': pd.Series(outlier_counts, index=X.columns),
            'total_outliers': outlier_mask.sum(),
            'method': 'zscore'
        }

    def detect_outliers_iqr(
        self,
        X: pd.DataFrame,
        iqr_multiplier: float = 1.5
    ) -> Dict[str, any]:
        """
        Detect outliers using IQR (Interquartile Range) method

        Args:
            X: Feature matrix
            iqr_multiplier: Multiplier for IQR (default: 1.5)

        Returns:
            Dictionary with outlier information (same structure as zscore)
        """
        outlier_mask_per_feature = np.zeros(X.shape, dtype=bool)

        for i, col in enumerate(X.columns):
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            outlier_mask_per_feature[:, i] = (
                (X[col] < lower_bound) | (X[col] > upper_bound)
            )

        # Overall outlier mask
        outlier_mask = outlier_mask_per_feature.any(axis=1)

        # Count outliers per feature
        outlier_counts = outlier_mask_per_feature.sum(axis=0)

        # Identify affected features
        affected_features = X.columns[outlier_counts > 0].tolist()

        return {
            'outlier_mask': outlier_mask,
            'outlier_mask_per_feature': pd.DataFrame(
                outlier_mask_per_feature, columns=X.columns, index=X.index
            ),
            'affected_features': affected_features,
            'outlier_counts': pd.Series(outlier_counts, index=X.columns),
            'total_outliers': outlier_mask.sum(),
            'method': 'iqr'
        }

    def detect_distribution_shift_ks(
        self,
        X_baseline: pd.DataFrame,
        X_degraded: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Detect distribution shift using Kolmogorov-Smirnov test

        Args:
            X_baseline: Baseline feature matrix
            X_degraded: Degraded feature matrix

        Returns:
            Dictionary with:
            - ks_statistics: KS statistic per feature
            - p_values: p-value per feature
            - affected_features: Features with significant shift (p < 0.05)
            - significant_shifts: Boolean mask for significant shifts
        """
        assert X_baseline.shape[1] == X_degraded.shape[1], \
            "Baseline and degraded must have same features"

        ks_statistics = {}
        p_values = {}

        for col in X_baseline.columns:
            ks_stat, p_value = stats.ks_2samp(
                X_baseline[col].dropna(),
                X_degraded[col].dropna()
            )
            ks_statistics[col] = ks_stat
            p_values[col] = p_value

        ks_statistics = pd.Series(ks_statistics)
        p_values = pd.Series(p_values)

        # Significant shifts (p < 0.05)
        significant_shifts = p_values < 0.05
        affected_features = p_values[significant_shifts].index.tolist()

        return {
            'ks_statistics': ks_statistics,
            'p_values': p_values,
            'affected_features': affected_features,
            'significant_shifts': significant_shifts,
            'n_affected': len(affected_features),
            'method': 'ks_test'
        }

    def detect_distribution_shift_psi(
        self,
        X_baseline: pd.DataFrame,
        X_degraded: pd.DataFrame,
        n_bins: Optional[int] = None,
        psi_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Detect distribution shift using Population Stability Index (PSI)

        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change

        Args:
            X_baseline: Baseline feature matrix
            X_degraded: Degraded feature matrix
            n_bins: Number of bins (default: self.psi_bins)
            psi_threshold: Threshold for significant shift (default: 0.1)

        Returns:
            Dictionary with PSI scores and affected features
        """
        if n_bins is None:
            n_bins = self.psi_bins

        assert X_baseline.shape[1] == X_degraded.shape[1], \
            "Baseline and degraded must have same features"

        psi_scores = {}

        for col in X_baseline.columns:
            baseline_col = X_baseline[col].dropna()
            degraded_col = X_degraded[col].dropna()

            # Create bins based on baseline
            _, bin_edges = np.histogram(baseline_col, bins=n_bins)

            # Count samples in each bin
            baseline_counts, _ = np.histogram(baseline_col, bins=bin_edges)
            degraded_counts, _ = np.histogram(degraded_col, bins=bin_edges)

            # Convert to proportions
            baseline_prop = baseline_counts / len(baseline_col)
            degraded_prop = degraded_counts / len(degraded_col)

            # Avoid division by zero
            baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
            degraded_prop = np.where(degraded_prop == 0, 0.0001, degraded_prop)

            # Calculate PSI
            psi = np.sum(
                (degraded_prop - baseline_prop) * np.log(degraded_prop / baseline_prop)
            )
            psi_scores[col] = psi

        psi_scores = pd.Series(psi_scores)

        # Significant shifts (PSI >= threshold)
        significant_shifts = psi_scores >= psi_threshold
        affected_features = psi_scores[significant_shifts].index.tolist()

        return {
            'psi_scores': psi_scores,
            'affected_features': affected_features,
            'significant_shifts': significant_shifts,
            'n_affected': len(affected_features),
            'threshold': psi_threshold,
            'method': 'psi'
        }

    def detect_missing_pattern(
        self,
        X: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, any]:
        """
        Detect missing value patterns

        Args:
            X: Feature matrix
            threshold: Minimum missing rate to consider (default: 0.05)

        Returns:
            Dictionary with missing value information
        """
        missing_counts = X.isnull().sum()
        missing_rates = missing_counts / len(X)

        # Features with missing values above threshold
        affected_features = missing_rates[missing_rates >= threshold].index.tolist()

        return {
            'missing_counts': missing_counts,
            'missing_rates': missing_rates,
            'affected_features': affected_features,
            'n_affected': len(affected_features),
            'total_missing': missing_counts.sum(),
            'method': 'missing_pattern'
        }


class BaselineComparator:
    """
    Compare XAI-based detection with traditional baseline methods

    Metrics:
    - Precision: Of detected features, how many are truly affected?
    - Recall: Of truly affected features, how many are detected?
    - NDCG: Ranking quality of detected features
    """

    def __init__(self):
        """Initialize comparator"""
        pass

    def compare_detection_accuracy(
        self,
        true_affected: List[str],
        xai_ranking: pd.Series,
        traditional_ranking: pd.Series,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Compare detection accuracy between XAI and traditional methods

        Args:
            true_affected: List of truly affected feature names
            xai_ranking: XAI importance scores (feature -> score)
            traditional_ranking: Traditional scores (feature -> score)
            top_k: Number of top features to consider

        Returns:
            Dictionary with comparison metrics
        """
        # Get top-k features from each method
        xai_top_k = set(xai_ranking.nlargest(top_k).index)
        trad_top_k = set(traditional_ranking.nlargest(top_k).index)

        true_affected_set = set(true_affected)

        # Compute precision and recall for XAI
        xai_tp = len(xai_top_k & true_affected_set)
        xai_precision = xai_tp / len(xai_top_k) if len(xai_top_k) > 0 else 0
        xai_recall = xai_tp / len(true_affected_set) if len(true_affected_set) > 0 else 0
        xai_f1 = (
            2 * xai_precision * xai_recall / (xai_precision + xai_recall)
            if (xai_precision + xai_recall) > 0 else 0
        )

        # Compute precision and recall for traditional method
        trad_tp = len(trad_top_k & true_affected_set)
        trad_precision = trad_tp / len(trad_top_k) if len(trad_top_k) > 0 else 0
        trad_recall = trad_tp / len(true_affected_set) if len(true_affected_set) > 0 else 0
        trad_f1 = (
            2 * trad_precision * trad_recall / (trad_precision + trad_recall)
            if (trad_precision + trad_recall) > 0 else 0
        )

        # Compute NDCG
        xai_ndcg = self._compute_ndcg(xai_ranking, true_affected, top_k)
        trad_ndcg = self._compute_ndcg(traditional_ranking, true_affected, top_k)

        return {
            'xai_precision': xai_precision,
            'xai_recall': xai_recall,
            'xai_f1': xai_f1,
            'xai_ndcg': xai_ndcg,
            'traditional_precision': trad_precision,
            'traditional_recall': trad_recall,
            'traditional_f1': trad_f1,
            'traditional_ndcg': trad_ndcg,
            'precision_gain': xai_precision - trad_precision,
            'recall_gain': xai_recall - trad_recall,
            'f1_gain': xai_f1 - trad_f1,
            'ndcg_gain': xai_ndcg - trad_ndcg,
            'top_k': top_k
        }

    def _compute_ndcg(
        self,
        ranking: pd.Series,
        true_affected: List[str],
        top_k: int
    ) -> float:
        """
        Compute NDCG (Normalized Discounted Cumulative Gain)

        Args:
            ranking: Feature importance scores
            true_affected: Truly affected features
            top_k: Number of top features

        Returns:
            NDCG score
        """
        # Get top-k features
        top_features = ranking.nlargest(top_k).index.tolist()

        # Create relevance vector (1 if affected, 0 otherwise)
        relevance = [1 if feat in true_affected else 0 for feat in top_features]

        # Pad if needed
        if len(relevance) < top_k:
            relevance.extend([0] * (top_k - len(relevance)))

        # Convert to numpy arrays (sklearn expects 2D)
        y_true = np.array([relevance])
        y_score = np.array([ranking[top_features].values])

        try:
            ndcg = ndcg_score(y_true, y_score, k=top_k)
        except:
            ndcg = 0.0

        return ndcg

    def compare_multiple_methods(
        self,
        true_affected: List[str],
        method_rankings: Dict[str, pd.Series],
        top_k_values: List[int] = [5, 10, 15]
    ) -> pd.DataFrame:
        """
        Compare multiple methods across different top-k values

        Args:
            true_affected: Truly affected features
            method_rankings: Dictionary mapping method name to ranking
            top_k_values: List of top-k values to evaluate

        Returns:
            DataFrame with comparison results
        """
        results = []

        for top_k in top_k_values:
            for method_name, ranking in method_rankings.items():
                top_features = set(ranking.nlargest(top_k).index)
                true_set = set(true_affected)

                tp = len(top_features & true_set)
                precision = tp / len(top_features) if len(top_features) > 0 else 0
                recall = tp / len(true_set) if len(true_set) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0
                )
                ndcg = self._compute_ndcg(ranking, true_affected, top_k)

                results.append({
                    'method': method_name,
                    'top_k': top_k,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ndcg': ndcg
                })

        return pd.DataFrame(results)


if __name__ == '__main__':
    # Test traditional detectors
    print("=" * 60)
    print("Traditional Quality Detectors Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Inject outliers in feature_0 and feature_1
    X.loc[:50, 'feature_0'] += 5  # Outliers
    X.loc[:30, 'feature_1'] += 6

    detector = TraditionalQualityDetector()

    # Test Z-score
    print("\n[Z-score Outlier Detection]")
    zscore_result = detector.detect_outliers_zscore(X)
    print(f"Total outliers: {zscore_result['total_outliers']}")
    print(f"Affected features: {zscore_result['affected_features']}")
    print(f"Outlier counts:\n{zscore_result['outlier_counts'][zscore_result['outlier_counts'] > 0]}")

    # Test IQR
    print("\n[IQR Outlier Detection]")
    iqr_result = detector.detect_outliers_iqr(X)
    print(f"Total outliers: {iqr_result['total_outliers']}")
    print(f"Affected features: {iqr_result['affected_features']}")

    # Test KS
    print("\n[KS Test for Distribution Shift]")
    X_degraded = X.copy()
    X_degraded['feature_0'] += 0.5  # Shift distribution
    X_degraded['feature_1'] += 0.3

    ks_result = detector.detect_distribution_shift_ks(X, X_degraded)
    print(f"Affected features: {ks_result['affected_features']}")
    print(f"KS statistics:\n{ks_result['ks_statistics'][ks_result['significant_shifts']]}")

    # Test PSI
    print("\n[PSI for Distribution Shift]")
    psi_result = detector.detect_distribution_shift_psi(X, X_degraded)
    print(f"Affected features: {psi_result['affected_features']}")
    print(f"PSI scores:\n{psi_result['psi_scores'][psi_result['significant_shifts']]}")

    # Test comparator
    print("\n[Baseline Comparator Test]")
    comparator = BaselineComparator()

    true_affected = ['feature_0', 'feature_1']

    # Simulate XAI ranking (feature_0 and feature_1 have high importance)
    xai_ranking = pd.Series(np.random.rand(n_features), index=X.columns)
    xai_ranking['feature_0'] = 0.9
    xai_ranking['feature_1'] = 0.85

    # Traditional ranking (Z-scores)
    trad_ranking = zscore_result['outlier_counts'] / zscore_result['outlier_counts'].sum()

    comparison = comparator.compare_detection_accuracy(
        true_affected, xai_ranking, trad_ranking, top_k=5
    )

    print("\nComparison Results:")
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ Traditional baselines test completed!")
