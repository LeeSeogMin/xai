"""
Statistical Validation Framework
Phase 5: Experiments and Analysis

Improvement 2: Statistical Rigor
- 10-fold Cross-Validation
- Paired t-tests
- Confidence Intervals
- Effect Size (Cohen's d)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42


@dataclass
class CVResult:
    """Results from cross-validation"""
    fold_scores: np.ndarray  # (n_folds,) or (n_folds, n_features)
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


class StatisticalValidator:
    """
    Provides statistical validation for XAI experiments

    Features:
    - k-fold stratified cross-validation
    - Confidence interval estimation
    - Paired t-tests for significance testing
    - Effect size computation
    """

    def __init__(self, n_folds: int = 10, random_state: int = RANDOM_SEED):
        """
        Initialize validator

        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )

    def cross_validate_experiment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        experiment_func: Callable,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run experiment with cross-validation

        Args:
            X: Feature matrix
            y: Target labels
            experiment_func: Function that runs experiment on train/test split
                            Should return dict of metrics
            **kwargs: Additional arguments for experiment_func

        Returns:
            Dictionary mapping metric names to fold results

        Example:
            >>> def my_experiment(X_train, y_train, X_test, y_test):
            ...     # Train model, compute metrics
            ...     return {'accuracy': 0.85, 'f1': 0.80}
            >>> results = validator.cross_validate_experiment(X, y, my_experiment)
            >>> print(results['accuracy'])  # shape: (n_folds,)
        """
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Run experiment
            metrics = experiment_func(X_train, y_train, X_test, y_test, **kwargs)
            fold_results.append(metrics)

        # Aggregate results across folds
        aggregated = {}
        metric_names = fold_results[0].keys()

        for metric_name in metric_names:
            scores = np.array([fold[metric_name] for fold in fold_results])
            aggregated[metric_name] = scores

        return aggregated

    def compute_confidence_intervals(
        self,
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> CVResult:
        """
        Compute confidence intervals for scores

        Args:
            scores: Array of scores (e.g., from CV folds)
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            CVResult with mean, std, and CI bounds
        """
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # Sample std

        # Compute CI using t-distribution
        se = std / np.sqrt(n)  # Standard error
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * se

        return CVResult(
            fold_scores=scores,
            mean=mean,
            std=std,
            ci_lower=mean - margin,
            ci_upper=mean + margin,
            confidence_level=confidence
        )

    def paired_t_test(
        self,
        baseline_scores: np.ndarray,
        treatment_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Perform paired t-test (e.g., baseline vs degraded quality)

        Args:
            baseline_scores: Scores from baseline condition (n_folds,)
            treatment_scores: Scores from treatment condition (n_folds,)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with:
            - t_statistic: t-value
            - p_value: p-value
            - effect_size: Cohen's d
            - significant: bool (p < 0.05)
            - mean_diff: mean difference
        """
        assert len(baseline_scores) == len(treatment_scores), \
            "Baseline and treatment must have same length"

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(
            baseline_scores,
            treatment_scores,
            alternative=alternative
        )

        # Cohen's d for paired samples
        diff = baseline_scores - treatment_scores
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': cohens_d,
            'significant': p_value < 0.05,
            'mean_diff': mean_diff,
            'alternative': alternative
        }

    def aggregate_cv_results(
        self,
        cv_results: Dict[str, np.ndarray],
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Aggregate cross-validation results into summary table

        Args:
            cv_results: Dict mapping metric names to fold scores
            confidence: Confidence level for CIs

        Returns:
            DataFrame with columns: metric, mean, std, ci_lower, ci_upper
        """
        rows = []

        for metric_name, scores in cv_results.items():
            cv_result = self.compute_confidence_intervals(scores, confidence)
            rows.append({
                'metric': metric_name,
                'mean': cv_result.mean,
                'std': cv_result.std,
                'ci_lower': cv_result.ci_lower,
                'ci_upper': cv_result.ci_upper,
                'n_folds': len(scores)
            })

        return pd.DataFrame(rows)

    def compare_conditions(
        self,
        baseline_cv: Dict[str, np.ndarray],
        treatment_cv: Dict[str, np.ndarray],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare two experimental conditions using paired t-tests

        Args:
            baseline_cv: CV results from baseline condition
            treatment_cv: CV results from treatment condition
            metric_names: Metrics to compare (default: all common metrics)

        Returns:
            DataFrame with comparison statistics
        """
        if metric_names is None:
            metric_names = list(set(baseline_cv.keys()) & set(treatment_cv.keys()))

        rows = []

        for metric in metric_names:
            baseline_scores = baseline_cv[metric]
            treatment_scores = treatment_cv[metric]

            # Compute CIs
            baseline_ci = self.compute_confidence_intervals(baseline_scores)
            treatment_ci = self.compute_confidence_intervals(treatment_scores)

            # Paired t-test
            test_result = self.paired_t_test(baseline_scores, treatment_scores)

            rows.append({
                'metric': metric,
                'baseline_mean': baseline_ci.mean,
                'baseline_ci': f"[{baseline_ci.ci_lower:.4f}, {baseline_ci.ci_upper:.4f}]",
                'treatment_mean': treatment_ci.mean,
                'treatment_ci': f"[{treatment_ci.ci_lower:.4f}, {treatment_ci.ci_upper:.4f}]",
                'mean_diff': test_result['mean_diff'],
                'p_value': test_result['p_value'],
                'effect_size': test_result['effect_size'],
                'significant': '***' if test_result['p_value'] < 0.001 else
                              '**' if test_result['p_value'] < 0.01 else
                              '*' if test_result['p_value'] < 0.05 else 'ns'
            })

        return pd.DataFrame(rows)


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent groups

    Args:
        group1: Scores from group 1
        group2: Scores from group 2

    Returns:
        Cohen's d (standardized mean difference)
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = RANDOM_SEED
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval

    Args:
        data: Input data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        (statistic, ci_lower, ci_upper)
    """
    np.random.seed(random_state)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    observed_stat = statistic_func(data)

    return observed_stat, ci_lower, ci_upper


if __name__ == '__main__':
    # Test StatisticalValidator
    print("=" * 60)
    print("Statistical Validation Framework Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.randint(0, 2, n_samples))

    validator = StatisticalValidator(n_folds=5)

    # Define a simple experiment
    def dummy_experiment(X_train, y_train, X_test, y_test):
        # Simulate accuracy and F1 scores
        accuracy = 0.85 + np.random.randn() * 0.05
        f1 = 0.80 + np.random.randn() * 0.05
        return {'accuracy': accuracy, 'f1': f1}

    # Run CV
    print("\n[Running 5-fold CV...]")
    cv_results = validator.cross_validate_experiment(X, y, dummy_experiment)

    print(f"\nAccuracy scores: {cv_results['accuracy']}")
    print(f"F1 scores: {cv_results['f1']}")

    # Compute CIs
    print("\n[Confidence Intervals (95%)]")
    summary = validator.aggregate_cv_results(cv_results)
    print(summary.to_string(index=False))

    # Paired t-test
    print("\n[Paired t-test: Baseline vs Treatment]")
    baseline_scores = cv_results['accuracy']
    treatment_scores = baseline_scores - 0.02  # Simulate degradation

    test_result = validator.paired_t_test(baseline_scores, treatment_scores)
    print(f"Mean difference: {test_result['mean_diff']:.4f}")
    print(f"t-statistic: {test_result['t_statistic']:.4f}")
    print(f"p-value: {test_result['p_value']:.4f}")
    print(f"Cohen's d: {test_result['effect_size']:.4f}")
    print(f"Significant: {test_result['significant']}")

    print("\n✅ Statistical validation test completed!")
