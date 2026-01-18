"""
Real Quality Problem Analyzer
Phase 5: Experiments and Analysis

Improvement 4: Real Quality Problem Validation
- Validate XAI-based detection on real-world quality issues
- Compare with simulated quality problems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class RealQualityAnalyzer:
    """
    Analyze XAI's ability to detect real quality problems

    Validates that XAI-based feature importance divergence can identify:
    - Natural missing values (Ames Housing)
    - Distribution shift (KDD Cup 99)
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize analyzer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def validate_xai_diagnosis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        known_issues: Dict[str, List[str]],
        xai_method: str = 'feature_importances',
        model_class = RandomForestClassifier,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Validate XAI's ability to detect known quality issues

        Args:
            X: Feature matrix
            y: Target labels
            known_issues: Dictionary mapping issue type to affected features
                         e.g., {'missing': ['PoolQC', 'Alley']}
            xai_method: XAI method to use (default: 'feature_importances')
            model_class: Model class to use
            top_k: Number of top features to consider

        Returns:
            Dictionary with detection accuracy metrics
        """
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Handle missing values for model training (fill with median/mode)
        X_train_filled = X_train.copy()
        X_test_filled = X_test.copy()

        for col in X_train.columns:
            if X_train[col].dtype in [np.float64, np.int64]:
                median_val = X_train[col].median()
                X_train_filled[col] = X_train[col].fillna(median_val)
                X_test_filled[col] = X_test[col].fillna(median_val)
            else:
                mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'unknown'
                X_train_filled[col] = X_train[col].fillna(mode_val)
                X_test_filled[col] = X_test[col].fillna(mode_val)

        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        for col in X_train_filled.columns:
            if X_train_filled[col].dtype == 'object':
                le = LabelEncoder()
                X_train_filled[col] = le.fit_transform(X_train_filled[col].astype(str))
                X_test_filled[col] = X_test_filled[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                label_encoders[col] = le

        # Train model
        model = model_class(random_state=self.random_state)
        model.fit(X_train_filled, y_train)

        # Get feature importance
        if xai_method == 'feature_importances':
            feature_importance = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            )
        else:
            raise ValueError(f"XAI method '{xai_method}' not supported yet")

        # Get top-k features by importance
        top_features = set(feature_importance.nlargest(top_k).index)

        # Evaluate detection accuracy for each issue type
        results = {}

        for issue_type, affected_features in known_issues.items():
            affected_set = set(affected_features)

            # True positives: Top-k features that are actually affected
            tp = len(top_features & affected_set)

            # False positives: Top-k features that are not affected
            fp = len(top_features - affected_set)

            # False negatives: Affected features not in top-k
            fn = len(affected_set - top_features)

            # Metrics
            precision = tp / len(top_features) if len(top_features) > 0 else 0
            recall = tp / len(affected_set) if len(affected_set) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[f'{issue_type}_detection_precision'] = precision
            results[f'{issue_type}_detection_recall'] = recall
            results[f'{issue_type}_detection_f1'] = f1
            results[f'{issue_type}_tp'] = tp
            results[f'{issue_type}_fp'] = fp
            results[f'{issue_type}_fn'] = fn

        # Overall detection (any issue type)
        all_affected = set()
        for features in known_issues.values():
            all_affected.update(features)

        overall_tp = len(top_features & all_affected)
        overall_precision = overall_tp / len(top_features) if len(top_features) > 0 else 0
        overall_recall = overall_tp / len(all_affected) if len(all_affected) > 0 else 0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0
        )

        results['overall_detection_precision'] = overall_precision
        results['overall_detection_recall'] = overall_recall
        results['overall_detection_f1'] = overall_f1
        results['top_k'] = top_k
        results['n_known_issues'] = len(all_affected)

        return results

    def compare_simulated_vs_real(
        self,
        simulated_results: Dict[str, float],
        real_results: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compare detection performance on simulated vs real quality problems

        Args:
            simulated_results: Results from simulated quality experiments
            real_results: Results from real quality validation

        Returns:
            DataFrame comparing simulated and real performance
        """
        comparison = []

        # Extract common metrics
        metrics = ['precision', 'recall', 'f1']

        for metric in metrics:
            sim_key = f'detection_{metric}'
            real_key = f'overall_detection_{metric}'

            sim_value = simulated_results.get(sim_key, np.nan)
            real_value = real_results.get(real_key, np.nan)

            comparison.append({
                'metric': metric,
                'simulated': sim_value,
                'real': real_value,
                'difference': real_value - sim_value,
                'relative_diff': (real_value - sim_value) / sim_value if sim_value > 0 else np.nan
            })

        return pd.DataFrame(comparison)

    def analyze_missing_value_detection(
        self,
        X: pd.DataFrame,
        feature_importance: pd.Series,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Analyze whether high-importance features have high missing rates

        Hypothesis: XAI should assign low importance to features with many missing values
        (because they provide less information)

        Args:
            X: Feature matrix (with missing values)
            feature_importance: Feature importance scores
            top_k: Number of top features to analyze

        Returns:
            Dictionary with analysis results
        """
        # Compute missing rates
        missing_rates = X.isnull().mean()

        # Get top-k features
        top_features = feature_importance.nlargest(top_k).index

        # Average missing rate in top-k
        avg_missing_top_k = missing_rates[top_features].mean()

        # Average missing rate in bottom features
        bottom_features = feature_importance.nsmallest(len(feature_importance) - top_k).index
        avg_missing_bottom = missing_rates[bottom_features].mean()

        # Correlation between importance and missing rate
        correlation = feature_importance.corr(missing_rates)

        return {
            'avg_missing_top_k': avg_missing_top_k,
            'avg_missing_bottom': avg_missing_bottom,
            'missing_rate_difference': avg_missing_bottom - avg_missing_top_k,
            'importance_missing_correlation': correlation,
            'interpretation': (
                'Negative correlation expected: high importance → low missing rate'
                if correlation < 0 else
                'Unexpected positive correlation'
            )
        }


if __name__ == '__main__':
    # Test analyzer
    print("=" * 60)
    print("Real Quality Analyzer Test")
    print("=" * 60)

    # Generate synthetic data with known issues
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Inject missing values in feature_0 and feature_1
    X.loc[:300, 'feature_0'] = np.nan
    X.loc[:200, 'feature_1'] = np.nan

    # Create target (depends on feature_5 and feature_6)
    y = (X['feature_5'] + X['feature_6'] > 0).astype(int)

    # Known issues
    known_issues = {
        'missing': ['feature_0', 'feature_1']
    }

    analyzer = RealQualityAnalyzer()

    # Test validation
    print("\n[Testing XAI Diagnosis Validation]")
    results = analyzer.validate_xai_diagnosis(
        X, y, known_issues, top_k=5
    )

    print("\nDetection Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test missing value analysis
    print("\n[Testing Missing Value Analysis]")
    # Compute feature importances (simplified)
    X_filled = X.fillna(X.median())
    model = RandomForestClassifier(random_state=42)
    model.fit(X_filled, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)

    missing_analysis = analyzer.analyze_missing_value_detection(
        X, feature_importance, top_k=5
    )

    print("\nMissing Value Analysis:")
    for key, value in missing_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ Real quality analyzer test completed!")
