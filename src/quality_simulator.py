'Utilities for injecting controlled data quality problems into tabular data.'

import pandas as pd
import numpy as np
from typing import Literal, Optional, Union, Tuple, List
from dataclasses import dataclass


RANDOM_SEED = 42


SEVERITY_LEVELS = [0.05, 0.10, 0.20, 0.30]  # 5%, 10%, 20%, 30%


QualityIssueType = Literal['missing', 'outlier', 'distribution_shift']


@dataclass
class QualityIssueConfig:
    'QualityIssueConfig helper class.'
    issue_type: QualityIssueType
    severity: float  # 0.0 ~ 1.0
    target_columns: list[str]
    random_seed: int = RANDOM_SEED


def inject_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    severity: float,
    random_seed: int = RANDOM_SEED,
    return_affected: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    'Utilities for injecting controlled data quality problems into tabular data.'
    np.random.seed(random_seed)
    result = df.copy()

    n_samples = len(df)
    n_missing = int(n_samples * severity)

    affected_columns = []

    for col in columns:
        if col not in result.columns:
            continue


        missing_indices = np.random.choice(
            result.index,
            size=n_missing,
            replace=False
        )
        result.loc[missing_indices, col] = np.nan
        affected_columns.append(col)

    if return_affected:
        return result, affected_columns
    return result


def inject_outliers(
    df: pd.DataFrame,
    columns: list[str],
    severity: float,
    method: Literal['iqr', 'zscore'] = 'iqr',
    multiplier: float = 3.0,
    random_seed: int = RANDOM_SEED,
    return_affected: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    'Utilities for injecting controlled data quality problems into tabular data.'
    np.random.seed(random_seed)
    result = df.copy()

    for col in columns:
        if col in result.columns and np.issubdtype(result[col].dtype, np.integer):
            result[col] = result[col].astype(float)

    n_samples = len(df)
    n_outliers = int(n_samples * severity)

    affected_columns = []

    for col in columns:
        if col not in result.columns:
            continue


        if not np.issubdtype(result[col].dtype, np.number):
            continue

        col_data = result[col].dropna()

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        else:  # zscore
            mean = col_data.mean()
            std = col_data.std()
            lower_bound = mean - multiplier * std
            upper_bound = mean + multiplier * std


        outlier_indices = np.random.choice(
            result.index,
            size=n_outliers,
            replace=False
        )


        for idx in outlier_indices:
            if np.random.random() > 0.5:

                result.loc[idx, col] = upper_bound + np.random.exponential(IQR if method == 'iqr' else std)
            else:

                result.loc[idx, col] = lower_bound - np.random.exponential(IQR if method == 'iqr' else std)

        affected_columns.append(col)

    if return_affected:
        return result, affected_columns
    return result


def inject_distribution_shift(
    df: pd.DataFrame,
    columns: list[str],
    severity: float,
    shift_type: Literal['class_imbalance', 'covariate_shift'] = 'class_imbalance',
    target_column: str = 'income',
    random_seed: int = RANDOM_SEED,
    return_affected: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    'Utilities for injecting controlled data quality problems into tabular data.'
    np.random.seed(random_seed)
    result = df.copy()

    for col in columns:
        if col in result.columns and np.issubdtype(result[col].dtype, np.integer):
            result[col] = result[col].astype(float)

    affected_columns = []

    if shift_type == 'class_imbalance':

        if target_column not in result.columns:
            if return_affected:
                return result, affected_columns
            return result

        value_counts = result[target_column].value_counts()
        minority_class = value_counts.idxmin()
        minority_indices = result[result[target_column] == minority_class].index


        n_remove = int(len(minority_indices) * severity)
        remove_indices = np.random.choice(minority_indices, size=n_remove, replace=False)
        result = result.drop(remove_indices).reset_index(drop=True)

        affected_columns = [target_column]

    elif shift_type == 'covariate_shift':

        n_samples = len(df)
        n_shift = int(n_samples * severity)

        for col in columns:
            if col not in result.columns:
                continue

            if not np.issubdtype(result[col].dtype, np.number):
                continue


            shift_indices = np.random.choice(
                result.index,
                size=n_shift,
                replace=False
            )

            col_mean = result[col].mean()
            col_std = result[col].std()


            shift_amount = col_std * 2
            result.loc[shift_indices, col] = result.loc[shift_indices, col] + shift_amount

            affected_columns.append(col)

    if return_affected:
        return result, affected_columns
    return result



class QualitySimulator:
    """Object-oriented wrapper around quality injection functions."""

    def inject_missing(self, df, columns=None, severity=0.2, mechanism='mcar'):
        """Inject missing values. The public wrapper currently supports MCAR."""
        if columns is None:
            columns = list(df.columns)
        if mechanism.lower() != 'mcar':
            raise ValueError("QualitySimulator.inject_missing currently supports mechanism='mcar'.")
        return inject_missing_values(df, columns, severity)

    def inject_outlier(self, df, columns=None, severity=0.2, method='iqr'):
        """Inject outliers into selected columns."""
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)
        return inject_outliers(df, columns, severity, method=method)

    def inject_distribution_shift(self, df, columns=None, severity=0.2):
        """Inject covariate shift into selected columns."""
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)
        return inject_distribution_shift(df, columns, severity, shift_type='covariate_shift')


def create_quality_variants(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    severity_levels: list[float] = None,
    random_seed: int = RANDOM_SEED
) -> dict:
    'Utilities for injecting controlled data quality problems into tabular data.'
    if severity_levels is None:
        severity_levels = SEVERITY_LEVELS

    variants = {
        ('clean', 0.0): df.copy()
    }

    for severity in severity_levels:

        all_cols = numerical_cols + categorical_cols
        variants[('missing', severity)] = inject_missing_values(
            df, all_cols, severity, random_seed
        )


        variants[('outlier', severity)] = inject_outliers(
            df, numerical_cols, severity, random_seed=random_seed
        )


        variants[('distribution_shift', severity)] = inject_distribution_shift(
            df, numerical_cols, severity,
            shift_type='covariate_shift', random_seed=random_seed
        )

    return variants


def get_quality_metrics(original: pd.DataFrame, modified: pd.DataFrame, columns: list[str]) -> dict:
    'Get quality metrics.'
    metrics = {}

    for col in columns:
        if col not in original.columns or col not in modified.columns:
            continue

        col_metrics = {}


        col_metrics['missing_rate_original'] = original[col].isnull().mean()
        col_metrics['missing_rate_modified'] = modified[col].isnull().mean()


        if np.issubdtype(original[col].dtype, np.number):
            orig_clean = original[col].dropna()
            mod_clean = modified[col].dropna()

            if len(mod_clean) > 0:
                col_metrics['mean_diff'] = mod_clean.mean() - orig_clean.mean()
                col_metrics['std_diff'] = mod_clean.std() - orig_clean.std()
                col_metrics['median_diff'] = mod_clean.median() - orig_clean.median()

        metrics[col] = col_metrics

    return metrics


if __name__ == '__main__':
    from data_loader import load_raw_data, clean_data, NUMERICAL_COLS, CATEGORICAL_COLS

    print("=" * 60)
    print("Data quality problem simulation test")
    print("=" * 60)


    train_df, _ = load_raw_data()
    clean_df = clean_data(train_df)
    print(f"\n[Clean data] Shape: {clean_df.shape}")


    print("\n[1. Missing-value injection test]")
    for severity in SEVERITY_LEVELS:
        missing_df = inject_missing_values(clean_df, NUMERICAL_COLS[:3], severity)
        actual_missing = missing_df[NUMERICAL_COLS[:3]].isnull().sum().sum()
        expected = len(clean_df) * severity * 3
        print(f"  Severity {severity*100:.0f}%: {actual_missing} missing values (expected: ~{expected:.0f})")


    print("\n[2. Outlier injection test]")
    for severity in SEVERITY_LEVELS:
        outlier_df = inject_outliers(clean_df, ['age', 'hours_per_week'], severity)

        for col in ['age', 'hours_per_week']:
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((outlier_df[col] < Q1 - 1.5*IQR) | (outlier_df[col] > Q3 + 1.5*IQR)).sum()
            print(f"  Severity {severity*100:.0f}% - {col}: {outliers} outliers")


    print("\n[3. Distribution-shift injection test]")
    for severity in SEVERITY_LEVELS:
        shifted_df = inject_distribution_shift(
            clean_df, ['age', 'hours_per_week'], severity,
            shift_type='covariate_shift'
        )
        for col in ['age', 'hours_per_week']:
            mean_diff = shifted_df[col].mean() - clean_df[col].mean()
            print(f"  Severity {severity*100:.0f}% - {col}: mean change {mean_diff:.2f}")


    print("\n[4. Generate all quality variants]")
    variants = create_quality_variants(clean_df, NUMERICAL_COLS, CATEGORICAL_COLS)
    print(f"  Generated variants: {len(variants)}")
    for key in list(variants.keys())[:5]:
        print(f"    - {key}: {variants[key].shape}")

    print("\n✅ Quality simulation test complete!")
