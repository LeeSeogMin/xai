"""
데이터 품질 문제 시뮬레이션 모듈
Phase 4: 데이터 품질 문제 주입

연구 설계에 따른 3가지 품질 문제 유형:
1. 결측치 (Missing Values - MCAR)
2. 이상치 (Outliers)
3. 분포 편향 (Distribution Shift)

심각도 수준: 5%, 10%, 20%, 30%
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional, Union, Tuple, List
from dataclasses import dataclass

# 재현성 확보
RANDOM_SEED = 42

# 심각도 수준
SEVERITY_LEVELS = [0.05, 0.10, 0.20, 0.30]  # 5%, 10%, 20%, 30%

# 품질 문제 유형
QualityIssueType = Literal['missing', 'outlier', 'distribution_shift']


@dataclass
class QualityIssueConfig:
    """품질 문제 설정"""
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
    """
    결측치 주입 (MCAR - Missing Completely at Random)

    Args:
        df: 원본 데이터프레임
        columns: 결측치를 주입할 컬럼 목록
        severity: 결측치 비율 (0.0 ~ 1.0)
        random_seed: 재현성을 위한 시드
        return_affected: True이면 영향받은 컬럼 목록도 반환 (default: False)

    Returns:
        결측치가 주입된 데이터프레임 (또는 (DataFrame, affected_columns) 튜플)
    """
    np.random.seed(random_seed)
    result = df.copy()

    n_samples = len(df)
    n_missing = int(n_samples * severity)

    affected_columns = []

    for col in columns:
        if col not in result.columns:
            continue

        # 무작위로 인덱스 선택
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
    """
    이상치 주입 (수치형 컬럼에만 적용)

    Args:
        df: 원본 데이터프레임
        columns: 이상치를 주입할 컬럼 목록
        severity: 이상치 비율 (0.0 ~ 1.0)
        method: 이상치 생성 방법 ('iqr' 또는 'zscore')
        multiplier: 이상치 강도 배수
        random_seed: 재현성을 위한 시드
        return_affected: True이면 영향받은 컬럼 목록도 반환 (default: False)

    Returns:
        이상치가 주입된 데이터프레임 (또는 (DataFrame, affected_columns) 튜플)
    """
    np.random.seed(random_seed)
    result = df.copy()
    # 수치형 컬럼을 float로 변환 (dtype 호환성)
    for col in columns:
        if col in result.columns and np.issubdtype(result[col].dtype, np.integer):
            result[col] = result[col].astype(float)

    n_samples = len(df)
    n_outliers = int(n_samples * severity)

    affected_columns = []

    for col in columns:
        if col not in result.columns:
            continue

        # 수치형 컬럼만 처리
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

        # 이상치 생성 (범위 밖 값)
        outlier_indices = np.random.choice(
            result.index,
            size=n_outliers,
            replace=False
        )

        # 상위/하위 이상치 무작위 배정
        for idx in outlier_indices:
            if np.random.random() > 0.5:
                # 상위 이상치
                result.loc[idx, col] = upper_bound + np.random.exponential(IQR if method == 'iqr' else std)
            else:
                # 하위 이상치
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
    """
    분포 편향 주입

    Args:
        df: 원본 데이터프레임
        columns: 편향을 주입할 컬럼 목록 (covariate_shift용)
        severity: 편향 정도 (0.0 ~ 1.0)
        shift_type: 편향 유형
            - 'class_imbalance': 타겟 클래스 불균형 증가
            - 'covariate_shift': 특정 특성의 분포 변경
        target_column: 타겟 컬럼명
        random_seed: 재현성을 위한 시드
        return_affected: True이면 영향받은 컬럼 목록도 반환 (default: False)

    Returns:
        분포 편향이 주입된 데이터프레임 (또는 (DataFrame, affected_columns) 튜플)
    """
    np.random.seed(random_seed)
    result = df.copy()
    # 수치형 컬럼을 float로 변환 (dtype 호환성)
    for col in columns:
        if col in result.columns and np.issubdtype(result[col].dtype, np.integer):
            result[col] = result[col].astype(float)

    affected_columns = []

    if shift_type == 'class_imbalance':
        # 클래스 불균형 증가: 소수 클래스 샘플 제거
        if target_column not in result.columns:
            if return_affected:
                return result, affected_columns
            return result

        value_counts = result[target_column].value_counts()
        minority_class = value_counts.idxmin()
        minority_indices = result[result[target_column] == minority_class].index

        # severity 비율만큼 소수 클래스 제거
        n_remove = int(len(minority_indices) * severity)
        remove_indices = np.random.choice(minority_indices, size=n_remove, replace=False)
        result = result.drop(remove_indices).reset_index(drop=True)

        affected_columns = [target_column]

    elif shift_type == 'covariate_shift':
        # 특정 특성 분포 변경 (평균/분산 조정)
        n_samples = len(df)
        n_shift = int(n_samples * severity)

        for col in columns:
            if col not in result.columns:
                continue

            if not np.issubdtype(result[col].dtype, np.number):
                continue

            # 무작위 샘플의 값을 분포 끝쪽으로 이동
            shift_indices = np.random.choice(
                result.index,
                size=n_shift,
                replace=False
            )

            col_mean = result[col].mean()
            col_std = result[col].std()

            # 한쪽 방향으로 편향
            shift_amount = col_std * 2  # 2 표준편차만큼 이동
            result.loc[shift_indices, col] = result.loc[shift_indices, col] + shift_amount

            affected_columns.append(col)

    if return_affected:
        return result, affected_columns
    return result


def create_quality_variants(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    severity_levels: list[float] = None,
    random_seed: int = RANDOM_SEED
) -> dict:
    """
    모든 품질 문제 조합 생성

    Args:
        df: 원본 (클린) 데이터프레임
        numerical_cols: 수치형 컬럼 목록
        categorical_cols: 범주형 컬럼 목록
        severity_levels: 심각도 수준 목록
        random_seed: 재현성을 위한 시드

    Returns:
        dict: {(issue_type, severity): DataFrame} 형태의 딕셔너리
    """
    if severity_levels is None:
        severity_levels = SEVERITY_LEVELS

    variants = {
        ('clean', 0.0): df.copy()  # 기준 클린 데이터
    }

    for severity in severity_levels:
        # 결측치 주입 (수치형 + 범주형 모두)
        all_cols = numerical_cols + categorical_cols
        variants[('missing', severity)] = inject_missing_values(
            df, all_cols, severity, random_seed
        )

        # 이상치 주입 (수치형만)
        variants[('outlier', severity)] = inject_outliers(
            df, numerical_cols, severity, random_seed=random_seed
        )

        # 분포 편향 주입 (수치형만)
        variants[('distribution_shift', severity)] = inject_distribution_shift(
            df, numerical_cols, severity,
            shift_type='covariate_shift', random_seed=random_seed
        )

    return variants


def get_quality_metrics(original: pd.DataFrame, modified: pd.DataFrame, columns: list[str]) -> dict:
    """
    품질 변화 측정

    Args:
        original: 원본 데이터프레임
        modified: 수정된 데이터프레임
        columns: 비교할 컬럼 목록

    Returns:
        품질 지표 딕셔너리
    """
    metrics = {}

    for col in columns:
        if col not in original.columns or col not in modified.columns:
            continue

        col_metrics = {}

        # 결측치 비율
        col_metrics['missing_rate_original'] = original[col].isnull().mean()
        col_metrics['missing_rate_modified'] = modified[col].isnull().mean()

        # 수치형 컬럼 통계
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
    print("데이터 품질 문제 시뮬레이션 테스트")
    print("=" * 60)

    # 데이터 로드 및 클리닝
    train_df, _ = load_raw_data()
    clean_df = clean_data(train_df)
    print(f"\n[클린 데이터] Shape: {clean_df.shape}")

    # 결측치 주입 테스트
    print("\n[1. 결측치 주입 테스트]")
    for severity in SEVERITY_LEVELS:
        missing_df = inject_missing_values(clean_df, NUMERICAL_COLS[:3], severity)
        actual_missing = missing_df[NUMERICAL_COLS[:3]].isnull().sum().sum()
        expected = len(clean_df) * severity * 3
        print(f"  심각도 {severity*100:.0f}%: 결측치 {actual_missing}개 (예상: ~{expected:.0f})")

    # 이상치 주입 테스트
    print("\n[2. 이상치 주입 테스트]")
    for severity in SEVERITY_LEVELS:
        outlier_df = inject_outliers(clean_df, ['age', 'hours_per_week'], severity)
        # IQR 기준 이상치 개수 확인
        for col in ['age', 'hours_per_week']:
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((outlier_df[col] < Q1 - 1.5*IQR) | (outlier_df[col] > Q3 + 1.5*IQR)).sum()
            print(f"  심각도 {severity*100:.0f}% - {col}: {outliers}개 이상치")

    # 분포 편향 주입 테스트
    print("\n[3. 분포 편향 주입 테스트]")
    for severity in SEVERITY_LEVELS:
        shifted_df = inject_distribution_shift(
            clean_df, ['age', 'hours_per_week'], severity,
            shift_type='covariate_shift'
        )
        for col in ['age', 'hours_per_week']:
            mean_diff = shifted_df[col].mean() - clean_df[col].mean()
            print(f"  심각도 {severity*100:.0f}% - {col}: 평균 변화 {mean_diff:.2f}")

    # 전체 품질 변종 생성
    print("\n[4. 전체 품질 변종 생성]")
    variants = create_quality_variants(clean_df, NUMERICAL_COLS, CATEGORICAL_COLS)
    print(f"  생성된 변종 수: {len(variants)}")
    for key in list(variants.keys())[:5]:
        print(f"    - {key}: {variants[key].shape}")

    print("\n✅ 품질 문제 시뮬레이션 테스트 완료!")
