"""
UCI Adult 데이터셋 로딩 및 전처리 모듈
Phase 4: 데이터 수집 및 전처리
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from pathlib import Path

# 재현성 확보
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 컬럼 정의 (adult.names 파일 기반)
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# 수치형/범주형 컬럼 구분
NUMERICAL_COLS = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country']


def load_raw_data(data_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    원본 UCI Adult 데이터 로딩

    Returns:
        train_df: 훈련 데이터
        test_df: 테스트 데이터
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / 'raw'
    else:
        data_dir = Path(data_dir)

    # 훈련 데이터 로드
    train_df = pd.read_csv(
        data_dir / 'adult.data',
        names=COLUMN_NAMES,
        sep=r',\s*',
        engine='python',
        na_values='?'
    )

    # 테스트 데이터 로드 (첫 번째 행은 설명이므로 스킵)
    test_df = pd.read_csv(
        data_dir / 'adult.test',
        names=COLUMN_NAMES,
        sep=r',\s*',
        engine='python',
        na_values='?',
        skiprows=1
    )

    # 테스트 데이터의 타겟 레이블 정리 (끝에 '.' 제거)
    test_df['income'] = test_df['income'].str.rstrip('.')

    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 데이터 클리닝 (결측치 처리)

    Args:
        df: 원본 데이터프레임

    Returns:
        클리닝된 데이터프레임 (결측치 있는 행 제거)
    """
    # 결측치 있는 행 제거 (baseline clean dataset 생성)
    clean_df = df.dropna().reset_index(drop=True)
    return clean_df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    범주형 변수 인코딩

    Args:
        df: 데이터프레임
        encoders: 기존 인코더 딕셔너리 (None이면 새로 생성)
        fit: True면 fit_transform, False면 transform만

    Returns:
        encoded_df: 인코딩된 데이터프레임
        encoders: 사용된 인코더 딕셔너리
    """
    encoded_df = df.copy()

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS + ['income']:
        if col not in encoded_df.columns:
            continue

        if fit:
            if col not in encoders:
                encoders[col] = LabelEncoder()
            encoded_df[col] = encoders[col].fit_transform(encoded_df[col].astype(str))
        else:
            if col in encoders:
                # 테스트 데이터에 없는 카테고리 처리
                le = encoders[col]
                encoded_df[col] = encoded_df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return encoded_df, encoders


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    수치형 변수 스케일링

    Args:
        df: 데이터프레임
        scaler: 기존 스케일러 (None이면 새로 생성)
        fit: True면 fit_transform, False면 transform만

    Returns:
        scaled_df: 스케일링된 데이터프레임
        scaler: 사용된 스케일러
    """
    scaled_df = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaled_df[NUMERICAL_COLS] = scaler.fit_transform(scaled_df[NUMERICAL_COLS])
    else:
        scaled_df[NUMERICAL_COLS] = scaler.transform(scaled_df[NUMERICAL_COLS])

    return scaled_df, scaler


def prepare_dataset(test_size: float = 0.2, scale: bool = True) -> dict:
    """
    전체 데이터셋 준비 파이프라인

    Args:
        test_size: 테스트 데이터 비율
        scale: 스케일링 적용 여부

    Returns:
        dict: 준비된 데이터셋 및 메타데이터
    """
    # 1. 원본 데이터 로드
    train_df, test_df = load_raw_data()

    # 2. 데이터 클리닝 (baseline clean dataset)
    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)

    # 3. 훈련/검증 분할
    train_data, val_data = train_test_split(
        train_clean,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=train_clean['income']
    )

    # 4. 인코딩
    train_encoded, encoders = encode_features(train_data, fit=True)
    val_encoded, _ = encode_features(val_data, encoders=encoders, fit=False)
    test_encoded, _ = encode_features(test_clean, encoders=encoders, fit=False)

    # 5. 스케일링 (옵션)
    scaler = None
    if scale:
        train_encoded, scaler = scale_features(train_encoded, fit=True)
        val_encoded, _ = scale_features(val_encoded, scaler=scaler, fit=False)
        test_encoded, _ = scale_features(test_encoded, scaler=scaler, fit=False)

    # 6. X, y 분리
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS

    result = {
        'X_train': train_encoded[feature_cols],
        'y_train': train_encoded['income'],
        'X_val': val_encoded[feature_cols],
        'y_val': val_encoded['income'],
        'X_test': test_encoded[feature_cols],
        'y_test': test_encoded['income'],
        'feature_names': feature_cols,
        'encoders': encoders,
        'scaler': scaler,
        'train_raw': train_data,  # 품질 문제 시뮬레이션용 원본
    }

    return result


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    데이터셋 기초 통계 정보
    """
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numerical_stats': df[NUMERICAL_COLS].describe().to_dict(),
        'categorical_value_counts': {col: df[col].value_counts().to_dict()
                                      for col in CATEGORICAL_COLS if col in df.columns},
        'target_distribution': df['income'].value_counts().to_dict() if 'income' in df.columns else None
    }
    return stats


def load_adult_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load UCI Adult dataset for DatasetRegistry

    Returns:
        X: Features (numerical + categorical, preprocessed)
        y: Target (binary: 0=<=50K, 1=>50K)
    """
    data = prepare_dataset()

    # Combine train and test for registry
    X = pd.concat([data['X_train'], data['X_test']], axis=0).reset_index(drop=True)
    y = pd.concat([data['y_train'], data['y_test']], axis=0).reset_index(drop=True)

    return X, y


def initialize_registry():
    """
    Initialize dataset registry with all available datasets

    Returns:
        DatasetRegistry with registered datasets (3 datasets)
        - UCI Adult
        - UNSW-NB15
        - Credit Card Fraud
    """
    from phase4_data.dataset_registry import create_default_registry

    # Create registry with all datasets
    registry = create_default_registry()

    return registry


if __name__ == '__main__':
    # 테스트 실행
    print("=" * 60)
    print("UCI Adult 데이터셋 로딩 및 전처리 테스트")
    print("=" * 60)

    # 원본 데이터 로드
    train_df, test_df = load_raw_data()
    print(f"\n[원본 데이터]")
    print(f"훈련 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")

    # 결측치 확인
    print(f"\n[결측치 현황 - 훈련 데이터]")
    missing = train_df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(train_df)*100:.2f}%)")

    # 전체 파이프라인 실행
    print("\n[전처리 파이프라인 실행]")
    data = prepare_dataset()

    print(f"\n[전처리 결과]")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"특성 개수: {len(data['feature_names'])}")
    print(f"특성 목록: {data['feature_names']}")

    print(f"\n[타겟 분포 - 훈련 데이터]")
    print(data['y_train'].value_counts())

    print("\n✅ 데이터 로딩 및 전처리 테스트 완료!")
