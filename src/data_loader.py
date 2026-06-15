'Dataset loading and preprocessing utilities for the UCI Adult dataset.'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from pathlib import Path


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


NUMERICAL_COLS = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country']


def load_raw_data(data_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    'Load raw data.'
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    else:
        data_dir = Path(data_dir)


    train_df = pd.read_csv(
        data_dir / 'adult.data',
        names=COLUMN_NAMES,
        sep=r',\s*',
        engine='python',
        na_values='?'
    )


    test_df = pd.read_csv(
        data_dir / 'adult.test',
        names=COLUMN_NAMES,
        sep=r',\s*',
        engine='python',
        na_values='?',
        skiprows=1
    )


    test_df['income'] = test_df['income'].str.rstrip('.')

    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    'Clean raw data by handling missing values.'

    clean_df = df.dropna().reset_index(drop=True)
    return clean_df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    'Encode features.'
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

                le = encoders[col]
                encoded_df[col] = encoded_df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return encoded_df, encoders


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    'Scale features.'
    scaled_df = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaled_df[NUMERICAL_COLS] = scaler.fit_transform(scaled_df[NUMERICAL_COLS])
    else:
        scaled_df[NUMERICAL_COLS] = scaler.transform(scaled_df[NUMERICAL_COLS])

    return scaled_df, scaler


def prepare_dataset(test_size: float = 0.2, scale: bool = True) -> dict:
    'Run the full dataset preparation pipeline.'

    train_df, test_df = load_raw_data()


    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)


    train_data, val_data = train_test_split(
        train_clean,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=train_clean['income']
    )


    train_encoded, encoders = encode_features(train_data, fit=True)
    val_encoded, _ = encode_features(val_data, encoders=encoders, fit=False)
    test_encoded, _ = encode_features(test_clean, encoders=encoders, fit=False)


    scaler = None
    if scale:
        train_encoded, scaler = scale_features(train_encoded, fit=True)
        val_encoded, _ = scale_features(val_encoded, scaler=scaler, fit=False)
        test_encoded, _ = scale_features(test_encoded, scaler=scaler, fit=False)


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
        'train_raw': train_data,
    }

    return result


def get_dataset_stats(df: pd.DataFrame) -> dict:
    'Get dataset stats.'
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



def load_dataset(name: str = 'uci_adult') -> tuple[pd.DataFrame, pd.Series]:
    """Load a registered dataset by name."""
    if name in {'adult', 'uci_adult'}:
        return load_adult_dataset()
    try:
        from .dataset_registry import create_default_registry
    except ImportError:
        from dataset_registry import create_default_registry
    return create_default_registry().get_dataset(name)

def initialize_registry():
    """
    Initialize dataset registry with all available datasets

    Returns:
        DatasetRegistry with registered datasets (3 datasets)
        - UCI Adult
        - UNSW-NB15
        - Credit Card Fraud
    """
    try:
        from .dataset_registry import create_default_registry
    except ImportError:
        from dataset_registry import create_default_registry

    # Create registry with all datasets
    registry = create_default_registry()

    return registry


if __name__ == '__main__':

    print("=" * 60)
    print('UCI Adult loading and preprocessing test')
    print("=" * 60)


    train_df, test_df = load_raw_data()
    print("\n[Raw data]")
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")


    print("\n[Missing values in training data]")
    missing = train_df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(train_df)*100:.2f}%)")


    print("\n[Running preprocessing pipeline]")
    data = prepare_dataset()

    print("\n[Preprocessing results]")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"Number of features: {len(data['feature_names'])}")
    print(f"Feature list: {data['feature_names']}")

    print("\n[Target distribution in training data]")
    print(data['y_train'].value_counts())

    print('Data loading and preprocessing test complete!')
