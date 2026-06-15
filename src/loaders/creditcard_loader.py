"""
Credit Card Fraud Detection Dataset Loader
Phase 4: Data Collection and Preprocessing

Dataset: Credit Card Fraud Detection
URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
Target: 'Class' (binary: 0=normal, 1=fraud)
Features: 30 features (V1-V28 PCA transformed, Amount, Time)
Samples: 284,807

Citation:
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Feature names
PCA_FEATURES = [f'V{i}' for i in range(1, 29)]  # V1-V28
OTHER_FEATURES = ['Time', 'Amount']
ALL_FEATURES = OTHER_FEATURES + PCA_FEATURES
TARGET_COLUMN = 'Class'

RANDOM_SEED = 42


def download_creditcard(data_dir: Path) -> bool:
    """
    Check if Credit Card Fraud dataset exists

    Note: This dataset requires manual download from Kaggle:
    https://www.kaggle.com/mlg-ulb/creditcardfraud

    File needed:
    - creditcard.csv

    Args:
        data_dir: Directory to store data

    Returns:
        True if file exists, False otherwise
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / 'creditcard.csv'

    if data_file.exists():
        return True

    print("=" * 60)
    print("Credit Card Fraud dataset not found!")
    print("=" * 60)
    print("\nPlease download the dataset manually:")
    print("1. Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("2. Download creditcard.csv (143.8 MB)")
    print(f"3. Place it in: {data_dir}")
    print("\nNote: You may need a Kaggle account to download.")
    print("=" * 60)

    return False


def load_raw_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load raw Credit Card Fraud dataset

    Args:
        data_dir: Directory containing creditcard.csv

    Returns:
        Raw dataframe
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'creditcard'
    else:
        data_dir = Path(data_dir)

    # Check if data exists
    if not download_creditcard(data_dir):
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found in {data_dir}. "
            "Please download from Kaggle."
        )

    data_file = data_dir / 'creditcard.csv'
    df = pd.read_csv(data_file)

    return df


def preprocess_creditcard(df: pd.DataFrame, scale_time_amount: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess Credit Card Fraud dataset

    Args:
        df: Raw dataframe
        scale_time_amount: Whether to scale Time and Amount features

    Returns:
        X: Preprocessed features
        y: Binary target (0=normal, 1=fraud)
    """
    df = df.copy()

    # 1. Extract target
    y = df[TARGET_COLUMN].astype(int)
    df = df.drop(columns=[TARGET_COLUMN])

    # 2. Handle missing values (should be none in this dataset)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    # 3. Scale Time and Amount (V1-V28 are already PCA-transformed and scaled)
    if scale_time_amount:
        scaler = StandardScaler()
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    X = df

    return X, y


def load_creditcard(data_dir: str = None, preprocess: bool = True, sample_frac: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess Credit Card Fraud dataset for DatasetRegistry

    Args:
        data_dir: Directory containing creditcard.csv
        preprocess: Whether to preprocess (scale Time/Amount)
        sample_frac: Fraction of data to sample (for faster testing, default 1.0 = all data)

    Returns:
        X: Feature dataframe (30 features: V1-V28, Time, Amount)
        y: Target series (binary: 0=normal, 1=fraud)

    Example:
        >>> X, y = load_creditcard()
        >>> print(X.shape, y.shape)
        (284807, 30) (284807,)
    """
    # Load raw data
    df = load_raw_data(data_dir)

    # Sample if requested (stratified by Class to maintain fraud ratio)
    if sample_frac < 1.0:
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(
            df,
            test_size=sample_frac,
            random_state=RANDOM_SEED,
            stratify=df[TARGET_COLUMN]
        )
        df = df.reset_index(drop=True)

    if preprocess:
        X, y = preprocess_creditcard(df)
    else:
        y = df[TARGET_COLUMN].astype(int)
        X = df.drop(columns=[TARGET_COLUMN])

    return X, y


def get_dataset_info() -> dict:
    """
    Get metadata about Credit Card Fraud dataset

    Returns:
        Dictionary with dataset information
    """
    return {
        'name': 'CreditCardFraud',
        'task': 'binary_classification',
        'target_column': 'Class',
        'n_features': 30,
        'n_samples': 284807,
        'numerical_features': ALL_FEATURES,
        'categorical_features': [],
        'target_distribution': {
            'normal': 0.9983,  # 99.83%
            'fraud': 0.0017    # 0.17%
        },
        'class_imbalance': 'severe',
        'imbalance_ratio': 577,  # normal:fraud = 577:1
        'download_url': 'https://www.kaggle.com/mlg-ulb/creditcardfraud',
        'citation': 'Dal Pozzolo et al. (2015)',
        'description': 'Credit card transactions with fraud labels. Features V1-V28 are PCA components.'
    }


def analyze_class_imbalance(y: pd.Series) -> dict:
    """
    Analyze class distribution and imbalance

    Args:
        y: Target series

    Returns:
        Dictionary with imbalance statistics
    """
    normal_count = (y == 0).sum()
    fraud_count = (y == 1).sum()
    total = len(y)

    return {
        'normal_count': normal_count,
        'fraud_count': fraud_count,
        'total': total,
        'normal_pct': normal_count / total * 100,
        'fraud_pct': fraud_count / total * 100,
        'imbalance_ratio': normal_count / fraud_count if fraud_count > 0 else np.inf
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Credit Card Fraud Dataset Loader Test")
    print("=" * 60)

    try:
        # Test loading (sample for faster test)
        print("\n[Loading dataset (10% sample for testing)...]")
        X, y = load_creditcard(sample_frac=0.1)

        print(f"\n[Dataset Statistics]")
        print(f"Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns)}")

        print(f"\n[Target Distribution]")
        imbalance = analyze_class_imbalance(y)
        print(f"Normal transactions: {imbalance['normal_count']:,} ({imbalance['normal_pct']:.2f}%)")
        print(f"Fraudulent transactions: {imbalance['fraud_count']:,} ({imbalance['fraud_pct']:.2f}%)")
        print(f"Imbalance ratio: {imbalance['imbalance_ratio']:.1f}:1")

        print(f"\n[Feature Statistics (first 5 features)]")
        print(X.iloc[:, :5].describe())

        print(f"\n[Amount Distribution]")
        print(f"Mean: ${X['Amount'].mean():.2f}")
        print(f"Median: ${X['Amount'].median():.2f}")
        print(f"Max: ${X['Amount'].max():.2f}")

        print("\n✅ Credit Card Fraud loader test completed!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nThis is expected if you haven't downloaded the dataset yet.")
        print("Follow the instructions above to download the data.")
