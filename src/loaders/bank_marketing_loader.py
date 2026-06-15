"""
Bank Marketing dataset loader.

Dataset: UCI Bank Marketing
URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
Target: y (binary: no/yes)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

TARGET_COLUMN = 'y'
RANDOM_SEED = 42

CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome'
]
NUMERICAL_FEATURES = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]


def check_bank_marketing(data_dir: Path) -> bool:
    """Return True when the Bank Marketing raw file exists."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / 'bank-additional-full.csv'
    if data_file.exists():
        return True

    print('=' * 60)
    print('Bank Marketing dataset not found!')
    print('=' * 60)
    print('Please download bank-additional-full.csv from:')
    print('https://archive.ics.uci.edu/dataset/222/bank+marketing')
    print(f'Place it in: {data_dir}')
    print('=' * 60)
    return False


def load_raw_data(data_dir: str = None) -> pd.DataFrame:
    """Load the raw Bank Marketing dataset."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'bank_marketing'
    else:
        data_dir = Path(data_dir)

    if not check_bank_marketing(data_dir):
        raise FileNotFoundError(
            f'Bank Marketing dataset not found in {data_dir}. Please download it manually.'
        )

    return pd.read_csv(data_dir / 'bank-additional-full.csv', sep=';')


def preprocess_bank_marketing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the Bank Marketing dataset."""
    df = df.copy()
    y = (df[TARGET_COLUMN].astype(str).str.lower() == 'yes').astype(int)
    df = df.drop(columns=[TARGET_COLUMN])
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    present_numeric = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if present_numeric:
        df[present_numeric] = scaler.fit_transform(df[present_numeric])

    return df, y


def load_bank_marketing(data_dir: str = None, preprocess: bool = True, sample_frac: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Bank Marketing dataset."""
    df = load_raw_data(data_dir)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED).reset_index(drop=True)

    if preprocess:
        return preprocess_bank_marketing(df)

    y = (df[TARGET_COLUMN].astype(str).str.lower() == 'yes').astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
