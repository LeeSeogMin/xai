"""
UNSW-NB15 Network Intrusion Dataset Loader
Phase 4: Data Collection and Preprocessing

Dataset: UNSW-NB15 (Network Security Dataset)
URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Target: 'label' (binary: 0=normal, 1=attack)
Features: 42 features (numerical + categorical)
Samples: ~175,341 (train) + ~82,332 (test)

Citation:
Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network
intrusion detection systems (UNSW-NB15 network data set).
Military Communications and Information Systems Conference (MilCIS), 2015, 1-6.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Feature names (based on UNSW-NB15 documentation)
FEATURE_NAMES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
    'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
    'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'attack_cat', 'label'
]

# Numerical features
NUMERICAL_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# Categorical features
CATEGORICAL_FEATURES = [
    'proto', 'state', 'service', 'is_sm_ips_ports', 'is_ftp_login'
]

# Features to drop (IP addresses, attack category - we only use binary label)
DROP_FEATURES = ['srcip', 'dstip', 'sport', 'dsport', 'attack_cat']

RANDOM_SEED = 42


def download_unsw_nb15(data_dir: Path) -> bool:
    """
    Download UNSW-NB15 dataset if not exists

    Note: This dataset requires manual download from:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset

    Files needed:
    - UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv
    or
    - UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv

    Args:
        data_dir: Directory to store data

    Returns:
        True if files exist, False otherwise
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for pre-split files
    train_file = data_dir / 'UNSW_NB15_training-set.csv'
    test_file = data_dir / 'UNSW_NB15_testing-set.csv'

    if train_file.exists() and test_file.exists():
        return True

    # Check for raw files
    raw_files = [data_dir / f'UNSW-NB15_{i}.csv' for i in range(1, 5)]
    if all(f.exists() for f in raw_files):
        return True

    print("=" * 60)
    print("UNSW-NB15 dataset not found!")
    print("=" * 60)
    print("\nPlease download the dataset manually:")
    print("1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("2. Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
    print(f"3. Place them in: {data_dir}")
    print("\nAlternatively, download UNSW-NB15_1.csv to UNSW-NB15_4.csv")
    print("=" * 60)

    return False


def load_raw_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load raw UNSW-NB15 dataset

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Combined dataframe
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'unsw_nb15'
    else:
        data_dir = Path(data_dir)

    # Check if data exists
    if not download_unsw_nb15(data_dir):
        raise FileNotFoundError(
            f"UNSW-NB15 dataset not found in {data_dir}. "
            "Please download manually."
        )

    # Try pre-split files first
    train_file = data_dir / 'UNSW_NB15_training-set.csv'
    test_file = data_dir / 'UNSW_NB15_testing-set.csv'

    if train_file.exists() and test_file.exists():
        train_df = pd.read_csv(train_file, header=0)
        test_df = pd.read_csv(test_file, header=0)
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    else:
        # Load from raw files
        raw_files = [data_dir / f'UNSW-NB15_{i}.csv' for i in range(1, 5)]
        dfs = []
        for f in raw_files:
            df_part = pd.read_csv(f, header=None, names=FEATURE_NAMES)
            dfs.append(df_part)
        df = pd.concat(dfs, axis=0, ignore_index=True)

    return df


def preprocess_unsw_nb15(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess UNSW-NB15 dataset

    Args:
        df: Raw dataframe

    Returns:
        X: Preprocessed features
        y: Binary target (0=normal, 1=attack)
    """
    df = df.copy()

    # 1. Drop IP addresses and attack category
    df = df.drop(columns=[col for col in DROP_FEATURES if col in df.columns])

    # 2. Extract target
    y = df['label'].astype(int)
    df = df.drop(columns=['label'])

    # 3. Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numerical columns with median
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

    # 4. Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # 5. Standardize numerical features
    scaler = StandardScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])

    X = df

    return X, y


def load_unsw_nb15(data_dir: str = None, preprocess: bool = True, sample_frac: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess UNSW-NB15 dataset for DatasetRegistry

    Args:
        data_dir: Directory containing data files
        preprocess: Whether to preprocess (encode, scale)
        sample_frac: Fraction of data to sample (for faster testing, default 1.0 = all data)

    Returns:
        X: Feature dataframe (37 features after dropping IPs)
        y: Target series (binary: 0=normal, 1=attack)

    Example:
        >>> X, y = load_unsw_nb15()
        >>> print(X.shape, y.shape)
        (175341, 37) (175341,)
    """
    # Load raw data
    df = load_raw_data(data_dir)

    # Sample if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED).reset_index(drop=True)

    if preprocess:
        X, y = preprocess_unsw_nb15(df)
    else:
        y = df['label'].astype(int)
        X = df.drop(columns=['label'])

    return X, y


def get_dataset_info() -> dict:
    """
    Get metadata about UNSW-NB15 dataset

    Returns:
        Dictionary with dataset information
    """
    return {
        'name': 'UNSW-NB15',
        'task': 'binary_classification',
        'target_column': 'label',
        'n_features': 37,  # After dropping IPs and attack_cat
        'n_samples': 175341,  # Approximate (train + test)
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'target_distribution': {
            'normal': 0.44,  # Approximate
            'attack': 0.56
        },
        'download_url': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset',
        'citation': 'Moustafa & Slay (2015)',
        'description': 'Network intrusion detection dataset with normal and attack traffic'
    }


if __name__ == '__main__':
    print("=" * 60)
    print("UNSW-NB15 Dataset Loader Test")
    print("=" * 60)

    try:
        # Test loading (sample for faster test)
        print("\n[Loading dataset (10% sample for testing)...]")
        X, y = load_unsw_nb15(sample_frac=0.1)

        print(f"\n[Dataset Statistics]")
        print(f"Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns[:5])}... (showing first 5)")

        print(f"\n[Target Distribution]")
        print(y.value_counts())
        print(f"Normal: {(y == 0).sum() / len(y) * 100:.1f}%")
        print(f"Attack: {(y == 1).sum() / len(y) * 100:.1f}%")

        print(f"\n[Feature Statistics]")
        print(X.describe())

        print("\n✅ UNSW-NB15 loader test completed!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nThis is expected if you haven't downloaded the dataset yet.")
        print("Follow the instructions above to download the data.")
