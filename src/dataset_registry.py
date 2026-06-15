"""Dataset registry for the public XAI reproducibility package."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for one dataset."""
    name: str
    loader_func: Callable
    target_column: str
    categorical_features: List[str]
    numerical_features: List[str]
    task_type: str
    download_url: str
    citation: str
    description: str = ''
    n_samples: Optional[int] = None
    n_features: Optional[int] = None


class DatasetRegistry:
    """Central registry for datasets used in the experiments."""

    def __init__(self):
        self.datasets: Dict[str, DatasetConfig] = {}
        self._cache: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}

    def register(self, config: DatasetConfig) -> None:
        """Register a dataset configuration."""
        self.datasets[config.name] = config

    def get_dataset(self, name: str, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Load a dataset by registry name."""
        if name not in self.datasets:
            available = ', '.join(sorted(self.datasets))
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
        if use_cache and name in self._cache:
            return self._cache[name]

        config = self.datasets[name]
        X, y = config.loader_func()
        if len(X) != len(y):
            raise ValueError(f"Shape mismatch for '{name}': X={len(X)}, y={len(y)}")
        if use_cache:
            self._cache[name] = (X, y)
        return X, y

    def get_config(self, name: str) -> DatasetConfig:
        """Return metadata for a registered dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        return self.datasets[name]

    def list_available(self) -> List[str]:
        """Return registered dataset names."""
        return sorted(self.datasets)


def create_default_registry() -> DatasetRegistry:
    """Create the registry used by the public experiment scripts."""
    try:
        from .data_loader import load_adult_dataset, NUMERICAL_COLS, CATEGORICAL_COLS
        from .loaders.unsw_nb15_loader import (
            load_unsw_nb15,
            NUMERICAL_FEATURES as UNSW_NUMERICAL,
            CATEGORICAL_FEATURES as UNSW_CATEGORICAL,
        )
        from .loaders.creditcard_loader import load_creditcard, ALL_FEATURES as CC_FEATURES
        from .loaders.bank_marketing_loader import (
            load_bank_marketing,
            NUMERICAL_FEATURES as BANK_NUMERICAL,
            CATEGORICAL_FEATURES as BANK_CATEGORICAL,
        )
    except ImportError:
        from data_loader import load_adult_dataset, NUMERICAL_COLS, CATEGORICAL_COLS
        from loaders.unsw_nb15_loader import (
            load_unsw_nb15,
            NUMERICAL_FEATURES as UNSW_NUMERICAL,
            CATEGORICAL_FEATURES as UNSW_CATEGORICAL,
        )
        from loaders.creditcard_loader import load_creditcard, ALL_FEATURES as CC_FEATURES
        from loaders.bank_marketing_loader import (
            load_bank_marketing,
            NUMERICAL_FEATURES as BANK_NUMERICAL,
            CATEGORICAL_FEATURES as BANK_CATEGORICAL,
        )

    registry = DatasetRegistry()
    registry.register(DatasetConfig(
        name='uci_adult', loader_func=load_adult_dataset, target_column='income',
        categorical_features=CATEGORICAL_COLS, numerical_features=NUMERICAL_COLS,
        task_type='binary', download_url='https://archive.ics.uci.edu/ml/datasets/adult',
        citation='Dua and Graff (2019). UCI Machine Learning Repository.',
        description='Census income prediction', n_samples=48842, n_features=14,
    ))
    registry.register(DatasetConfig(
        name='unsw_nb15', loader_func=load_unsw_nb15, target_column='label',
        categorical_features=UNSW_CATEGORICAL, numerical_features=UNSW_NUMERICAL,
        task_type='binary', download_url='https://research.unsw.edu.au/projects/unsw-nb15-dataset',
        citation='Moustafa and Slay (2015). UNSW-NB15 dataset.',
        description='Network intrusion detection', n_samples=175341, n_features=37,
    ))
    registry.register(DatasetConfig(
        name='creditcard', loader_func=load_creditcard, target_column='Class',
        categorical_features=[], numerical_features=CC_FEATURES,
        task_type='binary', download_url='https://www.kaggle.com/mlg-ulb/creditcardfraud',
        citation='Dal Pozzolo et al. (2015). Credit Card Fraud Detection.',
        description='Credit card fraud detection', n_samples=284807, n_features=30,
    ))
    registry.register(DatasetConfig(
        name='bank_marketing', loader_func=load_bank_marketing, target_column='y',
        categorical_features=BANK_CATEGORICAL, numerical_features=BANK_NUMERICAL,
        task_type='binary', download_url='https://archive.ics.uci.edu/dataset/222/bank+marketing',
        citation='Moro et al. (2014). Bank Marketing dataset.',
        description='Bank telemarketing success prediction', n_samples=41188, n_features=20,
    ))
    return registry
