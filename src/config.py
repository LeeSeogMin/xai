'Configuration constants for the XAI-based data quality diagnosis experiments.'

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'
MODELS_DIR = PROJECT_ROOT / 'models'


for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42


QUALITY_ISSUE_TYPES = ['missing', 'outlier', 'distribution_shift']
SEVERITY_LEVELS = [0.05, 0.10, 0.20, 0.30]


MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
}


XAI_CONFIGS = {
    'shap': {
        'n_samples': 100,
        'check_additivity': False
    },
    'lime': {
        'n_samples': 5000,
        'n_features': 10
    }
}


MODEL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
XAI_METRICS = ['js_divergence', 'spearman_correlation', 'rank_change_ratio']


N_REPEATS = 3

# ====================================================================

# ====================================================================
DATASETS = ['uci_adult', 'unsw_nb15', 'creditcard', 'bank_marketing']
RUN_ALL_DATASETS = True


DATASET_SAMPLING = {
    'uci_adult': {
        'max_samples': None,
        'stratify': True
    },
    'unsw_nb15': {
        'max_samples': 100_000,
        'stratify': True
    },
    'creditcard': {
        'max_samples': 50_000,
        'stratify': True,
        'apply_smote': True
    },
    'bank_marketing': {
        'max_samples': None,
        'stratify': True
    }
}

# ====================================================================

# ====================================================================

N_FOLDS = 10
CV_RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95
USE_CROSS_VALIDATION = True


SIGNIFICANCE_LEVEL = 0.05  # α for hypothesis testing
MIN_EFFECT_SIZE = 0.2  # Minimum Cohen's d to consider meaningful

# ====================================================================

# ====================================================================
TRADITIONAL_METHODS = ['zscore', 'iqr', 'ks_test', 'psi']


ZSCORE_THRESHOLD = 3.0


PSI_BINS = 10
PSI_THRESHOLD = 0.1  # PSI > 0.1 indicates shift

# ====================================================================

# ====================================================================

DL_MODELS = ['MLP', 'AttentionNet']


DL_EPOCHS = 50
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001
DL_WEIGHT_DECAY = 1e-5
DL_DROPOUT = 0.3


import torch
import platform

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'


PLATFORM = platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'
IS_WINDOWS = PLATFORM == 'Windows'
IS_MACOS = PLATFORM == 'Darwin'
IS_LINUX = PLATFORM == 'Linux'


CUDA_VERSION = torch.version.cuda if USE_GPU else None
GPU_NAME = torch.cuda.get_device_name(0) if USE_GPU and torch.cuda.device_count() > 0 else None
GPU_MEMORY_GB = 11  # RTX 2080 Ti has 11GB VRAM
COMPUTE_CAPABILITY = 7.5  # RTX 2080 Ti compute capability


if USE_GPU:

    gpu_props = torch.cuda.get_device_properties(0)
    GPU_MEMORY_GB = gpu_props.total_memory / (1024 ** 3)  # bytes to GB



    torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
    torch.backends.cudnn.deterministic = False  # For speed (set True for reproducibility)



    USE_MIXED_PRECISION = True


    MAX_MEMORY_ALLOCATED_GB = GPU_MEMORY_GB * 0.85


    def clear_gpu_memory():
        'Clear gpu memory.'
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
else:
    USE_MIXED_PRECISION = False
    MAX_MEMORY_ALLOCATED_GB = 0

    def clear_gpu_memory():
        'Clear gpu memory.'
        pass


DL_XAI_METHODS = ['integrated_gradients', 'attention', 'gradient_input']


IG_N_STEPS = 50  # Integration steps (trade-off: accuracy vs speed)
IG_BATCH_SIZE = 32  # Batch size for IG computation (adjust based on GPU memory)
