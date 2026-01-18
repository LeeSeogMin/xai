"""
Phase 5 실험 설정 상수
XAI 기반 데이터 품질 진단 프레임워크
"""

import os
from pathlib import Path

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE4_DIR = PROJECT_ROOT / 'phase4_data'
PHASE5_DIR = PROJECT_ROOT / 'phase5_experiments'
RESULTS_DIR = PHASE5_DIR / 'results'
FIGURES_DIR = PHASE5_DIR / 'figures'
MODELS_DIR = PHASE5_DIR / 'models'

# 디렉토리 생성
for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 재현성 설정
RANDOM_SEED = 42

# 품질 문제 설정
QUALITY_ISSUE_TYPES = ['missing', 'outlier', 'distribution_shift']
SEVERITY_LEVELS = [0.05, 0.10, 0.20, 0.30]

# 모델 설정
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

# XAI 설정
XAI_CONFIGS = {
    'shap': {
        'n_samples': 100,  # SHAP 분석용 샘플 수
        'check_additivity': False
    },
    'lime': {
        'n_samples': 5000,  # LIME 설명용 샘플 수
        'n_features': 10
    }
}

# 평가 지표
MODEL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
XAI_METRICS = ['js_divergence', 'spearman_correlation', 'rank_change_ratio']

# 실험 반복 횟수
N_REPEATS = 3

# ====================================================================
# Improvement 1: 다중 데이터셋 지원
# ====================================================================
DATASETS = ['uci_adult', 'unsw_nb15', 'creditcard', 'bank_marketing']
RUN_ALL_DATASETS = True

# 데이터셋별 샘플링 설정 (연구 설계 기반)
DATASET_SAMPLING = {
    'uci_adult': {
        'max_samples': None,  # 전체 사용 (~48K)
        'stratify': True
    },
    'unsw_nb15': {
        'max_samples': 100_000,  # 계층적 샘플링으로 10만 샘플
        'stratify': True
    },
    'creditcard': {
        'max_samples': 50_000,  # SMOTE 적용 전 샘플링
        'stratify': True,
        'apply_smote': True  # 불균형 처리
    },
    'bank_marketing': {
        'max_samples': None,  # 전체 사용 (~41K)
        'stratify': True
    }
}

# ====================================================================
# Improvement 2: 통계적 검증 프레임워크
# ====================================================================
# Cross-Validation 설정
N_FOLDS = 10
CV_RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95
USE_CROSS_VALIDATION = True

# 통계 검정 설정
SIGNIFICANCE_LEVEL = 0.05  # α for hypothesis testing
MIN_EFFECT_SIZE = 0.2  # Minimum Cohen's d to consider meaningful

# ====================================================================
# Improvement 3: 전통적 방법 비교
# ====================================================================
TRADITIONAL_METHODS = ['zscore', 'iqr', 'ks_test', 'psi']

# Z-score 설정
ZSCORE_THRESHOLD = 3.0

# PSI (Population Stability Index) 설정
PSI_BINS = 10
PSI_THRESHOLD = 0.1  # PSI > 0.1 indicates shift

# ====================================================================
# Improvement 5: 딥러닝 모델 확장
# ====================================================================
# 딥러닝 모델 목록
DL_MODELS = ['MLP', 'AttentionNet']

# 딥러닝 학습 설정
DL_EPOCHS = 50
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001
DL_WEIGHT_DECAY = 1e-5
DL_DROPOUT = 0.3

# GPU 설정 (CUDA 11.8 + RTX 2080 Ti)
import torch
import platform

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# 플랫폼 정보
PLATFORM = platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'
IS_WINDOWS = PLATFORM == 'Windows'
IS_MACOS = PLATFORM == 'Darwin'
IS_LINUX = PLATFORM == 'Linux'

# CUDA 11.8 + RTX 2080 Ti 사양
CUDA_VERSION = torch.version.cuda if USE_GPU else None
GPU_NAME = torch.cuda.get_device_name(0) if USE_GPU and torch.cuda.device_count() > 0 else None
GPU_MEMORY_GB = 11  # RTX 2080 Ti has 11GB VRAM
COMPUTE_CAPABILITY = 7.5  # RTX 2080 Ti compute capability

# GPU 메모리 최적화 설정
if USE_GPU:
    # GPU 정보 자동 탐지
    gpu_props = torch.cuda.get_device_properties(0)
    GPU_MEMORY_GB = gpu_props.total_memory / (1024 ** 3)  # bytes to GB

    # RTX 2080 Ti는 11GB VRAM 보유
    # 배치 크기와 모델 크기 조정 가능
    torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
    torch.backends.cudnn.deterministic = False  # For speed (set True for reproducibility)

    # Mixed precision training 설정 (FP16)
    # RTX 2080 Ti는 Tensor Cores 지원 (Turing architecture)
    USE_MIXED_PRECISION = True

    # GPU 메모리 관리 (실제 VRAM의 85% 사용)
    MAX_MEMORY_ALLOCATED_GB = GPU_MEMORY_GB * 0.85

    # CUDA 메모리 캐시 정리 함수
    def clear_gpu_memory():
        """GPU 메모리 캐시 정리"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
else:
    USE_MIXED_PRECISION = False
    MAX_MEMORY_ALLOCATED_GB = 0

    def clear_gpu_memory():
        """CPU 모드에서는 아무 동작 안함"""
        pass

# 딥러닝 XAI 메서드
DL_XAI_METHODS = ['integrated_gradients', 'attention', 'gradient_input']

# Integrated Gradients 설정
IG_N_STEPS = 50  # Integration steps (trade-off: accuracy vs speed)
IG_BATCH_SIZE = 32  # Batch size for IG computation (adjust based on GPU memory)

# 주의: deeplift와 gradient_shap는 captum에서 지원하지만
# 일부 모델 아키텍처와 호환되지 않을 수 있음
