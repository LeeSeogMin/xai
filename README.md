# XAI-based Data Quality Diagnosis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for diagnosing data quality problems in machine learning pipelines using Explainable AI (XAI) techniques.

## Overview

This repository contains the implementation of the **XAI-based Data Quality Diagnosis Framework** that uses SHAP and Integrated Gradients to detect and characterize data quality issues through Feature Importance (FI) Divergence analysis.

### Key Features

- **Multi-Dataset Support**: UCI Adult, UNSW-NB15, Credit Card Fraud, Bank Marketing
- **Multiple XAI Methods**: SHAP (TreeExplainer), Integrated Gradients
- **Quality Problem Detection**: Missing values (MCAR/MAR), Outliers, Distribution Shift
- **Statistical Validation**: 10-fold CV, 95% CI, paired t-tests, Cohen's d
- **Tree-based & Deep Learning Models**: RandomForest, XGBoost, MLP

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/LeeSeogMin/xai.git
cd xai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Access

Due to file size limitations, datasets are not included in this repository. Please download them from the following sources:

| Dataset | Source | Description |
|---------|--------|-------------|
| **UCI Adult** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) | Census income prediction (39,189 samples, 14 features) |
| **UNSW-NB15** | [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) | Network intrusion detection (use 100,000 stratified sample) |
| **Credit Card Fraud** | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Fraud detection (use 50,000 stratified sample) |
| **Bank Marketing** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) | Telemarketing success prediction (41,188 samples) |

### Data Preparation

1. Download datasets from the links above
2. Place raw files under `data/raw/`:
   ```
   data/
   ├── raw/
   │   ├── adult.data
   │   ├── adult.test
   │   ├── unsw_nb15/
   │   │   ├── UNSW_NB15_training-set.csv
   │   │   └── UNSW_NB15_testing-set.csv
   │   ├── creditcard/
   │   │   └── creditcard.csv
   │   └── bank_marketing/
   │       └── bank-additional-full.csv
   └── processed/
   ```
3. Run preprocessing:
   ```bash
   python src/data_loader.py
   ```

## Project Structure

```
xai/
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── quality_simulator.py    # Quality problem injection (MCAR, MAR, outliers, shift)
│   ├── models.py               # ML model training (RF, XGBoost)
│   ├── dl_models.py            # Deep learning models (MLP)
│   ├── xai_analyzer.py         # XAI analysis (SHAP)
│   ├── dl_xai.py               # Deep learning XAI (Integrated Gradients)
│   ├── metrics.py              # FI Divergence metrics (JS, Spearman, Rank Change)
│   ├── statistical_validation.py  # Cross-validation, CI, t-tests
│   ├── traditional_baselines.py   # Z-score, IQR, KS, PSI baselines
│   ├── visualizer.py           # Result visualization
│   ├── dataset_registry.py      # Dataset registry
│   ├── loaders/                 # Dataset-specific loaders
│   └── config.py               # Configuration settings
├── data/
│   ├── raw/                    # Raw dataset files (not included)
│   └── processed/              # Preprocessed data (not included)
├── results/                    # Experiment results
├── notebooks/                  # Jupyter notebooks for analysis
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

### Quick Start

```python
from src.data_loader import prepare_dataset, NUMERICAL_COLS
from src.quality_simulator import inject_missing_values
from src.models import create_model, train_model
from src.xai_analyzer import analyze_xai
from src.metrics import compute_fi_divergence

# Load and preprocess UCI Adult
data = prepare_dataset(scale=True)
X_train, y_train = data['X_train'], data['y_train']
X_test = data['X_test']

# Train baseline model
model = create_model('xgboost')
model = train_model(model, X_train, y_train)

# Compute baseline feature importance
baseline_fi = analyze_xai(model, X_train, X_test, method='shap', n_samples=100)

# Inject MCAR missingness into selected numerical features and impute medians
X_corrupted = inject_missing_values(X_train, NUMERICAL_COLS[:3], severity=0.2)
for col in NUMERICAL_COLS[:3]:
    X_corrupted[col] = X_corrupted[col].fillna(X_train[col].median())

# Train model on corrupted data and compute FI divergence
model_corrupted = create_model('xgboost')
model_corrupted = train_model(model_corrupted, X_corrupted, y_train)
corrupted_fi = analyze_xai(model_corrupted, X_corrupted, X_test, method='shap', n_samples=100)

divergence = compute_fi_divergence(baseline_fi, corrupted_fi)
print(f"JS Divergence: {divergence['js']:.4f}")
```

### Running Full Experiments

```bash
# Multi-dataset experiments (960 conditions)
python src/run_multi_dataset_experiments.py

# MAR vs MCAR comparison
python src/run_extended_experiments.py --experiment mar

# Deep learning experiments
python src/run_extended_experiments.py --experiment deep_learning
```

### Configuration

Edit `src/config.py` to customize:

```python
CONFIG = {
    'datasets': ['adult', 'unsw', 'creditcard', 'bank'],
    'quality_types': ['missing', 'outlier', 'shift'],
    'severities': [0.05, 0.10, 0.20, 0.30],
    'models': ['rf', 'xgboost'],
    'cv_folds': 10,
    'random_seed': 42
}
```

## FI Divergence Thresholds

Based on experiments across 1,080+ conditions:

| FI Divergence (JS) | Interpretation | Recommended Action |
|-------------------|----------------|-------------------|
| < 0.005 | Normal variation | Continue monitoring |
| 0.005 - 0.01 | Minor concern | Investigate affected features |
| 0.01 - 0.02 | Moderate issue | Root cause analysis required |
| > 0.02 | Severe problem | Halt pipeline, remediate data |

## Key Results

- **H1**: XAI detects quality problems across all 1,080 conditions (FI Divergence > 0)
- **H2**: XGBoost shows 54% higher sensitivity than RandomForest
- **H3**: Monotonic severity-divergence relationship (r = 0.94)
- **H4**: Framework generalizes across 4 diverse domains
- **H5**: MAR produces 20.7% higher FI Divergence than MCAR
- **H6**: Deep learning outlier detection shows highest sensitivity (JS = 0.0602)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lee2026xai,
  title={XAI-based Data Quality Diagnosis Framework for Machine Learning Pipelines},
  author={Lee, Seog-Min},
  journal={Journal of Information Processing Systems},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SHAP library by Scott Lundberg
- Captum library for PyTorch interpretability
- UCI Machine Learning Repository for datasets
