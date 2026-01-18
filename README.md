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
2. Place raw files in `data/raw/` directory:
   ```
   data/
   ├── raw/
   │   ├── adult.data
   │   ├── adult.test
   │   ├── UNSW_NB15_training-set.csv
   │   ├── creditcard.csv
   │   └── bank-additional-full.csv
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
from src.data_loader import load_dataset
from src.quality_simulator import QualitySimulator
from src.models import train_model
from src.xai_analyzer import XAIAnalyzer
from src.metrics import compute_fi_divergence

# Load dataset
X_train, X_test, y_train, y_test = load_dataset('adult')

# Train baseline model
model = train_model(X_train, y_train, model_type='xgboost')

# Compute baseline feature importance
analyzer = XAIAnalyzer(model, X_train)
baseline_fi = analyzer.get_global_importance()

# Inject quality problem
simulator = QualitySimulator()
X_corrupted = simulator.inject_missing(X_train, severity=0.2, mechanism='mcar')

# Train model on corrupted data
model_corrupted = train_model(X_corrupted, y_train, model_type='xgboost')

# Compute FI divergence
analyzer_corrupted = XAIAnalyzer(model_corrupted, X_corrupted)
corrupted_fi = analyzer_corrupted.get_global_importance()

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
