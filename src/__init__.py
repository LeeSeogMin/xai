"""
XAI-based Data Quality Diagnosis Framework

A framework for diagnosing data quality problems in ML pipelines using
Explainable AI (XAI) techniques including SHAP and Integrated Gradients.
"""

__version__ = "1.0.0"
__author__ = "Seog-Min Lee"

from .config import *
from .models import create_model, train_model, evaluate_model
from .xai_analyzer import analyze_xai, SHAPAnalyzer
from .metrics import calculate_fi_divergence_metrics
from .data_loader import load_dataset
from .quality_simulator import QualitySimulator
