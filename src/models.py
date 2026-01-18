"""
모델 학습 및 평가 모듈
Phase 5: 실험 수행
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from typing import Literal

from config import MODEL_CONFIGS, MODELS_DIR, RANDOM_SEED


def create_model(
    model_type: Literal['random_forest', 'xgboost'] = 'random_forest'
) -> RandomForestClassifier | XGBClassifier:
    """
    모델 생성

    Args:
        model_type: 모델 유형 ('random_forest' 또는 'xgboost')

    Returns:
        생성된 모델 객체
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(**MODEL_CONFIGS['random_forest'])
    elif model_type == 'xgboost':
        return XGBClassifier(**MODEL_CONFIGS['xgboost'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None
) -> dict:
    """
    모델 학습

    Args:
        model: 학습할 모델
        X_train: 훈련 데이터
        y_train: 훈련 레이블
        X_val: 검증 데이터 (옵션)
        y_val: 검증 레이블 (옵션)

    Returns:
        학습된 모델 및 학습 정보
    """
    # XGBoost용 early stopping (검증 데이터가 있는 경우)
    if isinstance(model, XGBClassifier) and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    모델 평가

    Args:
        model: 평가할 모델
        X_test: 테스트 데이터
        y_test: 테스트 레이블

    Returns:
        평가 지표 딕셔너리
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    return metrics


def get_feature_importance(
    model,
    feature_names: list[str]
) -> pd.Series:
    """
    모델의 Feature Importance 추출

    Args:
        model: 학습된 모델
        feature_names: 특성 이름 목록

    Returns:
        Feature Importance Series
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")

    return pd.Series(importance, index=feature_names, name='importance')


def save_model(model, name: str, model_dir: Path = MODELS_DIR):
    """모델 저장"""
    model_path = model_dir / f"{name}.pkl"
    joblib.dump(model, model_path)
    return model_path


def load_model(name: str, model_dir: Path = MODELS_DIR):
    """모델 로드"""
    model_path = model_dir / f"{name}.pkl"
    return joblib.load(model_path)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'phase4_data'))
    from data_loader import prepare_dataset

    print("=" * 60)
    print("모델 학습 및 평가 테스트")
    print("=" * 60)

    # 데이터 준비
    print("\n[데이터 로딩]")
    data = prepare_dataset()
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")

    # RandomForest 학습 및 평가
    print("\n[RandomForest 모델]")
    rf_model = create_model('random_forest')
    rf_model = train_model(rf_model, data['X_train'], data['y_train'])
    rf_metrics = evaluate_model(rf_model, data['X_test'], data['y_test'])
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {rf_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {rf_metrics['roc_auc']:.4f}")

    # XGBoost 학습 및 평가
    print("\n[XGBoost 모델]")
    xgb_model = create_model('xgboost')
    xgb_model = train_model(xgb_model, data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'])
    xgb_metrics = evaluate_model(xgb_model, data['X_test'], data['y_test'])
    print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {xgb_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

    # Feature Importance
    print("\n[Feature Importance - RandomForest]")
    rf_fi = get_feature_importance(rf_model, data['feature_names'])
    print(rf_fi.sort_values(ascending=False).head(5))

    print("\n✅ 모델 테스트 완료!")
