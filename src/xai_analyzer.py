"""
XAI 분석 모듈
Phase 5: SHAP 및 LIME 분석

XAI 기법을 사용하여 Feature Importance를 추출하고
데이터 품질 문제의 영향을 분석합니다.
"""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from typing import Literal
import warnings

from config import XAI_CONFIGS, RANDOM_SEED

warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """SHAP 분석 클래스"""

    def __init__(self, model, X_train: pd.DataFrame, model_type: str = 'tree'):
        """
        Args:
            model: 학습된 모델
            X_train: 훈련 데이터 (SHAP background용)
            model_type: 'tree' 또는 'kernel'
        """
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.feature_names = list(X_train.columns)

        # Explainer 생성
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer용 background 샘플
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)

    def get_shap_values(
        self,
        X: pd.DataFrame,
        n_samples: int = None
    ) -> np.ndarray:
        """
        SHAP 값 계산

        Args:
            X: 분석할 데이터
            n_samples: 분석할 샘플 수 (None이면 전체)

        Returns:
            SHAP 값 배열
        """
        if n_samples is not None and n_samples < len(X):
            np.random.seed(RANDOM_SEED)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[indices]
        else:
            X_sample = X

        shap_values = self.explainer.shap_values(X_sample)

        # 이진 분류의 경우 positive class의 SHAP 값 반환
        if isinstance(shap_values, list):
            # RandomForest의 경우 [class_0, class_1] 형태
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            # XGBoost 등: shape가 (n_samples, n_features)면 그대로 사용
            # 또는 (n_features, n_classes)면 positive class 선택
            if shap_values.shape[0] == len(self.feature_names) and shap_values.shape[1] == 2:
                shap_values = shap_values[:, 1]

        return shap_values

    def get_global_importance(
        self,
        X: pd.DataFrame,
        n_samples: int = None
    ) -> pd.Series:
        """
        Global Feature Importance (평균 |SHAP|)

        Args:
            X: 분석할 데이터
            n_samples: 분석할 샘플 수

        Returns:
            Feature Importance Series
        """
        shap_values = self.get_shap_values(X, n_samples)
        global_importance = np.abs(shap_values).mean(axis=0)

        return pd.Series(
            global_importance,
            index=self.feature_names,
            name='shap_importance'
        )


class LIMEAnalyzer:
    """LIME 분석 클래스"""

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        mode: Literal['classification', 'regression'] = 'classification'
    ):
        """
        Args:
            model: 학습된 모델
            X_train: 훈련 데이터
            mode: 'classification' 또는 'regression'
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self.mode = mode

        # Explainer 생성
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            mode=mode,
            random_state=RANDOM_SEED
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10
    ) -> dict:
        """
        단일 인스턴스 설명

        Args:
            instance: 설명할 인스턴스 (1D array)
            num_features: 표시할 특성 수

        Returns:
            설명 딕셔너리
        """
        if self.mode == 'classification':
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
        else:
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )

        return {
            'feature_weights': dict(exp.as_list()),
            'intercept': exp.intercept[1] if self.mode == 'classification' else exp.intercept,
            'score': exp.score
        }

    def get_global_importance(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
        num_features: int = 10
    ) -> pd.Series:
        """
        Global Feature Importance (평균 |LIME weight|)

        Args:
            X: 분석할 데이터
            n_samples: 분석할 샘플 수
            num_features: 각 설명에서 사용할 특성 수

        Returns:
            Feature Importance Series
        """
        n_samples = min(n_samples, len(X))
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X), n_samples, replace=False)

        # 특성별 중요도 누적
        importance_sum = {name: 0.0 for name in self.feature_names}
        importance_count = {name: 0 for name in self.feature_names}

        for idx in indices:
            instance = X.iloc[idx].values
            exp = self.explain_instance(instance, num_features)

            for feature_str, weight in exp['feature_weights'].items():
                # LIME 특성 문자열에서 특성명 추출
                for name in self.feature_names:
                    if name in feature_str:
                        importance_sum[name] += abs(weight)
                        importance_count[name] += 1
                        break

        # 평균 계산
        global_importance = {
            name: importance_sum[name] / max(importance_count[name], 1)
            for name in self.feature_names
        }

        return pd.Series(
            global_importance,
            name='lime_importance'
        )


def analyze_xai(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: Literal['shap', 'lime'] = 'shap',
    n_samples: int = 100
) -> pd.Series:
    """
    XAI 분석 통합 함수

    Args:
        model: 학습된 모델
        X_train: 훈련 데이터
        X_test: 테스트 데이터
        method: 'shap' 또는 'lime'
        n_samples: 분석할 샘플 수

    Returns:
        Feature Importance Series
    """
    if method == 'shap':
        analyzer = SHAPAnalyzer(model, X_train, model_type='tree')
        return analyzer.get_global_importance(X_test, n_samples)
    elif method == 'lime':
        analyzer = LIMEAnalyzer(model, X_train)
        return analyzer.get_global_importance(X_test, n_samples)
    else:
        raise ValueError(f"Unknown XAI method: {method}")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'phase4_data'))
    from data_loader import prepare_dataset
    from models import create_model, train_model

    print("=" * 60)
    print("XAI 분석 테스트")
    print("=" * 60)

    # 데이터 준비
    print("\n[데이터 로딩]")
    data = prepare_dataset()
    print(f"X_train: {data['X_train'].shape}")

    # 모델 학습
    print("\n[모델 학습]")
    model = create_model('random_forest')
    model = train_model(model, data['X_train'], data['y_train'])
    print("RandomForest 모델 학습 완료")

    # SHAP 분석
    print("\n[SHAP 분석]")
    shap_importance = analyze_xai(
        model, data['X_train'], data['X_test'],
        method='shap', n_samples=100
    )
    print("Top 5 SHAP Feature Importance:")
    print(shap_importance.sort_values(ascending=False).head())

    # LIME 분석
    print("\n[LIME 분석]")
    lime_importance = analyze_xai(
        model, data['X_train'], data['X_test'],
        method='lime', n_samples=50
    )
    print("Top 5 LIME Feature Importance:")
    print(lime_importance.sort_values(ascending=False).head())

    print("\n✅ XAI 분석 테스트 완료!")
