'XAI analysis utilities for extracting feature importances with SHAP and LIME.'

import numpy as np
import pandas as pd
from typing import Literal
import warnings

try:
    from .config import XAI_CONFIGS, RANDOM_SEED
except ImportError:
    from config import XAI_CONFIGS, RANDOM_SEED

warnings.filterwarnings('ignore')


def _get_shap():
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SHAP is required for method='shap'. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return shap


def _get_lime_tabular_explainer():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIME is required for method='lime'. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return LimeTabularExplainer


class SHAPAnalyzer:
    'SHAP analysis helper.'

    def __init__(self, model, X_train: pd.DataFrame, model_type: str = 'tree'):
        '  init  .'
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.feature_names = list(X_train.columns)
        shap = _get_shap()


        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:

            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)

    def get_shap_values(
        self,
        X: pd.DataFrame,
        n_samples: int = None
    ) -> np.ndarray:
        'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
        if n_samples is not None and n_samples < len(X):
            np.random.seed(RANDOM_SEED)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[indices]
        else:
            X_sample = X

        shap_values = self.explainer.shap_values(X_sample)


        if isinstance(shap_values, list):

            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:


            if shap_values.shape[0] == len(self.feature_names) and shap_values.shape[1] == 2:
                shap_values = shap_values[:, 1]

        return shap_values

    def get_global_importance(
        self,
        X: pd.DataFrame,
        n_samples: int = None
    ) -> pd.Series:
        'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
        shap_values = self.get_shap_values(X, n_samples)
        global_importance = np.abs(shap_values).mean(axis=0)

        return pd.Series(
            global_importance,
            index=self.feature_names,
            name='shap_importance'
        )


class LIMEAnalyzer:
    'LIME analysis helper.'

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        mode: Literal['classification', 'regression'] = 'classification'
    ):
        'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
        self.model = model
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self.mode = mode
        LimeTabularExplainer = _get_lime_tabular_explainer()


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
        'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
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
        'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
        n_samples = min(n_samples, len(X))
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X), n_samples, replace=False)


        importance_sum = {name: 0.0 for name in self.feature_names}
        importance_count = {name: 0 for name in self.feature_names}

        for idx in indices:
            instance = X.iloc[idx].values
            exp = self.explain_instance(instance, num_features)

            for feature_str, weight in exp['feature_weights'].items():

                for name in self.feature_names:
                    if name in feature_str:
                        importance_sum[name] += abs(weight)
                        importance_count[name] += 1
                        break


        global_importance = {
            name: importance_sum[name] / max(importance_count[name], 1)
            for name in self.feature_names
        }

        return pd.Series(
            global_importance,
            name='lime_importance'
        )



XAIAnalyzer = SHAPAnalyzer

def analyze_xai(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: Literal['shap', 'lime'] = 'shap',
    n_samples: int = 100
) -> pd.Series:
    'XAI analysis utilities for extracting feature importances with SHAP and LIME.'
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
    from data_loader import prepare_dataset
    from models import create_model, train_model

    print("=" * 60)
    print("XAI analysis test")
    print("=" * 60)


    print("\n[Data loading]")
    data = prepare_dataset()
    print(f"X_train: {data['X_train'].shape}")


    print("\n[Model training]")
    model = create_model('random_forest')
    model = train_model(model, data['X_train'], data['y_train'])
    print('RandomForest training complete')


    print("\n[SHAP analysis]")
    shap_importance = analyze_xai(
        model, data['X_train'], data['X_test'],
        method='shap', n_samples=100
    )
    print("Top 5 SHAP Feature Importance:")
    print(shap_importance.sort_values(ascending=False).head())


    print("\n[LIME analysis]")
    lime_importance = analyze_xai(
        model, data['X_train'], data['X_test'],
        method='lime', n_samples=50
    )
    print("Top 5 LIME Feature Importance:")
    print(lime_importance.sort_values(ascending=False).head())

    print("\n✅ XAI analysis test Done!")
