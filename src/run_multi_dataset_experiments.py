"""
다중 데이터셋 실험 스크립트
Phase 5: XAI 기반 데이터 품질 진단 프레임워크

연구 설계에 따른 4개 데이터셋 실험:
1. UCI Adult (48,842 샘플) - 전체 사용
2. UNSW-NB15 (100,000 샘플) - 계층적 샘플링
3. Credit Card Fraud (50,000 + SMOTE) - 불균형 처리
4. Bank Marketing (41,188 샘플) - 전체 사용

실험 설계:
- 3가지 품질 문제 (결측치, 이상치, 분포 편향)
- 4가지 심각도 수준 (5%, 10%, 20%, 30%)
- 2가지 ML 모델 (Random Forest, XGBoost)
- 10-fold 교차 검증
- FI Divergence (SHAP 기반) 측정
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import warnings
from typing import Dict, List, Tuple, Optional

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase4_data'))
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# 설정 임포트
from config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
    QUALITY_ISSUE_TYPES, SEVERITY_LEVELS, MODEL_CONFIGS,
    DATASETS, DATASET_SAMPLING, N_FOLDS, CONFIDENCE_LEVEL
)


def load_dataset_with_sampling(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    데이터셋을 로드하고 샘플링 설정에 따라 처리

    Returns:
        X, y, numerical_features, categorical_features
    """
    from phase4_data.dataset_registry import create_default_registry

    print(f"\n[Loading {dataset_name}...]")
    registry = create_default_registry()

    X, y = registry.get_dataset(dataset_name)
    config = registry.get_config(dataset_name)

    # 샘플링 설정 적용
    sampling_config = DATASET_SAMPLING.get(dataset_name, {})
    max_samples = sampling_config.get('max_samples')

    if max_samples and len(X) > max_samples:
        print(f"  Sampling {max_samples:,} from {len(X):,} samples (stratified)")
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y if sampling_config.get('stratify', True) else None,
            random_state=RANDOM_SEED
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    # SMOTE 적용 (Credit Card 등 극심한 불균형 데이터)
    if sampling_config.get('apply_smote', False):
        print("  Applying SMOTE for class imbalance...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=RANDOM_SEED)
            X_arr, y_arr = smote.fit_resample(X.values, y.values)
            X = pd.DataFrame(X_arr, columns=X.columns)
            y = pd.Series(y_arr)
            print(f"  After SMOTE: {len(X):,} samples")
        except ImportError:
            print("  Warning: imbalanced-learn not installed. Skipping SMOTE.")

    print(f"  Final: {len(X):,} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(y.value_counts())}")

    return X, y, config.numerical_features, config.categorical_features


def inject_quality_issue(
    X: pd.DataFrame,
    issue_type: str,
    severity: float,
    numerical_features: List[str],
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    데이터에 품질 문제 주입
    """
    X_corrupted = X.copy()
    np.random.seed(seed)

    # 수치형 피처만 대상으로
    target_features = [f for f in numerical_features if f in X.columns][:3]

    if not target_features:
        print(f"  Warning: No numerical features found for corruption")
        return X_corrupted

    if issue_type == 'missing':
        # MCAR 결측치 주입
        for col in target_features:
            mask = np.random.random(len(X)) < severity
            X_corrupted.loc[mask, col] = np.nan
        # 중앙값으로 대체
        for col in target_features:
            median_val = X[col].median()
            X_corrupted[col] = X_corrupted[col].fillna(median_val)

    elif issue_type == 'outlier':
        # IQR 기반 이상치 주입
        for col in target_features:
            n_outliers = int(len(X) * severity)
            outlier_idx = np.random.choice(X_corrupted.index, n_outliers, replace=False)
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            # 상위/하위 이상치 랜덤 선택
            outlier_values = np.where(
                np.random.random(n_outliers) > 0.5,
                q3 + 3 * iqr * (1 + np.random.random(n_outliers)),
                q1 - 3 * iqr * (1 + np.random.random(n_outliers))
            )
            X_corrupted.loc[outlier_idx, col] = outlier_values

    elif issue_type == 'distribution_shift':
        # Covariate shift: 평균 이동
        shift_amount = severity * 3  # severity를 표준편차 단위로 변환
        for col in target_features:
            std = X[col].std()
            X_corrupted[col] = X_corrupted[col] + shift_amount * std

    return X_corrupted


def calculate_fi_divergence(fi_baseline: Dict, fi_corrupted: Dict) -> Dict:
    """
    Feature Importance Divergence 계산
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr

    results = {}

    for method in fi_baseline.keys():
        if method not in fi_corrupted:
            continue

        # pd.Series 또는 dict 처리
        base_val = fi_baseline[method]
        corr_val = fi_corrupted[method]

        if hasattr(base_val, 'values'):
            base = np.array(base_val.values)
        elif isinstance(base_val, dict):
            base = np.array(list(base_val.values()))
        else:
            base = np.array(base_val)

        if hasattr(corr_val, 'values'):
            corr = np.array(corr_val.values)
        elif isinstance(corr_val, dict):
            corr = np.array(list(corr_val.values()))
        else:
            corr = np.array(corr_val)

        # 정규화
        base = base / (base.sum() + 1e-10)
        corr = corr / (corr.sum() + 1e-10)

        # Jensen-Shannon Divergence
        js_div = jensenshannon(base, corr) ** 2  # squared for proper metric

        # Spearman 상관계수
        spearman_corr, _ = spearmanr(base, corr)

        # 랭크 변화
        rank_base = np.argsort(np.argsort(-base))
        rank_corr = np.argsort(np.argsort(-corr))
        rank_change = np.mean(np.abs(rank_base - rank_corr))

        results[method] = {
            'js_divergence': float(js_div),
            'spearman_correlation': float(spearman_corr),
            'rank_change': float(rank_change)
        }

    return results


def run_single_fold_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: List[str],
    model_type: str,
    issue_type: str,
    severity: float,
    fold_idx: int
) -> Dict:
    """
    단일 폴드에서 실험 수행
    """
    from models import create_model, train_model, evaluate_model
    from xai_analyzer import analyze_xai

    results = {
        'fold': fold_idx,
        'model': model_type,
        'issue_type': issue_type,
        'severity': severity
    }

    # 1. 베이스라인 모델 학습
    model_baseline = create_model(model_type)
    model_baseline = train_model(model_baseline, X_train, y_train)

    # 2. 베이스라인 평가
    baseline_metrics = evaluate_model(model_baseline, X_test, y_test)
    results['baseline_accuracy'] = baseline_metrics['accuracy']
    results['baseline_f1'] = baseline_metrics['f1']
    results['baseline_auc'] = baseline_metrics['roc_auc']

    # 3. 베이스라인 SHAP 분석
    try:
        fi_baseline = analyze_xai(
            model_baseline, X_train, X_test,
            method='shap', n_samples=min(100, len(X_test))
        )
        fi_baseline = {'shap': fi_baseline}
    except Exception as e:
        print(f"    SHAP baseline failed: {e}")
        fi_baseline = {'shap': {col: 1.0/len(X_train.columns) for col in X_train.columns}}

    # 4. 품질 문제 주입
    X_train_corrupted = inject_quality_issue(
        X_train, issue_type, severity, numerical_features
    )

    # 5. 손상된 데이터로 모델 재학습
    model_corrupted = create_model(model_type)
    model_corrupted = train_model(model_corrupted, X_train_corrupted, y_train)

    # 6. 손상된 모델 평가
    corrupted_metrics = evaluate_model(model_corrupted, X_test, y_test)
    results['corrupted_accuracy'] = corrupted_metrics['accuracy']
    results['corrupted_f1'] = corrupted_metrics['f1']
    results['corrupted_auc'] = corrupted_metrics['roc_auc']

    # 성능 저하
    results['accuracy_drop'] = baseline_metrics['accuracy'] - corrupted_metrics['accuracy']
    results['f1_drop'] = baseline_metrics['f1'] - corrupted_metrics['f1']
    results['auc_drop'] = baseline_metrics['roc_auc'] - corrupted_metrics['roc_auc']

    # 7. 손상된 모델 SHAP 분석
    try:
        fi_corrupted = analyze_xai(
            model_corrupted, X_train_corrupted, X_test,
            method='shap', n_samples=min(100, len(X_test))
        )
        fi_corrupted = {'shap': fi_corrupted}
    except Exception as e:
        print(f"    SHAP corrupted failed: {e}")
        fi_corrupted = {'shap': {col: 1.0/len(X_train.columns) for col in X_train.columns}}

    # 8. FI Divergence 계산
    fi_divergence = calculate_fi_divergence(fi_baseline, fi_corrupted)
    if 'shap' in fi_divergence:
        results['js_divergence'] = fi_divergence['shap']['js_divergence']
        results['spearman_correlation'] = fi_divergence['shap']['spearman_correlation']
        results['rank_change'] = fi_divergence['shap']['rank_change']

    return results


def run_dataset_experiments(dataset_name: str) -> List[Dict]:
    """
    단일 데이터셋에 대한 전체 실험 수행
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    # 데이터 로드
    X, y, numerical_features, categorical_features = load_dataset_with_sampling(dataset_name)

    all_results = []

    # 10-fold 교차검증
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    total_experiments = N_FOLDS * len(['random_forest', 'xgboost']) * len(QUALITY_ISSUE_TYPES) * len(SEVERITY_LEVELS)

    with tqdm(total=total_experiments, desc=f"Experiments ({dataset_name})") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_type in ['random_forest', 'xgboost']:
                for issue_type in QUALITY_ISSUE_TYPES:
                    for severity in SEVERITY_LEVELS:
                        try:
                            result = run_single_fold_experiment(
                                X_train, X_test, y_train, y_test,
                                numerical_features, model_type,
                                issue_type, severity, fold_idx
                            )
                            result['dataset'] = dataset_name
                            all_results.append(result)
                        except Exception as e:
                            print(f"\n  Error in {dataset_name}/{model_type}/{issue_type}/{severity}: {e}")

                        pbar.update(1)

    return all_results


def compute_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    그룹별 통계 계산 (평균, 표준편차, 95% CI)
    """
    from scipy import stats

    grouped = results_df.groupby(['dataset', 'model', 'issue_type', 'severity'])

    stats_list = []
    for name, group in grouped:
        dataset, model, issue_type, severity = name

        n = len(group)

        stat = {
            'dataset': dataset,
            'model': model,
            'issue_type': issue_type,
            'severity': severity,
            'n_folds': n,

            # 성능 지표
            'accuracy_drop_mean': group['accuracy_drop'].mean(),
            'accuracy_drop_std': group['accuracy_drop'].std(),
            'f1_drop_mean': group['f1_drop'].mean(),
            'f1_drop_std': group['f1_drop'].std(),

            # FI Divergence
            'js_divergence_mean': group['js_divergence'].mean(),
            'js_divergence_std': group['js_divergence'].std(),
            'spearman_mean': group['spearman_correlation'].mean(),
            'spearman_std': group['spearman_correlation'].std(),
        }

        # 95% CI 계산
        ci_mult = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1) if n > 1 else 0
        stat['js_divergence_ci'] = ci_mult * stat['js_divergence_std'] / np.sqrt(n) if n > 1 else 0

        stats_list.append(stat)

    return pd.DataFrame(stats_list)


def main():
    """메인 실험 실행"""
    print("=" * 60)
    print("Multi-Dataset XAI Experiments")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print(f"\nDatasets: {DATASETS}")
    print(f"Quality Issues: {QUALITY_ISSUE_TYPES}")
    print(f"Severity Levels: {SEVERITY_LEVELS}")
    print(f"Cross-Validation Folds: {N_FOLDS}")

    all_results = []

    for dataset_name in DATASETS:
        try:
            results = run_dataset_experiments(dataset_name)
            all_results.extend(results)
            print(f"\n✓ {dataset_name}: {len(results)} experiments completed")
        except Exception as e:
            print(f"\n✗ {dataset_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # 결과 저장
    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'multi_dataset_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Raw results saved: {results_file}")

    # 통계 계산 및 저장
    stats_df = compute_statistics(results_df)
    stats_file = RESULTS_DIR / f'multi_dataset_stats_{timestamp}.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved: {stats_file}")

    # JSON 형식으로도 저장
    json_file = RESULTS_DIR / f'multi_dataset_results_{timestamp}.json'
    results_df.to_json(json_file, orient='records', indent=2)
    print(f"✓ JSON results saved: {json_file}")

    # 요약 출력
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for dataset in DATASETS:
        if dataset in results_df['dataset'].values:
            ds_results = results_df[results_df['dataset'] == dataset]
            print(f"\n[{dataset.upper()}]")
            print(f"  Total experiments: {len(ds_results)}")
            print(f"  Mean JS Divergence: {ds_results['js_divergence'].mean():.4f}")
            print(f"  Mean Accuracy Drop: {ds_results['accuracy_drop'].mean():.4f}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results_df, stats_df


if __name__ == '__main__':
    results_df, stats_df = main()
