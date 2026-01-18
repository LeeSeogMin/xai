"""
메인 실험 스크립트
Phase 5: XAI 기반 데이터 품질 진단 실험

실험 흐름:
1. 클린 데이터로 베이스라인 모델 학습
2. 베이스라인 XAI 분석 (SHAP, LIME)
3. 품질 문제 주입 (결측치, 이상치, 분포 편향)
4. 품질 문제 데이터로 모델 재학습
5. XAI 분석 및 FI Divergence 계산
6. 결과 저장 및 시각화
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import warnings

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'phase4_data'))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    prepare_dataset, NUMERICAL_COLS, CATEGORICAL_COLS,
    encode_features, scale_features
)
from quality_simulator import (
    inject_missing_values, inject_outliers,
    inject_distribution_shift, SEVERITY_LEVELS
)
from config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED,
    QUALITY_ISSUE_TYPES, MODEL_CONFIGS
)
from models import create_model, train_model, evaluate_model, get_feature_importance
from xai_analyzer import analyze_xai
from metrics import calculate_fi_divergence_metrics

warnings.filterwarnings('ignore')

# 실험 설정
EXPERIMENT_CONFIG = {
    'models': ['random_forest', 'xgboost'],
    'xai_methods': ['shap', 'lime'],
    'quality_issues': ['missing', 'outlier', 'distribution_shift'],
    'severity_levels': SEVERITY_LEVELS,
    'n_xai_samples': 100,  # XAI 분석용 샘플 수
    'random_seed': RANDOM_SEED
}


def run_baseline_experiment(data: dict) -> dict:
    """
    베이스라인 실험 수행

    Args:
        data: prepare_dataset()의 반환값

    Returns:
        베이스라인 결과 딕셔너리
    """
    print("\n" + "=" * 60)
    print("Phase 1: 베이스라인 실험")
    print("=" * 60)

    baseline_results = {}

    for model_type in EXPERIMENT_CONFIG['models']:
        print(f"\n[{model_type.upper()}]")

        # 모델 학습
        model = create_model(model_type)
        model = train_model(
            model, data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )

        # 모델 평가
        metrics = evaluate_model(model, data['X_test'], data['y_test'])
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        # 모델 자체 Feature Importance
        model_fi = get_feature_importance(model, data['feature_names'])

        # XAI 분석
        xai_results = {}
        for xai_method in EXPERIMENT_CONFIG['xai_methods']:
            print(f"  XAI ({xai_method})...", end=' ')
            fi = analyze_xai(
                model, data['X_train'], data['X_test'],
                method=xai_method,
                n_samples=EXPERIMENT_CONFIG['n_xai_samples']
            )
            xai_results[xai_method] = fi
            print("완료")

        baseline_results[model_type] = {
            'model': model,
            'metrics': metrics,
            'model_fi': model_fi,
            'xai_fi': xai_results
        }

    return baseline_results


def run_quality_experiment(
    data: dict,
    baseline_results: dict,
    quality_issue: str,
    severity: float
) -> dict:
    """
    품질 문제 주입 후 실험 수행

    Args:
        data: prepare_dataset()의 반환값
        baseline_results: 베이스라인 결과
        quality_issue: 품질 문제 유형
        severity: 심각도

    Returns:
        실험 결과 딕셔너리
    """
    results = {}

    # 원본 훈련 데이터 가져오기
    train_raw = data['train_raw'].copy()
    encoders = data['encoders']
    scaler = data['scaler']

    # 품질 문제 주입
    if quality_issue == 'missing':
        # 수치형 컬럼에 결측치 주입
        corrupted = inject_missing_values(
            train_raw, NUMERICAL_COLS[:3], severity, RANDOM_SEED
        )
        # 결측치 처리: 중앙값으로 대체
        for col in NUMERICAL_COLS[:3]:
            median_val = train_raw[col].median()
            corrupted[col] = corrupted[col].fillna(median_val)
        target_cols = NUMERICAL_COLS[:3]

    elif quality_issue == 'outlier':
        corrupted = inject_outliers(
            train_raw, NUMERICAL_COLS[:3], severity,
            method='iqr', random_seed=RANDOM_SEED
        )
        target_cols = NUMERICAL_COLS[:3]

    elif quality_issue == 'distribution_shift':
        corrupted = inject_distribution_shift(
            train_raw, NUMERICAL_COLS[:3], severity,
            shift_type='covariate_shift', random_seed=RANDOM_SEED
        )
        target_cols = NUMERICAL_COLS[:3]

    else:
        raise ValueError(f"Unknown quality issue: {quality_issue}")

    # 전처리 (인코딩 + 스케일링)
    corrupted_encoded, _ = encode_features(corrupted, encoders=encoders, fit=False)
    corrupted_scaled, _ = scale_features(corrupted_encoded, scaler=scaler, fit=False)

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X_corrupted = corrupted_scaled[feature_cols]
    y_corrupted = corrupted_scaled['income']

    # 각 모델에 대해 실험
    for model_type in EXPERIMENT_CONFIG['models']:
        baseline = baseline_results[model_type]

        # 새 모델 학습 (품질 문제 데이터로)
        model = create_model(model_type)
        model = train_model(model, X_corrupted, y_corrupted)

        # 모델 평가 (테스트 데이터는 클린)
        metrics = evaluate_model(model, data['X_test'], data['y_test'])

        # 성능 저하 계산
        perf_degradation = {
            metric: baseline['metrics'][metric] - metrics[metric]
            for metric in metrics
        }

        # 모델 FI
        model_fi = get_feature_importance(model, data['feature_names'])

        # XAI 분석 및 FI Divergence 계산
        xai_results = {}
        divergence_results = {}

        for xai_method in EXPERIMENT_CONFIG['xai_methods']:
            fi = analyze_xai(
                model, X_corrupted, data['X_test'],
                method=xai_method,
                n_samples=EXPERIMENT_CONFIG['n_xai_samples']
            )
            xai_results[xai_method] = fi

            # FI Divergence 계산
            baseline_fi = baseline['xai_fi'][xai_method]
            divergence = calculate_fi_divergence_metrics(baseline_fi, fi)
            divergence_results[xai_method] = divergence

        results[model_type] = {
            'metrics': metrics,
            'perf_degradation': perf_degradation,
            'model_fi': model_fi,
            'xai_fi': xai_results,
            'divergence': divergence_results,
            'target_columns': target_cols
        }

    return results


def run_full_experiment() -> dict:
    """
    전체 실험 실행

    Returns:
        모든 실험 결과
    """
    print("\n" + "=" * 60)
    print("XAI 기반 데이터 품질 진단 실험")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 데이터 준비
    print("\n[데이터 로딩]")
    data = prepare_dataset()
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_val: {data['X_val'].shape}")
    print(f"  X_test: {data['X_test'].shape}")

    # 결과 저장 구조
    all_results = {
        'config': EXPERIMENT_CONFIG,
        'baseline': None,
        'experiments': {}
    }

    # 1. 베이스라인 실험
    baseline_results = run_baseline_experiment(data)
    all_results['baseline'] = {
        model_type: {
            'metrics': result['metrics'],
            'model_fi': result['model_fi'].to_dict(),
            'xai_fi': {
                method: fi.to_dict()
                for method, fi in result['xai_fi'].items()
            }
        }
        for model_type, result in baseline_results.items()
    }

    # 2. 품질 문제별 실험
    print("\n" + "=" * 60)
    print("Phase 2: 품질 문제 실험")
    print("=" * 60)

    total_experiments = (
        len(EXPERIMENT_CONFIG['quality_issues']) *
        len(EXPERIMENT_CONFIG['severity_levels'])
    )

    with tqdm(total=total_experiments, desc="실험 진행") as pbar:
        for quality_issue in EXPERIMENT_CONFIG['quality_issues']:
            all_results['experiments'][quality_issue] = {}

            for severity in EXPERIMENT_CONFIG['severity_levels']:
                pbar.set_description(f"{quality_issue} ({severity*100:.0f}%)")

                exp_results = run_quality_experiment(
                    data, baseline_results, quality_issue, severity
                )

                # 결과 저장 (직렬화 가능한 형태로)
                all_results['experiments'][quality_issue][severity] = {
                    model_type: {
                        'metrics': result['metrics'],
                        'perf_degradation': result['perf_degradation'],
                        'divergence': result['divergence'],
                        'target_columns': result['target_columns']
                    }
                    for model_type, result in exp_results.items()
                }

                pbar.update(1)

    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)

    return all_results


def save_results(results: dict, filename: str = None):
    """결과 저장"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"experiment_results_{timestamp}.json"

    filepath = RESULTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {filepath}")
    return filepath


def print_summary(results: dict):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)

    # 베이스라인 성능
    print("\n[베이스라인 성능]")
    for model_type, result in results['baseline'].items():
        metrics = result['metrics']
        print(f"  {model_type}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")

    # 품질 문제별 영향
    print("\n[품질 문제별 성능 저하 (최대)]")
    for quality_issue in results['experiments']:
        print(f"\n  {quality_issue.upper()}:")
        for model_type in ['random_forest', 'xgboost']:
            max_degradation = 0
            max_severity = 0
            for severity, exp in results['experiments'][quality_issue].items():
                degradation = exp[model_type]['perf_degradation']['accuracy']
                if degradation > max_degradation:
                    max_degradation = degradation
                    max_severity = severity
            print(f"    {model_type}: -{max_degradation:.4f} (at {float(max_severity)*100:.0f}%)")

    # XAI 기법별 FI Divergence
    print("\n[XAI 기법별 FI Divergence (심각도 30%)]")
    for quality_issue in results['experiments']:
        print(f"\n  {quality_issue.upper()}:")
        exp = results['experiments'][quality_issue][0.30]
        for model_type in ['random_forest', 'xgboost']:
            print(f"    {model_type}:")
            for xai_method in ['shap', 'lime']:
                div = exp[model_type]['divergence'][xai_method]
                print(f"      {xai_method}: JS={div['js_divergence']:.4f}, "
                      f"Spearman={div['spearman_correlation']:.4f}")


if __name__ == '__main__':
    # 전체 실험 실행
    results = run_full_experiment()

    # 결과 저장
    save_results(results)

    # 요약 출력
    print_summary(results)
