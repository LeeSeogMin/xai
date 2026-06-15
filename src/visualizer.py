'Visualization utilities for experiment results.'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

try:
    from .config import FIGURES_DIR, RESULTS_DIR
except ImportError:
    from config import FIGURES_DIR, RESULTS_DIR


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


def plot_feature_importance_comparison(
    fi_baseline: pd.Series,
    fi_corrupted: pd.Series,
    title: str = "Feature Importance Comparison",
    save_path: Path = None
):
    'Visualization utilities for experiment results.'
    fig, ax = plt.subplots(figsize=(12, 6))


    fi_baseline_sorted = fi_baseline.sort_values(ascending=False)
    features = fi_baseline_sorted.index

    x = np.arange(len(features))
    width = 0.35

    bars1 = ax.bar(x - width/2, fi_baseline_sorted.values, width, label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, fi_corrupted[features].values, width, label='Corrupted', color='coral')

    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_divergence_heatmap(
    results: dict,
    metric: str = 'js_divergence',
    save_path: Path = None
):
    'Visualization utilities for experiment results.'
    quality_issues = list(results['experiments'].keys())

    severity_levels = sorted(results['experiments'][quality_issues[0]].keys(), key=lambda x: float(x))


    data_shap = np.zeros((len(quality_issues), len(severity_levels)))
    data_lime = np.zeros((len(quality_issues), len(severity_levels)))

    for i, qi in enumerate(quality_issues):
        for j, sev in enumerate(severity_levels):
            rf_shap = results['experiments'][qi][sev]['random_forest']['divergence']['shap'][metric]
            xgb_shap = results['experiments'][qi][sev]['xgboost']['divergence']['shap'][metric]
            data_shap[i, j] = (rf_shap + xgb_shap) / 2

            rf_lime = results['experiments'][qi][sev]['random_forest']['divergence']['lime'][metric]
            xgb_lime = results['experiments'][qi][sev]['xgboost']['divergence']['lime'][metric]
            data_lime[i, j] = (rf_lime + xgb_lime) / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SHAP Heatmap
    sns.heatmap(
        data_shap,
        ax=axes[0],
        xticklabels=[f'{float(s)*100:.0f}%' for s in severity_levels],
        yticklabels=[qi.replace('_', ' ').title() for qi in quality_issues],
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    axes[0].set_title(f'SHAP - {metric.replace("_", " ").title()}')
    axes[0].set_xlabel('Severity Level')
    axes[0].set_ylabel('Quality Issue Type')

    # LIME Heatmap
    sns.heatmap(
        data_lime,
        ax=axes[1],
        xticklabels=[f'{float(s)*100:.0f}%' for s in severity_levels],
        yticklabels=[qi.replace('_', ' ').title() for qi in quality_issues],
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    axes[1].set_title(f'LIME - {metric.replace("_", " ").title()}')
    axes[1].set_xlabel('Severity Level')
    axes[1].set_ylabel('Quality Issue Type')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_degradation(
    results: dict,
    metric: str = 'accuracy',
    save_path: Path = None
):
    'Visualization utilities for experiment results.'
    quality_issues = list(results['experiments'].keys())

    severity_keys = sorted(results['experiments'][quality_issues[0]].keys(), key=lambda x: float(x))
    severity_levels = [float(s) for s in severity_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, model_type in enumerate(['random_forest', 'xgboost']):
        ax = axes[ax_idx]

        for qi in quality_issues:
            degradations = []
            for sev_key in severity_keys:
                deg = results['experiments'][qi][sev_key][model_type]['perf_degradation'][metric]
                degradations.append(deg)

            ax.plot(
                [s*100 for s in severity_levels],
                degradations,
                marker='o',
                label=qi.replace('_', ' ').title()
            )

        ax.set_xlabel('Severity Level (%)')
        ax.set_ylabel(f'{metric.title()} Degradation')
        ax.set_title(f'{model_type.replace("_", " ").title()} - Performance Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_xai_comparison(
    results: dict,
    quality_issue: str = 'missing',
    severity: float = 0.30,
    save_path: Path = None
):
    'Visualization utilities for experiment results.'

    severity_key = str(severity)
    exp = results['experiments'][quality_issue][severity_key]

    metrics = ['js_divergence', 'spearman_correlation', 'rank_change_ratio_top10']
    metric_labels = ['JS Divergence', 'Spearman Corr.', 'Rank Change (Top 10)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]

        x = np.arange(2)
        width = 0.35

        shap_values = [
            exp['random_forest']['divergence']['shap'][metric],
            exp['xgboost']['divergence']['shap'][metric]
        ]
        lime_values = [
            exp['random_forest']['divergence']['lime'][metric],
            exp['xgboost']['divergence']['lime'][metric]
        ]

        bars1 = ax.bar(x - width/2, shap_values, width, label='SHAP', color='steelblue')
        bars2 = ax.bar(x + width/2, lime_values, width, label='LIME', color='coral')

        ax.set_ylabel(label)
        ax.set_title(f'{label}')
        ax.set_xticks(x)
        ax.set_xticklabels(['RandomForest', 'XGBoost'])
        ax.legend()


        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

    fig.suptitle(
        f'XAI Comparison: {quality_issue.replace("_", " ").title()} (Severity: {severity*100:.0f}%)',
        fontsize=14
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_severity_vs_divergence(
    results: dict,
    model_type: str = 'random_forest',
    xai_method: str = 'shap',
    save_path: Path = None
):
    'Visualization utilities for experiment results.'
    quality_issues = list(results['experiments'].keys())

    severity_keys = sorted(results['experiments'][quality_issues[0]].keys(), key=lambda x: float(x))
    severity_levels = [float(s) for s in severity_keys]

    fig, ax = plt.subplots(figsize=(10, 6))

    for qi in quality_issues:
        divergences = []
        for sev_key in severity_keys:
            div = results['experiments'][qi][sev_key][model_type]['divergence'][xai_method]['js_divergence']
            divergences.append(div)

        ax.plot(
            [s*100 for s in severity_levels],
            divergences,
            marker='o',
            markersize=8,
            linewidth=2,
            label=qi.replace('_', ' ').title()
        )

    ax.set_xlabel('Severity Level (%)', fontsize=12)
    ax.set_ylabel('JS Divergence', fontsize=12)
    ax.set_title(
        f'Severity vs FI Divergence ({model_type.replace("_", " ").title()}, {xai_method.upper()})',
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_figures(results: dict, output_dir: Path = FIGURES_DIR):
    'Generate all figures.'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Creating visualizations]")

    # 1. Divergence Heatmap
    print("  1. Divergence Heatmap...", end=' ')
    plot_divergence_heatmap(
        results,
        metric='js_divergence',
        save_path=output_dir / 'divergence_heatmap.png'
    )
    print("Done")

    # 2. Performance Degradation
    print("  2. Performance Degradation...", end=' ')
    plot_performance_degradation(
        results,
        metric='accuracy',
        save_path=output_dir / 'performance_degradation.png'
    )
    print("Done")

    # 3. XAI Comparison
    print("  3. XAI Comparison...", end=' ')
    for qi in results['experiments'].keys():
        plot_xai_comparison(
            results,
            quality_issue=qi,
            severity=0.30,
            save_path=output_dir / f'xai_comparison_{qi}.png'
        )
    print("Done")

    # 4. Severity vs Divergence
    print("  4. Severity vs Divergence...", end=' ')
    for model in ['random_forest', 'xgboost']:
        for xai in ['shap', 'lime']:
            plot_severity_vs_divergence(
                results,
                model_type=model,
                xai_method=xai,
                save_path=output_dir / f'severity_divergence_{model}_{xai}.png'
            )
    print("Done")

    print(f"\nAll plots were saved to {output_dir}.")


if __name__ == '__main__':

    result_files = list(RESULTS_DIR.glob('experiment_results_*.json'))
    if result_files:
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        print(f"Loaded result file: {latest_file}")

        with open(latest_file, 'r') as f:
            results = json.load(f)


        generate_all_figures(results)
    else:
        print("No result file found. Run main.py first.")
