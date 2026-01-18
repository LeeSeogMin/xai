"""
Experiment Result Aggregation and Reporting
Phase 5: Experiments and Analysis

Improvement 2: Statistical Rigor
- Aggregate CV results across experiments
- Generate LaTeX tables for publication
- Export summary reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    dataset_name: str
    model_name: str
    xai_method: str
    quality_type: str  # 'missing', 'outlier', 'distribution_shift'
    severity: float  # 0.1, 0.2, 0.3, 0.4
    fold_id: Optional[int] = None  # For CV


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    fi_divergence_js: float
    fi_divergence_spearman: float
    model_accuracy_baseline: float
    model_accuracy_degraded: float
    detection_precision: Optional[float] = None  # For traditional method comparison
    detection_recall: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


class ExperimentResultAggregator:
    """
    Aggregate and export experiment results

    Features:
    - Store results from multiple experiments
    - Compute summary statistics
    - Generate publication-ready LaTeX tables
    - Export CSV reports
    """

    def __init__(self):
        """Initialize aggregator"""
        self.results: List[ExperimentResult] = []

    def add_result(
        self,
        config: ExperimentConfig,
        metrics: Dict[str, float]
    ):
        """
        Add experiment result

        Args:
            config: Experiment configuration
            metrics: Dictionary of metric values
        """
        result = ExperimentResult(
            config=config,
            fi_divergence_js=metrics.get('fi_divergence_js', np.nan),
            fi_divergence_spearman=metrics.get('fi_divergence_spearman', np.nan),
            model_accuracy_baseline=metrics.get('model_accuracy_baseline', np.nan),
            model_accuracy_degraded=metrics.get('model_accuracy_degraded', np.nan),
            detection_precision=metrics.get('detection_precision'),
            detection_recall=metrics.get('detection_recall'),
            p_value=metrics.get('p_value'),
            effect_size=metrics.get('effect_size')
        )

        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame

        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for result in self.results:
            row = {
                'dataset': result.config.dataset_name,
                'model': result.config.model_name,
                'xai_method': result.config.xai_method,
                'quality_type': result.config.quality_type,
                'severity': result.config.severity,
                'fold_id': result.config.fold_id,
                'fi_div_js': result.fi_divergence_js,
                'fi_div_spearman': result.fi_divergence_spearman,
                'acc_baseline': result.model_accuracy_baseline,
                'acc_degraded': result.model_accuracy_degraded,
                'detection_precision': result.detection_precision,
                'detection_recall': result.detection_recall,
                'p_value': result.p_value,
                'effect_size': result.effect_size
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_summary_table(
        self,
        group_by: List[str] = ['dataset', 'model', 'xai_method', 'quality_type', 'severity'],
        metrics: List[str] = ['fi_div_js', 'fi_div_spearman']
    ) -> pd.DataFrame:
        """
        Generate summary table with mean and 95% CI

        Args:
            group_by: Columns to group by
            metrics: Metrics to summarize

        Returns:
            Summary DataFrame
        """
        df = self.to_dataframe()

        if df.empty:
            return df

        # Group and compute statistics
        summary_rows = []

        for name, group in df.groupby(group_by):
            row = dict(zip(group_by, name if isinstance(name, tuple) else [name]))

            for metric in metrics:
                if metric in group.columns:
                    values = group[metric].dropna()

                    if len(values) > 0:
                        mean = values.mean()
                        std = values.std()
                        n = len(values)

                        # 95% CI using t-distribution
                        from scipy import stats
                        if n > 1:
                            ci = stats.t.interval(
                                0.95,
                                df=n - 1,
                                loc=mean,
                                scale=std / np.sqrt(n)
                            )
                            ci_lower, ci_upper = ci
                        else:
                            ci_lower = ci_upper = mean

                        row[f'{metric}_mean'] = mean
                        row[f'{metric}_std'] = std
                        row[f'{metric}_ci'] = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                        row[f'{metric}_n'] = n

            # Add p-value and effect size if available
            if 'p_value' in group.columns:
                p_values = group['p_value'].dropna()
                if len(p_values) > 0:
                    row['p_value_mean'] = p_values.mean()

            if 'effect_size' in group.columns:
                effect_sizes = group['effect_size'].dropna()
                if len(effect_sizes) > 0:
                    row['effect_size_mean'] = effect_sizes.mean()

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def export_latex_table(
        self,
        output_path: str,
        caption: str = "Experiment Results",
        label: str = "tab:results",
        metrics: List[str] = ['fi_div_js', 'fi_div_spearman']
    ):
        """
        Export results as LaTeX table

        Args:
            output_path: Path to save LaTeX file
            caption: Table caption
            label: LaTeX label
            metrics: Metrics to include
        """
        summary = self.generate_summary_table(metrics=metrics)

        if summary.empty:
            print("No results to export")
            return

        # Format for LaTeX
        latex_rows = []

        for _, row in summary.iterrows():
            latex_row = {
                'Dataset': row.get('dataset', ''),
                'Model': row.get('model', ''),
                'XAI': row.get('xai_method', ''),
                'Quality': row.get('quality_type', ''),
                'Severity': f"{row.get('severity', 0):.1f}"
            }

            for metric in metrics:
                mean_col = f'{metric}_mean'
                ci_col = f'{metric}_ci'

                if mean_col in row and ci_col in row:
                    latex_row[metric.upper()] = f"{row[mean_col]:.4f} {row[ci_col]}"

            # Add significance stars
            if 'p_value_mean' in row:
                p = row['p_value_mean']
                if p < 0.001:
                    latex_row['Sig.'] = '***'
                elif p < 0.01:
                    latex_row['Sig.'] = '**'
                elif p < 0.05:
                    latex_row['Sig.'] = '*'
                else:
                    latex_row['Sig.'] = 'ns'

            latex_rows.append(latex_row)

        latex_df = pd.DataFrame(latex_rows)

        # Generate LaTeX code
        latex_str = latex_df.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=label,
            column_format='l' * len(latex_df.columns)
        )

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(latex_str)

        print(f"✓ LaTeX table exported to: {output_path}")

    def export_csv(self, output_path: str):
        """
        Export all results to CSV

        Args:
            output_path: Path to save CSV file
        """
        df = self.to_dataframe()

        if df.empty:
            print("No results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"✓ Results exported to: {output_path}")

    def export_summary_csv(self, output_path: str):
        """
        Export summary table to CSV

        Args:
            output_path: Path to save CSV file
        """
        summary = self.generate_summary_table()

        if summary.empty:
            print("No results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary.to_csv(output_path, index=False)
        print(f"✓ Summary exported to: {output_path}")

    def get_best_configurations(
        self,
        metric: str = 'fi_div_js',
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Get best experiment configurations by metric

        Args:
            metric: Metric to rank by
            top_k: Number of top configs to return

        Returns:
            DataFrame with top configurations
        """
        summary = self.generate_summary_table(metrics=[metric])

        if summary.empty:
            return summary

        mean_col = f'{metric}_mean'

        if mean_col not in summary.columns:
            print(f"Metric {metric} not found")
            return pd.DataFrame()

        # Sort by mean (descending)
        sorted_summary = summary.sort_values(mean_col, ascending=False)

        return sorted_summary.head(top_k)

    def save_json(self, output_path: str):
        """
        Save results to JSON file

        Args:
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        results_dict = [
            {
                'config': asdict(r.config),
                'metrics': {
                    'fi_divergence_js': r.fi_divergence_js,
                    'fi_divergence_spearman': r.fi_divergence_spearman,
                    'model_accuracy_baseline': r.model_accuracy_baseline,
                    'model_accuracy_degraded': r.model_accuracy_degraded,
                    'detection_precision': r.detection_precision,
                    'detection_recall': r.detection_recall,
                    'p_value': r.p_value,
                    'effect_size': r.effect_size
                }
            }
            for r in self.results
        ]

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"✓ Results saved to: {output_path}")

    def load_json(self, input_path: str):
        """
        Load results from JSON file

        Args:
            input_path: Path to JSON file
        """
        with open(input_path, 'r') as f:
            results_dict = json.load(f)

        for item in results_dict:
            config = ExperimentConfig(**item['config'])
            self.add_result(config, item['metrics'])

        print(f"✓ Loaded {len(results_dict)} results from: {input_path}")


if __name__ == '__main__':
    # Test aggregator
    print("=" * 60)
    print("Experiment Result Aggregator Test")
    print("=" * 60)

    aggregator = ExperimentResultAggregator()

    # Add dummy results
    for dataset in ['uci_adult', 'unsw_nb15']:
        for model in ['RandomForest', 'XGBoost']:
            for xai in ['shap', 'lime']:
                for quality in ['missing', 'outlier']:
                    for severity in [0.1, 0.2, 0.3]:
                        for fold in range(3):  # 3-fold CV
                            config = ExperimentConfig(
                                dataset_name=dataset,
                                model_name=model,
                                xai_method=xai,
                                quality_type=quality,
                                severity=severity,
                                fold_id=fold
                            )

                            # Simulate metrics
                            metrics = {
                                'fi_divergence_js': 0.3 + np.random.randn() * 0.05,
                                'fi_divergence_spearman': 0.7 + np.random.randn() * 0.05,
                                'model_accuracy_baseline': 0.85,
                                'model_accuracy_degraded': 0.80,
                                'p_value': 0.01,
                                'effect_size': 0.5
                            }

                            aggregator.add_result(config, metrics)

    print(f"\n[Added {len(aggregator.results)} results]")

    # Generate summary
    print("\n[Summary Table]")
    summary = aggregator.generate_summary_table()
    print(summary.head(10).to_string(index=False))

    # Export CSV
    aggregator.export_csv('/tmp/test_results.csv')
    aggregator.export_summary_csv('/tmp/test_summary.csv')

    # Export LaTeX
    aggregator.export_latex_table('/tmp/test_results.tex')

    # Get best configs
    print("\n[Top 5 Configurations by FI Divergence JS]")
    best = aggregator.get_best_configurations(metric='fi_div_js', top_k=5)
    print(best.to_string(index=False))

    print("\n✅ Result aggregator test completed!")
