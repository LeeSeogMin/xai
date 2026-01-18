"""
Deep Learning XAI Methods
Phase 5: Experiments and Analysis

Improvement 5: Deep Learning Extension
- Integrated Gradients (model-agnostic gradient-based attribution)
- Attention-based attribution (for AttentionNet)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from captum.attr import IntegratedGradients
import warnings
warnings.filterwarnings('ignore')


class DLXAIAnalyzer:
    """
    XAI methods for deep learning models

    Methods:
    - Integrated Gradients (Captum)
    - Attention-based attribution
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize DL XAI analyzer

        Args:
            model: PyTorch model
            device: Device ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def integrated_gradients(
        self,
        X: np.ndarray,
        target_class: int = 1,
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution

        Args:
            X: Input features (n_samples, n_features)
            target_class: Target class for attribution
            n_steps: Number of steps in integral approximation
            baseline: Baseline input (default: zeros)

        Returns:
            Feature attributions (n_samples, n_features)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad = True

        # Set baseline (default: zeros)
        if baseline is None:
            baseline_tensor = torch.zeros_like(X_tensor).to(self.device)
        else:
            baseline_tensor = torch.FloatTensor(baseline).to(self.device)

        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.model)

        # Compute attributions
        # Note: Captum expects model to output raw logits
        attributions = ig.attribute(
            X_tensor,
            baselines=baseline_tensor,
            target=target_class,
            n_steps=n_steps
        )

        return attributions.detach().cpu().numpy()

    def attention_attribution(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract attention-based attribution (for AttentionNet)

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            (attributions, attention_weights)
            - attributions: (n_samples, n_features)
            - attention_weights: raw attention weights from model
        """
        # Check if model has attention
        if not hasattr(self.model, 'attention'):
            raise ValueError("Model does not have attention mechanism")

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(X_tensor)

            # Get attention weights
            if not hasattr(self.model, 'attention_weights') or self.model.attention_weights is None:
                raise ValueError("No attention weights available. Model may not support attention extraction.")

            attn_weights = self.model.attention_weights

            # Handle different shapes from MultiheadAttention
            # Shape can be: (batch_size, n_features, n_features) when average_attn_weights=True
            # or (batch_size, num_heads, n_features, n_features) when average_attn_weights=False
            if attn_weights.dim() == 3:
                # (batch_size, seq_len, seq_len) - already averaged
                attn_weights_avg = attn_weights
            elif attn_weights.dim() == 4:
                # (batch_size, num_heads, seq_len, seq_len)
                attn_weights_avg = attn_weights.mean(dim=1)  # Average across heads
            else:
                raise ValueError(f"Unexpected attention weights shape: {attn_weights.shape}")

            # Sum incoming attention (how much each feature is attended to)
            attributions = attn_weights_avg.sum(dim=1)  # (batch_size, seq_len)

        return (
            attributions.detach().cpu().numpy(),
            attn_weights.detach().cpu().numpy()
        )

    def gradient_input_attribution(
        self,
        X: np.ndarray,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Simple Gradient × Input attribution

        Args:
            X: Input features (n_samples, n_features)
            target_class: Target class

        Returns:
            Feature attributions (n_samples, n_features)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad = True

        # Forward pass
        outputs = self.model(X_tensor)

        # Get target class output
        target_output = outputs[:, target_class]

        # Backward pass
        self.model.zero_grad()
        target_output.sum().backward()

        # Gradient × Input
        gradients = X_tensor.grad.detach().cpu().numpy()
        attributions = gradients * X

        return attributions

    def get_feature_importance(
        self,
        X: np.ndarray,
        method: str = 'integrated_gradients',
        aggregate: str = 'mean_abs',
        **kwargs
    ) -> np.ndarray:
        """
        Get feature importance scores

        Args:
            X: Input features (n_samples, n_features)
            method: Attribution method
                - 'integrated_gradients'
                - 'attention' (for AttentionNet)
                - 'gradient_input'
            aggregate: Aggregation method
                - 'mean_abs': Mean absolute attribution
                - 'mean': Mean attribution
                - 'std': Standard deviation
            **kwargs: Additional arguments for attribution methods

        Returns:
            Feature importance scores (n_features,)
        """
        # Compute attributions
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients(X, **kwargs)
        elif method == 'attention':
            attributions, _ = self.attention_attribution(X)
        elif method == 'gradient_input':
            attributions = self.gradient_input_attribution(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Aggregate across samples
        if aggregate == 'mean_abs':
            importance = np.abs(attributions).mean(axis=0)
        elif aggregate == 'mean':
            importance = attributions.mean(axis=0)
        elif aggregate == 'std':
            importance = attributions.std(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")

        return importance


def compare_dl_xai_methods(
    model: nn.Module,
    X: np.ndarray,
    feature_names: list,
    methods: list = ['integrated_gradients', 'gradient_input'],
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Compare different DL XAI methods

    Args:
        model: PyTorch model
        X: Input features (n_samples, n_features)
        feature_names: Feature names
        methods: List of XAI methods to compare
        device: Device

    Returns:
        DataFrame with feature importance from each method
    """
    analyzer = DLXAIAnalyzer(model, device)

    results = {'feature': feature_names}

    for method in methods:
        try:
            importance = analyzer.get_feature_importance(X, method=method)
            results[method] = importance
        except Exception as e:
            print(f"Warning: {method} failed - {str(e)}")
            results[method] = np.zeros(len(feature_names))

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Test DL XAI methods
    print("=" * 60)
    print("Deep Learning XAI Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple MLP
    from dl_models import TabularMLP, DLModelWrapper

    print("\n[Training MLP for XAI testing]")
    mlp_wrapper = DLModelWrapper(
        model_class=TabularMLP,
        model_params={'hidden_dims': [64, 32]},
        epochs=20,
        batch_size=32,
        device='cpu'
    )
    mlp_wrapper.fit(X_train, y_train)

    # Test XAI methods
    print("\n[Testing Integrated Gradients]")
    analyzer = DLXAIAnalyzer(mlp_wrapper.model, device='cpu')

    # Integrated Gradients
    ig_attr = analyzer.integrated_gradients(X_test[:10], target_class=1, n_steps=50)
    print(f"IG attributions shape: {ig_attr.shape}")
    print(f"IG attributions range: [{ig_attr.min():.4f}, {ig_attr.max():.4f}]")

    # Gradient × Input
    print("\n[Testing Gradient × Input]")
    gi_attr = analyzer.gradient_input_attribution(X_test[:10], target_class=1)
    print(f"G×I attributions shape: {gi_attr.shape}")
    print(f"G×I attributions range: [{gi_attr.min():.4f}, {gi_attr.max():.4f}]")

    # Feature importance
    print("\n[Testing Feature Importance Extraction]")
    importance_ig = analyzer.get_feature_importance(
        X_test[:100],
        method='integrated_gradients',
        target_class=1
    )
    print(f"Feature importance (IG): {importance_ig[:5]}")
    print(f"Top 5 features: {np.argsort(importance_ig)[-5:][::-1]}")

    # Compare methods
    print("\n[Comparing XAI Methods]")
    feature_names = [f'feature_{i}' for i in range(n_features)]
    comparison = compare_dl_xai_methods(
        mlp_wrapper.model,
        X_test[:100],
        feature_names,
        methods=['integrated_gradients', 'gradient_input']
    )
    print("\nTop 5 features by each method:")
    for method in ['integrated_gradients', 'gradient_input']:
        if method in comparison.columns:
            top_features = comparison.nlargest(5, method)['feature'].values
            print(f"  {method}: {', '.join(top_features)}")

    print("\n✅ Deep learning XAI test completed!")
