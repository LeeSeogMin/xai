"""
Deep Learning Models for Tabular Data
Phase 5: Experiments and Analysis

Improvement 5: Deep Learning Extension
- MLP (Multi-Layer Perceptron)
- AttentionNet (Self-Attention based model)
- sklearn-like interface for easy integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data

    Architecture:
    - Input → 128 → 64 → 32 → Output
    - BatchNorm + Dropout + ReLU activations
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3
    ):
        """
        Initialize MLP

        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(TabularMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class TabularAttentionNet(nn.Module):
    """
    Self-Attention based network for tabular data

    Architecture:
    - Input → Feature Embedding
    - Self-Attention (multi-head) with skip connection
    - Feed-Forward → Output

    Improvements:
    - Better weight initialization
    - Pre-LayerNorm for stable training
    - Learnable positional encoding for features
    - Multiple attention layers option

    Attention weights can be used as feature importance!
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 2
    ):
        """
        Initialize AttentionNet

        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            num_layers: Number of attention layers
        """
        super(TabularAttentionNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Feature embedding: embed each feature to embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)

        # Learnable feature position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)

        # Create multiple attention layers
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.ff_norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Multi-head self-attention
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.norm_layers.append(nn.LayerNorm(embed_dim))

            # Feed-forward network
            self.ff_layers.append(nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),  # GELU works better than ReLU for transformers
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            ))
            self.ff_norm_layers.append(nn.LayerNorm(embed_dim))

        # Output layers with more capacity
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )

        # Store attention weights for feature importance
        self.attention_weights = None

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, n_features)

        Returns:
            Output logits (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # Embed each feature separately
        # (batch_size, n_features) → (batch_size, n_features, embed_dim)
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        x = self.feature_embed(x)  # (batch_size, n_features, embed_dim)

        # Add positional encoding
        x = x + self.pos_encoding

        # Apply attention layers
        for i in range(self.num_layers):
            # Pre-norm attention
            x_norm = self.norm_layers[i](x)
            attn_output, attn_weights = self.attention_layers[i](x_norm, x_norm, x_norm)

            # Store last layer's attention weights for feature importance
            if i == self.num_layers - 1:
                self.attention_weights = attn_weights

            # Residual connection
            x = x + attn_output

            # Pre-norm feed-forward
            x_norm = self.ff_norm_layers[i](x)
            ff_output = self.ff_layers[i](x_norm)
            x = x + ff_output

        # Global average pooling across features
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # Output with normalization
        x = self.output_norm(x)
        out = self.output(x)

        return out

    def get_attention_weights(self):
        """
        Get attention weights as feature importance

        Returns:
            Attention weights (averaged across heads)
            Shape: (n_features,)
        """
        if self.attention_weights is None:
            raise ValueError("No attention weights available. Run forward pass first.")

        # MultiheadAttention output shape: (batch_size, n_features, n_features)
        # when average_attn_weights=True (default)
        # or (batch_size, num_heads, n_features, n_features) when average_attn_weights=False
        weights = self.attention_weights

        # Handle different shapes
        if weights.dim() == 3:
            # (batch_size, n_features, n_features) - already averaged across heads
            weights = weights.mean(dim=0)  # (n_features, n_features)
        elif weights.dim() == 4:
            # (batch_size, num_heads, n_features, n_features)
            weights = weights.mean(dim=0).mean(dim=0)  # (n_features, n_features)
        else:
            raise ValueError(f"Unexpected attention weights shape: {weights.shape}")

        # Sum incoming attention (how much each feature is attended to)
        # This gives importance: features that are attended to more are more important
        feature_importance = weights.sum(dim=0)  # (n_features,)

        return feature_importance.detach().cpu().numpy()


class DLModelWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for PyTorch models

    Provides fit(), predict(), predict_proba() methods

    Features:
    - Mixed Precision Training (FP16) for GPU acceleration
    - Learning Rate Scheduler (Cosine Annealing)
    - Gradient Clipping for stable training
    - Early stopping option
    """

    def __init__(
        self,
        model_class,
        model_params: dict = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = None,
        random_state: int = 42,
        use_mixed_precision: bool = True,
        use_scheduler: bool = True,
        verbose: bool = True
    ):
        """
        Initialize wrapper

        Args:
            model_class: PyTorch model class (TabularMLP or TabularAttentionNet)
            model_params: Parameters for model initialization
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW optimizer
            device: Device ('cuda' or 'cpu', auto-detect if None)
            random_state: Random seed
            use_mixed_precision: Use FP16 training if GPU available
            use_scheduler: Use learning rate scheduler
            verbose: Print training progress
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.use_mixed_precision = use_mixed_precision and self.device == 'cuda'
        self.use_scheduler = use_scheduler
        self.verbose = verbose

        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.training_history = {'loss': [], 'lr': []}

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if self.device == 'cuda':
            torch.cuda.manual_seed(random_state)

    def fit(self, X, y):
        """
        Fit model

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Get classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize model
        input_dim = X.shape[1]
        self.model = self.model_class(
            input_dim=input_dim,
            output_dim=n_classes,
            **self.model_params
        ).to(self.device)

        # Create dataset and dataloader
        dataset = TabularDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device == 'cuda'),
            num_workers=0,  # Windows compatibility
            drop_last=True  # Drop last batch if size is 1 (BatchNorm issue)
        )

        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = None
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler
        amp_scaler = None
        if self.use_mixed_precision:
            amp_scaler = torch.amp.GradScaler('cuda')

        # Training loop
        self.model.train()
        self.training_history = {'loss': [], 'lr': []}

        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if self.use_mixed_precision:
                    # Mixed precision training
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    # Standard training
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update scheduler
            if scheduler:
                scheduler.step()

            avg_loss = total_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['loss'].append(avg_loss)
            self.training_history['lr'].append(current_lr)

            # Print progress every 10 epochs
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        return self

    def predict(self, X):
        """Predict class labels"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probas = F.softmax(outputs, dim=1)

        return probas.cpu().numpy()


if __name__ == '__main__':
    # Test models
    print("=" * 60)
    print("Deep Learning Models Test")
    print("=" * 60)

    # Check GPU availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_mem:.1f} GB")

    # Auto-select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate synthetic data - larger for better AttentionNet training
    np.random.seed(42)
    n_samples = 5000  # Increased from 1000 for better attention training
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Test MLP on GPU
    print("\n" + "=" * 60)
    print("[Testing MLP on GPU]" if device == 'cuda' else "[Testing MLP on CPU]")
    print("=" * 60)

    mlp_model = DLModelWrapper(
        model_class=TabularMLP,
        model_params={'hidden_dims': [128, 64, 32]},
        epochs=30,
        batch_size=64,
        learning_rate=0.001,
        device=device,
        use_mixed_precision=(device == 'cuda'),
        verbose=True
    )

    import time
    start_time = time.time()
    mlp_model.fit(X_train, y_train)
    mlp_time = time.time() - start_time

    y_pred = mlp_model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nMLP Test Accuracy: {accuracy:.4f}")
    print(f"Training time: {mlp_time:.2f}s")

    # Test AttentionNet on GPU
    print("\n" + "=" * 60)
    print("[Testing AttentionNet on GPU]" if device == 'cuda' else "[Testing AttentionNet on CPU]")
    print("=" * 60)

    attn_model = DLModelWrapper(
        model_class=TabularAttentionNet,
        model_params={
            'embed_dim': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'dropout': 0.2
        },
        epochs=50,  # More epochs for attention model
        batch_size=64,
        learning_rate=0.0005,  # Lower LR for attention
        device=device,
        use_mixed_precision=(device == 'cuda'),
        verbose=True
    )

    start_time = time.time()
    attn_model.fit(X_train, y_train)
    attn_time = time.time() - start_time

    y_pred = attn_model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nAttentionNet Test Accuracy: {accuracy:.4f}")
    print(f"Training time: {attn_time:.2f}s")

    # Test attention weights extraction
    print("\n" + "=" * 60)
    print("[Testing Attention Weights Extraction]")
    print("=" * 60)

    attn_model.model.eval()
    with torch.no_grad():
        # Scale test data the same way as training
        X_sample_scaled = attn_model.scaler.transform(X_test[:10])
        X_sample = torch.FloatTensor(X_sample_scaled).to(device)
        _ = attn_model.model(X_sample)
        weights = attn_model.model.get_attention_weights()
        print(f"Attention weights shape: {weights.shape}")
        print(f"Top 5 features by attention: {np.argsort(weights)[-5:][::-1]}")

    print("\n✅ Deep learning models test completed!")
