"""
Autoencoder model for style embeddings.

Days 6-7: Train an autoencoder, tune it, learn regularization.

You write this from scratch (mostly). The boilerplate gives you:
  - Module class and forward signature
  - Training loop skeleton with early stopping
  - Diagnostic helpers

You fill in:
  - Architecture (layer sizes, activations, dropout)
  - Hyperparameters (lr, weight_decay, patience)
  - Diagnostics (dead dims, effective rank, recon error per feature)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from datetime import datetime


# ============================================================================
# ARCHITECTURE
# ============================================================================

class StyleAE(nn.Module):
    """
    Simple autoencoder for style embeddings.
    
    Encoder: d_in -> 32 -> 16 -> d_latent
    Decoder: d_latent -> 16 -> 32 -> d_in
    
    Args:
        d_in: input feature dimension
        d_latent: latent space dimension (typically 8)
    """
    
    def __init__(self, d_in: int, d_latent: int = 8, dropout: float = 0.15):
        super().__init__()
        
        # You write the architecture
        # Recommendation: two hidden layers, 32 then 16, GELU activation, dropout
        self.enc = nn.Sequential(
            # First layer
            nn.Linear(d_in, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            # Second layer
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            # Latent (no activation)
            nn.Linear(16, d_latent),
        )
        
        self.dec = nn.Sequential(
            # Mirror the encoder
            nn.Linear(d_latent, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            # 
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            # Output
            nn.Linear(32, d_in),
        )
    
    def forward(self, x):
        """Forward pass."""
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


# ============================================================================
# TRAINING
# ============================================================================

def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      config: dict, device: str = "cpu") -> tuple:
    """
    Train the autoencoder with early stopping.
    
    Args:
        model: StyleAE instance
        train_loader, val_loader: PyTorch DataLoaders
        config: config dict with training hyperparameters
        device: "cpu" or "cuda"
    
    Returns:
        (trained model, history dict)
    """
    
    model = model.to(device)
    
    # Extract hyperparameters
    cfg = config["training"]["autoencoder"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    epochs_max = cfg["epochs_max"]
    patience = cfg["early_stopping_patience"]
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": [], "epoch": []}
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    print(f"Training on {device}...")
    print(f"Config: lr={lr}, weight_decay={weight_decay}, epochs_max={epochs_max}, patience={patience}")
    
    for epoch in range(epochs_max):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device) if isinstance(batch_x, (list, tuple)) else batch_x.to(device)
            
            x_hat, z = model(batch_x)
            loss = criterion(x_hat, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(device) if isinstance(batch_x, (list, tuple)) else batch_x.to(device)
                x_hat, z = model(batch_x)
                loss = criterion(x_hat, batch_x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (val loss not improving)")
            break
    
    return model, history


def run_diagnostics(model: nn.Module, train_loader: DataLoader, test_data: np.ndarray,
                    device: str = "cpu"):
    """
    Run diagnostic checks on the trained model.
    
    Checks:
      1. Dead latent dimensions (near-zero variance)
      2. Effective rank of latent space
      3. Per-feature reconstruction error
    """
    
    model = model.to(device)
    model.eval()
    
    # Encode all data to latent space
    latents = []
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(test_data).float().to(device)
        for i in range(0, len(X_test_tensor), 64):
            batch = X_test_tensor[i:i+64]
            _, z = model(batch)
            latents.append(z.cpu().numpy())
    
    Z = np.vstack(latents)
    
    print(f"\n{'='*70}")
    print("DIAGNOSTICS")
    print(f"{'='*70}")
    
    # Check 1: Dead dimensions
    print(f"\nDimension variance (should be non-zero for all):")
    dim_vars = Z.std(axis=0)
    for i, var in enumerate(dim_vars):
        marker = " <-- DEAD" if var < 0.01 else ""
        print(f"  Dim {i}: {var:.4f}{marker}")
    
    # Check 2: Effective rank
    print(f"\nEffective rank:")
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    explained = S.cumsum() / S.sum()
    eff_rank_90 = (explained < 0.9).sum()
    eff_rank_95 = (explained < 0.95).sum()
    print(f"  Singular values explain 90% at rank {eff_rank_90}")
    print(f"  Singular values explain 95% at rank {eff_rank_95}")
    print(f"  Total dims: {len(S)}")
    
    # Check 3: Per-feature reconstruction error
    print(f"\nPer-feature reconstruction error:")
    X_recon, _ = model(torch.from_numpy(test_data).float().to(device))
    X_recon = X_recon.cpu().detach().numpy()
    recon_error = ((test_data - X_recon) ** 2).mean(axis=0)
    top_hardest = np.argsort(-recon_error)[:5]
    for idx in top_hardest:
        print(f"  Feature {idx}: MSE {recon_error[idx]:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print("[Days 6-7] Autoencoder Training")
    
    # Load config
    with open("configs/v1.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading snapshots...")
    snapshots_train = pd.read_parquet("data/snapshots/v1/train.parquet")
    snapshots_val = pd.read_parquet("data/snapshots/v1/val.parquet")
    snapshots_test = pd.read_parquet("data/snapshots/v1/test.parquet")
    
    # Extract features
    feature_cols = [c for c in snapshots_train.columns 
                   if c not in ["fighter_id", "fight_id", "date", "split", "weight_class", "n_prior"]]
    
    X_train = snapshots_train[feature_cols].values.astype(np.float32)
    X_val = snapshots_val[feature_cols].values.astype(np.float32)
    X_test = snapshots_test[feature_cols].values.astype(np.float32)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create DataLoaders
    batch_size = config["training"]["autoencoder"]["batch_size"]
    train_dataset = TensorDataset(torch.from_numpy(X_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    d_in = X_train.shape[1]
    d_latent = args.latent_dim or config["training"]["autoencoder"]["d_latent"]
    device = config["training"]["device"]
    
    model = StyleAE(d_in=d_in, d_latent=d_latent)
    
    # Train
    model, history = train_autoencoder(model, train_loader, val_loader, config, device)
    
    # Run diagnostics
    run_diagnostics(model, train_loader, X_test, device)
    
    # Extract embeddings on all splits
    print(f"\nExtracting embeddings...")
    model.eval()
    with torch.no_grad():
        def encode(X):
            Z = []
            X_tensor = torch.from_numpy(X).float().to(device)
            for i in range(0, len(X_tensor), 64):
                batch = X_tensor[i:i+64]
                _, z = model(batch)
                Z.append(z.cpu().numpy())
            return np.vstack(Z)
        
        Z_train = encode(X_train)
        Z_val = encode(X_val)
        Z_test = encode(X_test)
    
    print(f"Train embeddings: {Z_train.shape}")
    print(f"Val embeddings: {Z_val.shape}")
    print(f"Test embeddings: {Z_test.shape}")
    
    # Run harness
    print(f"\n{'='*70}")
    print("Running evaluation harness...")
    print(f"{'='*70}")
    
    # Reconstruct meta
    snapshots = pd.concat([snapshots_train, snapshots_val, snapshots_test], ignore_index=False)
    meta = snapshots[["fighter_id", "fight_id", "date", "split"]].copy()
    
    # Load labels
    try:
        labels = pd.read_csv("data/labels/background.csv")
    except FileNotFoundError:
        labels = pd.DataFrame({"fighter_id": []})
    
    # Call harness on test
    from src.eval.harness import evaluate
    Z_all = np.vstack([Z_train, Z_val, Z_test])
    features = np.vstack([X_train, X_val, X_test])
    
    results = evaluate(Z_all, meta, features, labels, f"ae_{d_latent}", config)
    
    # Save model and metadata
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"data/models/ae_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Export weights to NumPy (for serving later)
    weights = []
    for module in model.enc.modules():
        if isinstance(module, nn.Linear):
            weights.append({
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy(),
            })
    
    np.savez(model_dir / "encoder_weights.npz", *[w.get("weight") for w in weights], *[w.get("bias") for w in weights])
    
    # Save config and results
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump({
            "d_in": d_in,
            "d_latent": d_latent,
            "feature_cols": feature_cols,
            "training": config["training"]["autoencoder"],
        }, f)
    
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Model and metadata saved to {model_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, help="Latent dimension (default from config)")
    parser.add_argument("--feature-block", choices=["style", "style+quality", "style+physical+dispersion", "all"],
                       default="style", help="Feature block to use")
    args = parser.parse_args()
    main(args)
