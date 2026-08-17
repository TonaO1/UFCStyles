"""
Contrastive learning (SimCLR-style) for style embeddings.

Day 8: Design non-trivial positive pairs, train with NT-Xent loss, evaluate.

You decide:
  - Pair strategy: gapped (snapshots i, i+5), round-subsample, or feature dropout
  - Loss function details
  - Projector architecture
  
The harness runs the same way as for AE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from datetime import datetime


# ============================================================================
# ENCODER (REUSE FROM AE IF AVAILABLE)
# ============================================================================

class StyleEncoder(nn.Module):
    """Encoder (same as AE encoder)."""
    
    def __init__(self, d_in: int, d_latent: int = 8, dropout: float = 0.15):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, d_latent),
        )
    
    def forward(self, x):
        return self.enc(x)


class StyleContrastive(nn.Module):
    """
    Encoder + projector for contrastive learning.
    
    Follows SimCLR: encoder -> latent -> projector -> loss space.
    For embedding extraction, use encoder output (latent), not projector.
    """
    
    def __init__(self, d_in: int, d_latent: int = 8, d_proj: int = 16, dropout: float = 0.15):
        super().__init__()
        self.encoder = StyleEncoder(d_in, d_latent, dropout)
        
        # Projector: latent -> d_proj -> 16 (or 128 in bigger models)
        self.projector = nn.Sequential(
            nn.Linear(d_latent, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 16),
        )
    
    def forward(self, x):
        """Return both latent (for embedding) and projected (for loss)."""
        z = self.encoder(x)
        p = self.projector(z)
        return z, p


# ============================================================================
# CONTRASTIVE LOSS
# ============================================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) loss with in-batch negatives.
    
    Args:
        z1, z2: (batch_size, d_proj) projected vectors from two views
        tau: temperature parameter
    
    Returns:
        scalar loss
    """
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Cosine similarity: (batch_size, batch_size)
    logits = (z1 @ z2.T) / tau
    
    # Labels: [0, 1, 2, ..., batch_size-1]
    # (i, i) should be the highest logit
    labels = torch.arange(len(z1), device=z1.device)
    
    # Symmetric loss: both (z1, z2) and (z2, z1)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    
    return loss / 2


# ============================================================================
# POSITIVE PAIR STRATEGIES
# ============================================================================

class ContrastiveSampler(Sampler):
    """
    Sample pairs of snapshots as (view1, view2).
    
    Strategy options:
      - "gapped": pair snapshot i with snapshot i+gap for same fighter
      - "round_subsample": split prior fights into two halves, build two vectors
      - "feature_dropout": augment same snapshot with random feature masking
    """
    
    def __init__(self, snapshots_df: pd.DataFrame, strategy: str = "gapped", gap: int = 5):
        self.snapshots = snapshots_df.reset_index(drop=True)
        self.strategy = strategy
        self.gap = gap
        
        if strategy == "gapped":
            # Build pairs: (i, i+gap) for same fighter
            self.pairs = []
            for fighter_id in self.snapshots["fighter_id"].unique():
                fighter_snaps = self.snapshots[self.snapshots["fighter_id"] == fighter_id].index.tolist()
                for i in range(len(fighter_snaps) - self.gap):
                    self.pairs.append((fighter_snaps[i], fighter_snaps[i + self.gap]))
        
        else:
            # Simpler: all rows are valid samples, augmentation happens in dataset
            self.pairs = [(i, i) for i in range(len(self.snapshots))]
    
    def __iter__(self):
        np.random.shuffle(self.pairs)
        return iter(self.pairs)
    
    def __len__(self):
        return len(self.pairs)


class ContrastiveDataset(TensorDataset):
    """
    Dataset that returns augmented pairs for contrastive learning.
    
    For strategy="gapped", returns two separate snapshots.
    For strategy="feature_dropout", returns same snapshot twice with dropout.
    """
    
    def __init__(self, X: np.ndarray, snapshots_df: pd.DataFrame,
                 strategy: str = "gapped", dropout_rate: float = 0.2):
        self.X = X
        self.snapshots = snapshots_df
        self.strategy = strategy
        self.dropout_rate = dropout_rate
    
    def __getitem__(self, idx):
        if self.strategy == "gapped":
            # Two separate rows
            i, j = idx
            x1 = torch.from_numpy(self.X[i]).float()
            x2 = torch.from_numpy(self.X[j]).float()
        
        else:  # "feature_dropout"
            x_orig = torch.from_numpy(self.X[idx[0]]).float()
            x1 = x_orig.clone()
            x2 = x_orig.clone()
            
            # Randomly dropout features
            mask = torch.rand(len(x1)) > self.dropout_rate
            x1[~mask] = 0
            x2[~mask] = 0
        
        return x1, x2


# ============================================================================
# TRAINING
# ============================================================================

def train_contrastive(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      config: dict, device: str = "cpu") -> tuple:
    """Train the contrastive model with NT-Xent loss."""
    
    model = model.to(device)
    
    cfg = config["training"]["contrastive"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    epochs_max = cfg["epochs_max"]
    patience = cfg["early_stopping_patience"]
    tau = cfg["nt_xent_tau"]
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {"train_loss": [], "val_loss": [], "epoch": []}
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    print(f"Training on {device}...")
    print(f"Config: lr={lr}, weight_decay={weight_decay}, tau={tau}, patience={patience}")
    
    for epoch in range(epochs_max):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x1, x2 = batch[0], batch[1]
            else:
                x1, x2 = batch, batch
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            # Forward
            z1, p1 = model(x1)
            z2, p2 = model(x2)
            
            # Loss on projector outputs
            loss = nt_xent_loss(p1, p2, tau=tau)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x1, x2 = batch[0], batch[1]
                else:
                    x1, x2 = batch, batch
                
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                z1, p1 = model(x1)
                z2, p2 = model(x2)
                loss = nt_xent_loss(p1, p2, tau=tau)
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
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print("[Day 8] Contrastive Learning")
    
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
    
    # Create datasets and loaders
    strategy = config["training"]["contrastive"]["pair_strategy"]
    batch_size = config["training"]["contrastive"]["batch_size"]
    
    print(f"Using pair strategy: {strategy}")
    
    if strategy == "gapped":
        sampler = ContrastiveSampler(snapshots_train, strategy="gapped", gap=5)
        train_dataset = ContrastiveDataset(X_train, snapshots_train, strategy="gapped")
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, shuffle=False)
    else:
        train_dataset = ContrastiveDataset(X_train, snapshots_train, strategy=strategy)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ContrastiveDataset(X_val, snapshots_val, strategy="feature_dropout")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    d_in = X_train.shape[1]
    d_latent = args.latent_dim or config["training"]["contrastive"]["d_latent"]
    d_proj = config["training"]["contrastive"]["d_proj"]
    device = config["training"]["device"]
    
    model = StyleContrastive(d_in=d_in, d_latent=d_latent, d_proj=d_proj)
    
    # Train
    model, history = train_contrastive(model, train_loader, val_loader, config, device)
    
    # Extract embeddings (use encoder, not projector)
    print(f"\nExtracting embeddings...")
    model.eval()
    with torch.no_grad():
        def encode(X):
            Z = []
            X_tensor = torch.from_numpy(X).float().to(device)
            for i in range(0, len(X_tensor), 64):
                batch = X_tensor[i:i+64]
                z, _ = model(batch)
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
    
    snapshots = pd.concat([snapshots_train, snapshots_val, snapshots_test], ignore_index=False)
    meta = snapshots[["fighter_id", "fight_id", "date", "split"]].copy()
    
    try:
        labels = pd.read_csv("data/labels/background.csv")
    except FileNotFoundError:
        labels = pd.DataFrame({"fighter_id": []})
    
    from src.eval.harness import evaluate
    Z_all = np.vstack([Z_train, Z_val, Z_test])
    features = np.vstack([X_train, X_val, X_test])
    
    results = evaluate(Z_all, meta, features, labels, f"contrastive_{d_latent}", config)
    
    # Save
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"data/models/contrastive_{run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    weights = []
    for module in model.encoder.enc.modules():
        if isinstance(module, nn.Linear):
            weights.append({
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy(),
            })
    
    np.savez(model_dir / "encoder_weights.npz", *[w.get("weight") for w in weights], *[w.get("bias") for w in weights])
    
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump({
            "d_in": d_in,
            "d_latent": d_latent,
            "d_proj": d_proj,
            "pair_strategy": strategy,
            "feature_cols": feature_cols,
            "training": config["training"]["contrastive"],
        }, f)
    
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Model saved to {model_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, help="Latent dimension (default from config)")
    args = parser.parse_args()
    main(args)
