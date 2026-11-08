#!/usr/bin/env python3
"""
Masked Expression Recovery Experiment.

Task: Randomly mask a fraction of gene expressions per spot.
      Models must predict masked values using spatial neighbors.

Why this works for quality gate:
- Low mask rate (10%): neighbors still have most values -> spatial helps -> gate should OPEN
- High mask rate (50%): neighbors also missing many values -> spatial hurts -> gate should CLOSE
- This creates a natural test of whether quality-driven gating adapts to input reliability.

Models compared:
1. No-Spatial AutoEncoder (expression-only, no graph)
2. Neighbor Mean Imputation (simple spatial baseline)
3. GCN Imputation (STAGATE-like)
4. GAT Imputation
5. Quality-Driven Spatial GNN (ours)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Loading
# ============================================================

def load_data():
    print("Loading data...")
    adata = ad.read_h5ad(PROC_DIR / 'adata_processed.h5ad')
    edges_df = pd.read_csv(PROC_DIR / 'spatial_edges.csv')
    with open(PROC_DIR / 'data_splits.json') as f:
        splits = json.load(f)
    with open(PROC_DIR / 'metadata.json') as f:
        metadata = json.load(f)
    return adata, edges_df, splits, metadata


def prepare_recovery_data(adata, edges_df, splits, mask_rate=0.3, seed=42, n_top_genes=3000):
    """
    Prepare data for masked expression recovery.
    
    Feature selection is done on the raw/numpy level (top variable genes),
    then only selected genes are converted to dense torch tensors.
    """
    print(f"Converting to PyG format (mask_rate={mask_rate}, top_genes={n_top_genes})...")
    
    # Work with numpy for feature selection
    X_raw = adata.X
    if hasattr(X_raw, 'toarray'):
        # Sparse: compute variance per gene without full densification
        from scipy.sparse import issparse
        if issparse(X_raw):
            # Variance = E[X^2] - E[X]^2, computed column-wise on sparse
            X_csc = X_raw.tocsc()
            n = X_csc.shape[0]
            mean_sq = np.array(X_csc.power(2).mean(axis=0)).flatten()
            sq_mean = np.array(X_csc.mean(axis=0)).flatten() ** 2
            gene_var = mean_sq - sq_mean
        else:
            gene_var = X_raw.var(axis=0)
    else:
        gene_var = np.array(X_raw.var(axis=0)).flatten()
    
    # Select top variable genes
    top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
    top_idx = np.sort(top_idx)  # sort for consistency
    print(f"  Selected {len(top_idx)} top variable genes")
    
    # Now extract only selected genes and convert to dense
    if hasattr(X_raw, 'toarray'):
        X_sel = X_raw[:, top_idx].toarray().astype(np.float32)
    else:
        X_sel = np.array(X_raw[:, top_idx], dtype=np.float32)
    
    X_full = torch.tensor(X_sel, dtype=torch.float32)
    N, G = X_full.shape
    print(f"  Dense matrix: {N} spots x {G} genes")
    
    # Log-normalize
    X_log = torch.log1p(X_full)
    
    # Z-score per gene
    gene_mean = X_log.mean(dim=0, keepdim=True)
    gene_std = X_log.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_norm = (X_log - gene_mean) / gene_std
    
    # Create gene mask: randomly mask mask_rate fraction per spot
    torch.manual_seed(seed)
    N, G = X_norm.shape
    gene_mask = torch.rand(N, G) < mask_rate  # True = masked
    
    # Masked input: set masked positions to 0 in standardised space.
    # No mask-indicator channel is provided; all models receive the same input.
    x_masked = X_norm.clone()
    x_masked[gene_mask] = 0.0
    
    # Edge index
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    edge_attr = torch.tensor(edges_df['distance'].values, dtype=torch.float32).unsqueeze(1)
    
    # Node-level masks (same as before - sample-level split)
    node_train = torch.tensor(splits['train'], dtype=torch.bool)
    node_val = torch.tensor(splits['val'], dtype=torch.bool)
    node_test = torch.tensor(splits['test'], dtype=torch.bool)
    
    data = Data(
        x=x_masked,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_true=X_norm,          # target: full normalized expression
        x_full=X_full,          # original raw counts (for reporting)
        gene_mask=gene_mask,    # which (spot, gene) positions are masked
        gene_mean=gene_mean,
        gene_std=gene_std,
        train_mask=node_train,
        val_mask=node_val,
        test_mask=node_test,
    )
    
    total_elements = N * G
    masked_elements = gene_mask.sum().item()
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Genes: {data.num_features}")
    print(f"  Masked: {masked_elements}/{total_elements} ({masked_elements/total_elements*100:.1f}%)")
    print(f"  Mask rate per spot: ~{mask_rate*100:.0f}%")
    
    return data


# ============================================================
# Evaluation Metrics
# ============================================================

def evaluate_recovery(pred, target, gene_mask, node_mask, gene_mean, gene_std):
    """
    Evaluate expression recovery on masked positions.
    
    pred, target: [N, G] in normalized space
    gene_mask: [N, G] True = masked
    node_mask: [N] which spots to evaluate
    """
    # Only evaluate on masked positions within the node mask
    eval_mask = gene_mask[node_mask]  # [n_nodes, G]
    pred_eval = pred[node_mask][eval_mask]
    target_eval = target[node_mask][eval_mask]
    
    if len(pred_eval) == 0:
        return {'mse': float('nan'), 'mae': float('nan'), 'pearson': float('nan'), 'r2': float('nan')}
    
    # Denormalize for reporting
    # target_eval is in z-score space, convert back to log space
    # We need the gene_mean and gene_std for the masked genes
    gene_mask_indices = torch.where(eval_mask)[1]  # which gene each masked element belongs to
    means = gene_mean[0, gene_mask_indices]
    stds = gene_std[0, gene_mask_indices]
    
    pred_denorm = pred_eval * stds + means
    target_denorm = target_eval * stds + means
    
    # Metrics in log space
    mse = F.mse_loss(pred_denorm, target_denorm).item()
    mae = F.l1_loss(pred_denorm, target_denorm).item()
    
    # Pearson correlation (overall)
    pred_np = pred_denorm.detach().cpu().numpy()
    target_np = target_denorm.detach().cpu().numpy()
    
    if len(pred_np) > 2:
        pearson_r, _ = pearsonr(pred_np, target_np)
    else:
        pearson_r = float('nan')
    
    # R2 score
    ss_res = ((pred_np - target_np) ** 2).sum()
    ss_tot = ((target_np - target_np.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    
    return {'mse': mse, 'mae': mae, 'pearson': pearson_r, 'r2': r2}


# ============================================================
# Baseline 1: No-Spatial AutoEncoder
# ============================================================

class NoSpatialAE(nn.Module):
    """Predict masked genes from observed genes only, no spatial info."""
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


# ============================================================
# Baseline 2: Neighbor Mean Imputation
# ============================================================

class NeighborMeanImpute(nn.Module):
    """Replace masked values with neighbor average, then refine with MLP."""
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # concat original + neighbor avg
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index):
        # Compute neighbor mean via index_add
        row, col = edge_index
        N, G = x.shape
        
        neighbor_sum = torch.zeros(N, G, device=x.device)
        neighbor_sum.index_add_(0, row, x[col])
        
        neighbor_count = torch.zeros(N, 1, device=x.device)
        neighbor_count.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        neighbor_count = neighbor_count.clamp(min=1)
        neighbor_avg = neighbor_sum / neighbor_count
        
        combined = torch.cat([x, neighbor_avg], dim=1)
        return self.refiner(combined)


# ============================================================
# Baseline 3: GCN Imputation
# ============================================================

class GCNImputer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return self.decoder(h)


# ============================================================
# Baseline 4: GAT Imputer
# ============================================================

class GATImputer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        return self.decoder(h)


# ============================================================
# Our Method: Quality-Driven Spatial GNN for Recovery
# ============================================================

class QualityDrivenRecoveryGNN(nn.Module):
    """
    Quality-driven spatial gating for expression recovery.
    
    Two branches:
    1. Non-spatial: autoencoder on current spot's (masked) expression
    2. Spatial: aggregate neighbor expressions, then decode
    
    Quality gate decides per-spot how much to rely on spatial info.
    Gate inputs include:
    - Fraction of masked genes in current spot
    - Fraction of masked genes in neighbors
    - Branch disagreement
    - Neighborhood expression variance
    """
    
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        
        # Non-spatial branch: encode current spot expression
        self.nonspatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Spatial branch: encode neighbor aggregated expression
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Spatial aggregation: attention-weighted neighbor mean
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Quality gate
        # Inputs: h_nonspatial, h_spatial, disagreement, quality features
        # Quality features: self_mask_rate, neighbor_mask_rate, neighbor_var, rep_distance
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 4, hidden_dim),  # 2*hidden + 1(cos_sim) + 4 quality
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        # --- Non-spatial branch ---
        h_ns = self.nonspatial_encoder(x)
        
        # --- Spatial branch ---
        # First encode all nodes
        h_enc = self.spatial_encoder(x)
        
        # Attention-weighted aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        # Batch neighbor aggregation using scatter
        # For attention, we need per-edge scores
        if row.size(0) > 0:
            # Edge features: [src_feat, dst_feat]
            edge_feat = torch.cat([h_enc[row], h_enc[col]], dim=1)
            edge_attn = self.spatial_attention(edge_feat).squeeze(1)  # [E]
            
            # Softmax per destination node
            # For each node i, softmax over edges where col == i
            # Use scatter-based softmax
            edge_attn_max = torch.full((N,), float('-inf'), device=x.device)
            edge_attn_max.scatter_(0, col, torch.maximum(edge_attn_max[col], edge_attn))
            # Fix: use index_select for proper gathering
            edge_attn_max_gathered = edge_attn_max[col]
            edge_attn_exp = torch.exp(edge_attn - edge_attn_max_gathered)
            
            edge_attn_sum = torch.zeros(N, device=x.device)
            edge_attn_sum.scatter_add_(0, col, edge_attn_exp)
            edge_attn_sum = edge_attn_sum[col].clamp(min=1e-8)
            edge_attn_norm = edge_attn_exp / edge_attn_sum
            
            # Weighted aggregation
            weighted_feats = h_enc[row] * edge_attn_norm.unsqueeze(1)
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(weighted_feats), weighted_feats)
        
        # --- Quality features (all vectorized) ---
        # 1. Self mask rate
        if gene_mask is not None:
            self_mask_rate = gene_mask.float().mean(dim=1, keepdim=True)
        else:
            self_mask_rate = torch.zeros(N, 1, device=x.device)
        
        # 2. Neighbor mask rate (vectorized via scatter)
        neighbor_mask_rate = torch.zeros(N, 1, device=x.device)
        if gene_mask is not None:
            # For each node i, average the mask rate of its neighbors
            nbr_mask_vals = gene_mask[col].float().mean(dim=1, keepdim=True)  # [E, 1]
            neighbor_mask_sum = torch.zeros(N, 1, device=x.device)
            neighbor_mask_sum.scatter_add_(0, row.unsqueeze(1), nbr_mask_vals)
            neighbor_count = torch.zeros(N, 1, device=x.device)
            neighbor_count.scatter_add_(0, row.unsqueeze(1), torch.ones(row.size(0), 1, device=x.device))
            neighbor_mask_rate = neighbor_mask_sum / neighbor_count.clamp(min=1)
        
        # 3. Neighborhood expression variance (vectorized)
        neighbor_var = torch.zeros(N, 1, device=x.device)
        # Use scatter to compute neighbor mean and mean-of-squares
        x_sq = x * x
        nbr_sum = torch.zeros_like(x)
        nbr_sq_sum = torch.zeros_like(x)
        nbr_count = torch.zeros(N, 1, device=x.device)
        nbr_sum.scatter_add_(0, row.unsqueeze(1).expand_as(x[col]), x[col])
        nbr_sq_sum.scatter_add_(0, row.unsqueeze(1).expand_as(x_sq[col]), x_sq[col])
        nbr_count.scatter_add_(0, row.unsqueeze(1), torch.ones(row.size(0), 1, device=x.device))
        nbr_count_clamped = nbr_count.clamp(min=1)
        nbr_mean = nbr_sum / nbr_count_clamped
        nbr_var_per_gene = nbr_sq_sum / nbr_count_clamped - nbr_mean * nbr_mean
        neighbor_var = nbr_var_per_gene.mean(dim=1, keepdim=True).clamp(min=0)
        
        # 4. Representation distance
        rep_distance = F.pairwise_distance(h_ns, h_spatial, p=2).unsqueeze(1)
        
        # 5. Branch disagreement (cosine similarity)
        cos_sim = F.cosine_similarity(h_ns, h_spatial, dim=1).unsqueeze(1)
        
        # --- Compute gate ---
        gate_input = torch.cat([
            h_ns, h_spatial,
            cos_sim,
            self_mask_rate, neighbor_mask_rate, neighbor_var, rep_distance
        ], dim=1)
        gate = self.quality_gate(gate_input)
        
        # --- Fuse ---
        h_fused = gate * h_spatial + (1 - gate) * h_ns
        
        # --- Decode ---
        out = self.decoder(h_fused)
        
        return out, gate, h_ns, h_spatial


# ============================================================
# Training
# ============================================================

def train_model(model, data, epochs=200, lr=0.001, weight_decay=1e-4, 
                use_edges=True, model_type='standard'):
    """
    Train a recovery model.
    
    Loss = MSE on masked positions only.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    gate_history = []
    train_losses = []
    val_losses = []
    
    gene_mask = data.gene_mask
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if model_type == 'nospatial':
            pred = model(data.x)
        elif model_type == 'neighbor':
            pred = model(data.x, data.edge_index)
        elif model_type == 'quality':
            pred, gate, _, _ = model(data.x, data.edge_index, gene_mask)
        else:
            pred = model(data.x, data.edge_index)
        
        # Loss only on masked positions in training nodes
        train_gene_mask = gene_mask[data.train_mask]
        pred_train = pred[data.train_mask][train_gene_mask]
        target_train = data.y_true[data.train_mask][train_gene_mask]
        
        loss = F.mse_loss(pred_train, target_train)
        
        # Auxiliary loss for quality model: encourage gate diversity
        if model_type == 'quality':
            gate_train = gate[data.train_mask]
            gate_mean = gate_train.mean()
            gate_std = gate_train.std().clamp(min=1e-6)
            # Encourage gate to use full range (not collapse to 0 or 1)
            aux_loss = -gate_std + 0.1 * (gate_mean - 0.5).abs()
            loss = loss + 0.01 * aux_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == 'nospatial':
                pred = model(data.x)
            elif model_type == 'neighbor':
                pred = model(data.x, data.edge_index)
            elif model_type == 'quality':
                pred, gate, _, _ = model(data.x, data.edge_index, gene_mask)
            else:
                pred = model(data.x, data.edge_index)
            
            val_gene_mask = gene_mask[data.val_mask]
            pred_val = pred[data.val_mask][val_gene_mask]
            target_val = data.y_true[data.val_mask][val_gene_mask]
            val_loss = F.mse_loss(pred_val, target_val).item()
            val_losses.append(val_loss)
            
            if model_type == 'quality':
                gate_stats = {
                    'mean': gate[data.val_mask].mean().item(),
                    'std': gate[data.val_mask].std().item(),
                }
                gate_history.append(gate_stats)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 40 == 0:
            extra = ""
            if model_type == 'quality':
                extra = f", Gate: {gate_stats['mean']:.3f}+/-{gate_stats['std']:.3f}"
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss:.4f}{extra}")
    
    model.load_state_dict(best_model_state)
    return model, best_val_loss, gate_history, train_losses, val_losses


def evaluate_model(model, data, mask_name='test', use_edges=True, model_type='standard'):
    """Evaluate recovery performance."""
    model.eval()
    with torch.no_grad():
        if model_type == 'nospatial':
            pred = model(data.x)
            gate = None
        elif model_type == 'neighbor':
            pred = model(data.x, data.edge_index)
            gate = None
        elif model_type == 'quality':
            pred, gate, _, _ = model(data.x, data.edge_index, data.gene_mask)
        else:
            pred = model(data.x, data.edge_index)
            gate = None
        
        if mask_name == 'test':
            node_mask = data.test_mask
        elif mask_name == 'val':
            node_mask = data.val_mask
        else:
            node_mask = data.train_mask
        
        metrics = evaluate_recovery(pred, data.y_true, data.gene_mask, node_mask,
                                     data.gene_mean, data.gene_std)
        
        gate_stats = None
        if gate is not None:
            gate_stats = {
                'mean': gate[node_mask].mean().item(),
                'std': gate[node_mask].std().item(),
                'min': gate[node_mask].min().item(),
                'max': gate[node_mask].max().item(),
                'spatial_dominant': (gate[node_mask] > 0.5).float().mean().item(),
            }
        
        return metrics, gate_stats


# ============================================================
# Main Experiment
# ============================================================

def run_single_mask_rate(mask_rate, epochs=150, n_top_genes=3000):
    """Run all models at a single mask rate."""
    print(f"\n{'='*70}")
    print(f"MASK RATE = {mask_rate*100:.0f}%")
    print(f"{'='*70}\n")
    
    adata, edges_df, splits, metadata = load_data()
    data = prepare_recovery_data(adata, edges_df, splits, mask_rate=mask_rate, n_top_genes=n_top_genes)
    input_dim = data.num_features
    
    results = {}
    
    # 1. No-Spatial AutoEncoder
    print("\n1. No-Spatial AutoEncoder")
    model = NoSpatialAE(input_dim, hidden_dim=256)
    model, val_loss, _, _, _ = train_model(model, data, epochs=epochs, model_type='nospatial')
    test_m, _ = evaluate_model(model, data, 'test', model_type='nospatial')
    val_m, _ = evaluate_model(model, data, 'val', model_type='nospatial')
    results['No-Spatial AE'] = {'val': val_m, 'test': test_m}
    print(f"  Test: MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, "
          f"Pearson={test_m['pearson']:.4f}, R2={test_m['r2']:.4f}")
    
    # 2. Neighbor Mean Imputation
    print("\n2. Neighbor Mean Imputation + MLP")
    model = NeighborMeanImpute(input_dim, hidden_dim=256)
    model, val_loss, _, _, _ = train_model(model, data, epochs=epochs, model_type='neighbor')
    test_m, _ = evaluate_model(model, data, 'test', model_type='neighbor')
    val_m, _ = evaluate_model(model, data, 'val', model_type='neighbor')
    results['Neighbor Impute'] = {'val': val_m, 'test': test_m}
    print(f"  Test: MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, "
          f"Pearson={test_m['pearson']:.4f}, R2={test_m['r2']:.4f}")
    
    # 3. GCN Imputer
    print("\n3. GCN Imputer (STAGATE-like)")
    model = GCNImputer(input_dim, hidden_dim=256)
    model, val_loss, _, _, _ = train_model(model, data, epochs=epochs, model_type='gcn')
    test_m, _ = evaluate_model(model, data, 'test', model_type='gcn')
    val_m, _ = evaluate_model(model, data, 'val', model_type='gcn')
    results['GCN Imputer'] = {'val': val_m, 'test': test_m}
    print(f"  Test: MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, "
          f"Pearson={test_m['pearson']:.4f}, R2={test_m['r2']:.4f}")
    
    # 4. GAT Imputer
    print("\n4. GAT Imputer")
    model = GATImputer(input_dim, hidden_dim=256)
    model, val_loss, _, _, _ = train_model(model, data, epochs=epochs, model_type='gat')
    test_m, _ = evaluate_model(model, data, 'test', model_type='gat')
    val_m, _ = evaluate_model(model, data, 'val', model_type='gat')
    results['GAT Imputer'] = {'val': val_m, 'test': test_m}
    print(f"  Test: MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, "
          f"Pearson={test_m['pearson']:.4f}, R2={test_m['r2']:.4f}")
    
    # 5. Quality-Driven Spatial GNN (Ours)
    print("\n5. Quality-Driven Spatial GNN (Ours)")
    model = QualityDrivenRecoveryGNN(input_dim, hidden_dim=256)
    model, val_loss, gate_history, train_losses, val_losses = train_model(
        model, data, epochs=epochs, model_type='quality')
    test_m, test_gate = evaluate_model(model, data, 'test', model_type='quality')
    val_m, val_gate = evaluate_model(model, data, 'val', model_type='quality')
    results['Quality GNN (Ours)'] = {
        'val': val_m, 'test': test_m,
        'val_gate': val_gate, 'test_gate': test_gate,
        'gate_history': gate_history,
    }
    print(f"  Test: MSE={test_m['mse']:.4f}, MAE={test_m['mae']:.4f}, "
          f"Pearson={test_m['pearson']:.4f}, R2={test_m['r2']:.4f}")
    if test_gate:
        print(f"  Gate: mean={test_gate['mean']:.3f}, std={test_gate['std']:.3f}, "
              f"spatial_dominant={test_gate['spatial_dominant']*100:.1f}%")
    
    return results, data, gate_history


def main():
    all_results = {}
    
    # Run at multiple gene panel sizes (simulating targeted panels)
    # Extreme conditions: very small panels + very high mask rates
    for n_genes in [50, 100, 200, 500, 3000]:
        print(f"\n{'#'*70}")
        print(f"# GENE PANEL SIZE = {n_genes}")
        print(f"{'#'*70}")
        mask_rates = [0.1, 0.3, 0.5, 0.7, 0.9] if n_genes <= 200 else [0.1, 0.3, 0.5]
        for mask_rate in mask_rates:
            results, data, gate_history = run_single_mask_rate(mask_rate, epochs=150, n_top_genes=n_genes)
            all_results[f"{n_genes}genes_{mask_rate}"] = results
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("OVERALL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    gene_panels = [50, 100, 200, 500, 3000]
    mask_rates_all = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for n_genes in gene_panels:
        mr_list = [0.1, 0.3, 0.5, 0.7, 0.9] if n_genes <= 200 else [0.1, 0.3, 0.5]
        for mask_rate in mr_list:
            key = f"{n_genes}genes_{mask_rate}"
            if key not in all_results:
                continue
            print(f"\n--- {n_genes} Genes, Mask Rate {mask_rate*100:.0f}% ---")
            print(f"{'Method':25s} {'MSE':>8s} {'MAE':>8s} {'Pearson':>8s} {'R2':>8s} {'Gate':>8s}")
            print("-" * 70)
            for method, res in all_results[key].items():
                test = res['test']
                gate_str = ""
                if 'test_gate' in res and res['test_gate']:
                    gate_str = f"{res['test_gate']['mean']:.3f}"
                print(f"{method:25s} {test['mse']:8.4f} {test['mae']:8.4f} "
                      f"{test['pearson']:8.4f} {test['r2']:8.4f} {gate_str:>8s}")
    
    # Save results with custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.dim() == 0 else obj.tolist()
            return super().default(obj)
    
    results_path = EXP_DIR / 'recovery_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")
    
    # ============================================================
    # Key comparison: How does spatial help when gene panel is small?
    # ============================================================
    print(f"\n{'='*70}")
    print("KEY COMPARISON: No-Spatial AE vs Quality GNN by Gene Panel Size")
    print(f"{'='*70}")
    print(f"{'Panel':>6s} {'Mask':>5s} {'NoSpat_R2':>10s} {'Qual_R2':>10s} {'Delta':>8s} {'Gate':>6s}")
    print("-" * 50)
    for n_genes in gene_panels:
        mr_list = [0.1, 0.3, 0.5, 0.7, 0.9] if n_genes <= 200 else [0.1, 0.3, 0.5]
        for mr in mr_list:
            key = f"{n_genes}genes_{mr}"
            if key not in all_results:
                continue
            nospatial_r2 = all_results[key]['No-Spatial AE']['test']['r2']
            qual_r2 = all_results[key]['Quality GNN (Ours)']['test']['r2']
            delta = qual_r2 - nospatial_r2
            tg = all_results[key]['Quality GNN (Ours)'].get('test_gate')
            gate_str = f"{tg['mean']:.3f}" if tg else "N/A"
            print(f"{n_genes:>6d} {mr*100:>4.0f}% {nospatial_r2:>10.4f} {qual_r2:>10.4f} {delta:>+8.4f} {gate_str:>6s}")
    
    # ============================================================
    # Plots
    # ============================================================
    
    # Plot 1: R2 comparison - focus on small gene panels where differences are largest
    focus_panels = [50, 100, 200]
    focus_masks = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(len(focus_panels), len(focus_masks), figsize=(20, 12))
    methods = ['No-Spatial AE', 'Neighbor Impute', 'GCN Imputer', 'GAT Imputer', 'Quality GNN (Ours)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, n_genes in enumerate(focus_panels):
        for j, mr in enumerate(focus_masks):
            ax = axes[i, j]
            key = f"{n_genes}genes_{mr}"
            if key not in all_results:
                ax.set_visible(False)
                continue
            vals = [all_results[key][m]['test']['r2'] for m in methods]
            bars = ax.bar(range(len(methods)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.split()[0] for m in methods], fontsize=5, rotation=45, ha='right')
            ax.set_title(f'{n_genes}g/{mr*100:.0f}%', fontsize=9)
            ymin = min(vals) - 0.02
            ymax = max(vals) + 0.02
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3, axis='y')
            best_idx = np.argmax(vals)
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            if i == 0:
                ax.set_xlabel('')
    
    plt.suptitle('R2 Score by Method (small gene panels: 50/100/200 genes)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'recovery_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {EXP_DIR / 'recovery_comparison.png'}")
    
    # Plot 2: Delta (Quality GNN - No-Spatial AE) heatmap - all conditions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    all_panels = [50, 100, 200, 500, 3000]
    all_masks = [0.1, 0.3, 0.5, 0.7, 0.9]
    delta_matrix = np.full((len(all_panels), len(all_masks)), np.nan)
    
    for i, n_genes in enumerate(all_panels):
        for j, mr in enumerate(all_masks):
            key = f"{n_genes}genes_{mr}"
            if key in all_results:
                nospatial_r2 = all_results[key]['No-Spatial AE']['test']['r2']
                qual_r2 = all_results[key]['Quality GNN (Ours)']['test']['r2']
                delta_matrix[i, j] = qual_r2 - nospatial_r2
    
    im = ax.imshow(delta_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(all_masks)))
    ax.set_xticklabels([f'{mr*100:.0f}%' for mr in all_masks])
    ax.set_yticks(range(len(all_panels)))
    ax.set_yticklabels([f'{ng} genes' for ng in all_panels])
    ax.set_xlabel('Mask Rate')
    ax.set_ylabel('Gene Panel Size')
    ax.set_title('Quality GNN R2 Improvement over No-Spatial AE\n(green = spatial helps, red = spatial hurts)')
    for i in range(len(all_panels)):
        for j in range(len(all_masks)):
            if not np.isnan(delta_matrix[i, j]):
                ax.text(j, i, f'{delta_matrix[i,j]:+.4f}', ha='center', va='center', fontsize=9,
                       fontweight='bold' if delta_matrix[i,j] > 0.005 else 'normal')
    plt.colorbar(im, label='R2 Delta')
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'r2_delta_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Delta heatmap saved to {EXP_DIR / 'r2_delta_heatmap.png'}")
    
    # Plot 3: Gate behavior across all conditions
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    gate_panels = [50, 100, 200, 500, 3000]
    gate_masks = [0.1, 0.3, 0.5]
    gate_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    x = np.arange(len(gate_masks))
    width = 0.15
    for k, n_genes in enumerate(gate_panels):
        gate_vals = []
        for mr in gate_masks:
            key = f"{n_genes}genes_{mr}"
            tg = all_results[key]['Quality GNN (Ours)'].get('test_gate')
            gate_vals.append(tg['mean'] if tg else 0)
        ax.bar(x + k * width, gate_vals, width, label=f'{n_genes} genes', color=gate_colors[k], alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.set_xlabel('Mask Rate')
    ax.set_ylabel('Gate Mean (test)')
    ax.set_title('Quality Gate Activation: How Much Does Model Rely on Spatial Info?')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'{mr*100:.0f}%' for mr in gate_masks])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'gate_summary.png', dpi=150, bbox_inches='tight')
    print(f"Gate summary plot saved to {EXP_DIR / 'gate_summary.png'}")


if __name__ == "__main__":
    main()
