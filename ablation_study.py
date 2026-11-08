#!/usr/bin/env python3
"""
Ablation Study for Quality-Driven Spatial GNN.

Tests the necessity of each component:
1. Full Model (Ours): Quality-Driven Spatial GNN with all components
2. No Gate (Fixed Fusion): Replace learned gate with fixed 0.5
3. No Quality Features: Remove quality features from gate input
4. No Attention (Mean Agg): Replace attention with simple mean aggregation
5. No Spatial Branch: Only non-spatial branch (gate=0)
6. No Non-Spatial Branch: Only spatial branch (gate=1)

Focus on most informative conditions:
- 50 genes + 90% mask (maximum spatial benefit)
- 200 genes + 70% mask (moderate spatial benefit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Loading (reuse from main experiment)
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
    from scipy.sparse import issparse
    
    print(f"Converting to PyG format (mask_rate={mask_rate}, top_genes={n_top_genes})...")
    
    X_raw = adata.X
    if hasattr(X_raw, 'toarray'):
        if issparse(X_raw):
            X_csc = X_raw.tocsc()
            mean_sq = np.array(X_csc.power(2).mean(axis=0)).flatten()
            sq_mean = np.array(X_csc.mean(axis=0)).flatten() ** 2
            gene_var = mean_sq - sq_mean
        else:
            gene_var = X_raw.var(axis=0)
    else:
        gene_var = np.array(X_raw.var(axis=0)).flatten()
    
    top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
    top_idx = np.sort(top_idx)
    print(f"  Selected {len(top_idx)} top variable genes")
    
    if hasattr(X_raw, 'toarray'):
        X_sel = X_raw[:, top_idx].toarray().astype(np.float32)
    else:
        X_sel = np.array(X_raw[:, top_idx], dtype=np.float32)
    
    X_full = torch.tensor(X_sel, dtype=torch.float32)
    N, G = X_full.shape
    
    # Log-normalize
    X_log = torch.log1p(X_full)
    
    # Z-score per gene
    gene_mean = X_log.mean(dim=0, keepdim=True)
    gene_std = X_log.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_norm = (X_log - gene_mean) / gene_std
    
    # Apply mask
    torch.manual_seed(seed)
    gene_mask = torch.bernoulli(torch.full((N, G), mask_rate)).bool()
    X_masked = X_norm.clone()
    X_masked[gene_mask] = 0.0
    
    # Edge index
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    
    # Masks
    train_mask = torch.tensor(splits['train'], dtype=torch.bool)
    val_mask = torch.tensor(splits['val'], dtype=torch.bool)
    test_mask = torch.tensor(splits['test'], dtype=torch.bool)
    
    data = Data(
        x=X_masked,
        y_true=X_norm,
        edge_index=edge_index,
        gene_mask=gene_mask,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        gene_mean=gene_mean,
        gene_std=gene_std,
    )
    
    data.num_features = G
    print(f"  Nodes: {data.num_nodes}, Features: {data.num_features}")
    print(f"  Mask rate: {gene_mask.float().mean():.3f}")
    
    return data


# ============================================================
# Model Variants for Ablation
# ============================================================

class FullQualityGNN(nn.Module):
    """Full model: all components."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.nonspatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Full gate input: h_ns + h_spatial + cos_sim + 4 quality features
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_ns = self.nonspatial_encoder(x)
        h_enc = self.spatial_encoder(x)
        
        # Attention-weighted aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        if row.size(0) > 0:
            edge_feat = torch.cat([h_enc[row], h_enc[col]], dim=1)
            edge_attn = self.spatial_attention(edge_feat).squeeze(1)
            
            edge_attn_max = torch.full((N,), float('-inf'), device=x.device)
            edge_attn_max.scatter_(0, col, torch.maximum(edge_attn_max[col], edge_attn))
            edge_attn_max_gathered = edge_attn_max[col]
            edge_attn_exp = torch.exp(edge_attn - edge_attn_max_gathered)
            
            edge_attn_sum = torch.zeros(N, device=x.device)
            edge_attn_sum.scatter_add_(0, col, edge_attn_exp)
            edge_attn_sum = edge_attn_sum[col].clamp(min=1e-8)
            edge_attn_norm = edge_attn_exp / edge_attn_sum
            
            weighted_feats = h_enc[row] * edge_attn_norm.unsqueeze(1)
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(weighted_feats), weighted_feats)
        
        # Quality features
        if gene_mask is not None:
            self_mask_rate = gene_mask.float().mean(dim=1, keepdim=True)
        else:
            self_mask_rate = torch.zeros(N, 1, device=x.device)
        
        neighbor_mask_rate = torch.zeros(N, 1, device=x.device)
        if gene_mask is not None:
            nbr_mask_vals = gene_mask[col].float().mean(dim=1, keepdim=True)
            neighbor_mask_sum = torch.zeros(N, 1, device=x.device)
            neighbor_mask_sum.scatter_add_(0, row.unsqueeze(1), nbr_mask_vals)
            neighbor_count = torch.zeros(N, 1, device=x.device)
            neighbor_count.scatter_add_(0, row.unsqueeze(1), torch.ones(row.size(0), 1, device=x.device))
            neighbor_mask_rate = neighbor_mask_sum / neighbor_count.clamp(min=1)
        
        neighbor_var = torch.zeros(N, 1, device=x.device)
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
        
        rep_distance = F.pairwise_distance(h_ns, h_spatial, p=2).unsqueeze(1)
        cos_sim = F.cosine_similarity(h_ns, h_spatial, dim=1).unsqueeze(1)
        
        gate_input = torch.cat([
            h_ns, h_spatial,
            cos_sim,
            self_mask_rate, neighbor_mask_rate, neighbor_var, rep_distance
        ], dim=1)
        gate = self.quality_gate(gate_input)
        
        h_fused = gate * h_spatial + (1 - gate) * h_ns
        out = self.decoder(h_fused)
        
        return out, gate, h_ns, h_spatial


class NoGateFixedFusion(nn.Module):
    """Ablation: Fixed 0.5 fusion instead of learned gate."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.nonspatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # No gate network - fixed 0.5
        self.fixed_gate = 0.5
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_ns = self.nonspatial_encoder(x)
        h_enc = self.spatial_encoder(x)
        
        # Attention-weighted aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        if row.size(0) > 0:
            edge_feat = torch.cat([h_enc[row], h_enc[col]], dim=1)
            edge_attn = self.spatial_attention(edge_feat).squeeze(1)
            
            edge_attn_max = torch.full((N,), float('-inf'), device=x.device)
            edge_attn_max.scatter_(0, col, torch.maximum(edge_attn_max[col], edge_attn))
            edge_attn_max_gathered = edge_attn_max[col]
            edge_attn_exp = torch.exp(edge_attn - edge_attn_max_gathered)
            
            edge_attn_sum = torch.zeros(N, device=x.device)
            edge_attn_sum.scatter_add_(0, col, edge_attn_exp)
            edge_attn_sum = edge_attn_sum[col].clamp(min=1e-8)
            edge_attn_norm = edge_attn_exp / edge_attn_sum
            
            weighted_feats = h_enc[row] * edge_attn_norm.unsqueeze(1)
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(weighted_feats), weighted_feats)
        
        # Fixed 0.5 fusion
        gate = torch.full((N, 1), self.fixed_gate, device=x.device)
        h_fused = self.fixed_gate * h_spatial + (1 - self.fixed_gate) * h_ns
        out = self.decoder(h_fused)
        
        return out, gate, h_ns, h_spatial


class NoQualityFeatures(nn.Module):
    """Ablation: Gate uses only branch representations, no quality features."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.nonspatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Gate input: only h_ns + h_spatial + cos_sim (no quality features)
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_ns = self.nonspatial_encoder(x)
        h_enc = self.spatial_encoder(x)
        
        # Attention-weighted aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        if row.size(0) > 0:
            edge_feat = torch.cat([h_enc[row], h_enc[col]], dim=1)
            edge_attn = self.spatial_attention(edge_feat).squeeze(1)
            
            edge_attn_max = torch.full((N,), float('-inf'), device=x.device)
            edge_attn_max.scatter_(0, col, torch.maximum(edge_attn_max[col], edge_attn))
            edge_attn_max_gathered = edge_attn_max[col]
            edge_attn_exp = torch.exp(edge_attn - edge_attn_max_gathered)
            
            edge_attn_sum = torch.zeros(N, device=x.device)
            edge_attn_sum.scatter_add_(0, col, edge_attn_exp)
            edge_attn_sum = edge_attn_sum[col].clamp(min=1e-8)
            edge_attn_norm = edge_attn_exp / edge_attn_sum
            
            weighted_feats = h_enc[row] * edge_attn_norm.unsqueeze(1)
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(weighted_feats), weighted_feats)
        
        # Only cos_sim, no quality features
        cos_sim = F.cosine_similarity(h_ns, h_spatial, dim=1).unsqueeze(1)
        
        gate_input = torch.cat([h_ns, h_spatial, cos_sim], dim=1)
        gate = self.quality_gate(gate_input)
        
        h_fused = gate * h_spatial + (1 - gate) * h_ns
        out = self.decoder(h_fused)
        
        return out, gate, h_ns, h_spatial


class NoAttentionMeanAgg(nn.Module):
    """Ablation: Simple mean aggregation instead of attention."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.nonspatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # No attention network
        
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_ns = self.nonspatial_encoder(x)
        h_enc = self.spatial_encoder(x)
        
        # Simple mean aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        if row.size(0) > 0:
            # Sum neighbor features
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(h_enc[row]), h_enc[row])
            # Count neighbors
            neighbor_count = torch.zeros(N, 1, device=x.device)
            neighbor_count.scatter_add_(0, row.unsqueeze(1), torch.ones(row.size(0), 1, device=x.device))
            neighbor_count = neighbor_count.clamp(min=1)
            h_spatial = h_spatial / neighbor_count
        
        # Quality features
        if gene_mask is not None:
            self_mask_rate = gene_mask.float().mean(dim=1, keepdim=True)
        else:
            self_mask_rate = torch.zeros(N, 1, device=x.device)
        
        neighbor_mask_rate = torch.zeros(N, 1, device=x.device)
        if gene_mask is not None:
            nbr_mask_vals = gene_mask[col].float().mean(dim=1, keepdim=True)
            neighbor_mask_sum = torch.zeros(N, 1, device=x.device)
            neighbor_mask_sum.scatter_add_(0, row.unsqueeze(1), nbr_mask_vals)
            neighbor_count = torch.zeros(N, 1, device=x.device)
            neighbor_count.scatter_add_(0, row.unsqueeze(1), torch.ones(row.size(0), 1, device=x.device))
            neighbor_mask_rate = neighbor_mask_sum / neighbor_count.clamp(min=1)
        
        neighbor_var = torch.zeros(N, 1, device=x.device)
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
        
        rep_distance = F.pairwise_distance(h_ns, h_spatial, p=2).unsqueeze(1)
        cos_sim = F.cosine_similarity(h_ns, h_spatial, dim=1).unsqueeze(1)
        
        gate_input = torch.cat([
            h_ns, h_spatial,
            cos_sim,
            self_mask_rate, neighbor_mask_rate, neighbor_var, rep_distance
        ], dim=1)
        gate = self.quality_gate(gate_input)
        
        h_fused = gate * h_spatial + (1 - gate) * h_ns
        out = self.decoder(h_fused)
        
        return out, gate, h_ns, h_spatial


class OnlySpatialBranch(nn.Module):
    """Ablation: Only spatial branch, no non-spatial branch (gate=1)."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_enc = self.spatial_encoder(x)
        
        # Attention-weighted aggregation
        row, col = edge_index
        h_spatial = torch.zeros_like(h_enc)
        
        if row.size(0) > 0:
            edge_feat = torch.cat([h_enc[row], h_enc[col]], dim=1)
            edge_attn = self.spatial_attention(edge_feat).squeeze(1)
            
            edge_attn_max = torch.full((N,), float('-inf'), device=x.device)
            edge_attn_max.scatter_(0, col, torch.maximum(edge_attn_max[col], edge_attn))
            edge_attn_max_gathered = edge_attn_max[col]
            edge_attn_exp = torch.exp(edge_attn - edge_attn_max_gathered)
            
            edge_attn_sum = torch.zeros(N, device=x.device)
            edge_attn_sum.scatter_add_(0, col, edge_attn_exp)
            edge_attn_sum = edge_attn_sum[col].clamp(min=1e-8)
            edge_attn_norm = edge_attn_exp / edge_attn_sum
            
            weighted_feats = h_enc[row] * edge_attn_norm.unsqueeze(1)
            h_spatial.scatter_add_(0, col.unsqueeze(1).expand_as(weighted_feats), weighted_feats)
        
        # Gate always 1 (only spatial)
        gate = torch.ones(N, 1, device=x.device)
        h_fused = h_spatial
        out = self.decoder(h_fused)
        
        # Dummy h_ns for compatibility
        h_ns = torch.zeros_like(h_spatial)
        
        return out, gate, h_ns, h_spatial


class OnlyNonSpatialBranch(nn.Module):
    """Ablation: Only non-spatial branch, no spatial branch (gate=0)."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.nonspatial_encoder = nn.Sequential(
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
    
    def forward(self, x, edge_index, gene_mask=None):
        N = x.size(0)
        
        h_ns = self.nonspatial_encoder(x)
        
        # Gate always 0 (only non-spatial)
        gate = torch.zeros(N, 1, device=x.device)
        h_fused = h_ns
        out = self.decoder(h_fused)
        
        # Dummy h_spatial for compatibility
        h_spatial = torch.zeros_like(h_ns)
        
        return out, gate, h_ns, h_spatial


# ============================================================
# Training and Evaluation
# ============================================================

def evaluate_recovery(pred, target, gene_mask, node_mask, gene_mean, gene_std):
    """Evaluate recovery performance on masked positions."""
    pred_np = pred[node_mask].cpu().numpy()
    target_np = target[node_mask].cpu().numpy()
    mask_np = gene_mask[node_mask].cpu().numpy()
    
    # Denormalize
    gm = gene_mean.cpu().numpy()
    gs = gene_std.cpu().numpy()
    pred_denorm = pred_np * gs + gm
    target_denorm = target_np * gs + gm
    
    # Only evaluate masked positions - flatten to get all masked gene values
    pred_masked = pred_denorm[mask_np]  # 1D array of all masked values
    target_masked = target_denorm[mask_np]  # 1D array
    
    mse = mean_squared_error(target_masked, pred_masked)
    mae = mean_absolute_error(target_masked, pred_masked)
    
    # Pearson correlation per spot (across genes), then average
    pearson_vals = []
    for i in range(pred_np.shape[0]):
        spot_mask = mask_np[i]
        if spot_mask.sum() > 2:  # Need at least 3 values for correlation
            pred_spot = pred_denorm[i][spot_mask]
            target_spot = target_denorm[i][spot_mask]
            if target_spot.std() > 1e-8 and pred_spot.std() > 1e-8:
                r, _ = pearsonr(target_spot, pred_spot)
                pearson_vals.append(r)
    pearson = np.mean(pearson_vals) if pearson_vals else 0.0
    
    # R2 score (overall)
    ss_res = np.sum((target_masked - pred_masked) ** 2)
    ss_tot = np.sum((target_masked - target_masked.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'pearson': float(pearson),
        'r2': float(r2),
    }


def train_ablation_model(model, data, epochs=150, lr=0.001, weight_decay=1e-4, 
                          model_type='full', fixed_gate=None):
    """Train ablation model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    gate_history = []
    
    gene_mask = data.gene_mask
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, gate, _, _ = model(data.x, data.edge_index, gene_mask)
        
        # Loss on masked positions
        train_gene_mask = gene_mask[data.train_mask]
        pred_train = pred[data.train_mask][train_gene_mask]
        target_train = data.y_true[data.train_mask][train_gene_mask]
        loss = F.mse_loss(pred_train, target_train)
        
        # Auxiliary loss for learned gate models
        if model_type in ['full', 'no_quality']:
            gate_train = gate[data.train_mask]
            gate_mean = gate_train.mean()
            gate_std = gate_train.std().clamp(min=1e-6)
            aux_loss = -gate_std + 0.1 * (gate_mean - 0.5).abs()
            loss = loss + 0.01 * aux_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred, gate, _, _ = model(data.x, data.edge_index, gene_mask)
            val_gene_mask = gene_mask[data.val_mask]
            pred_val = pred[data.val_mask][val_gene_mask]
            target_val = data.y_true[data.val_mask][val_gene_mask]
            val_loss = F.mse_loss(pred_val, target_val).item()
            
            gate_stats = {
                'mean': gate[data.val_mask].mean().item(),
                'std': gate[data.val_mask].std().item(),
            }
            gate_history.append(gate_stats)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            extra = f", Gate: {gate_stats['mean']:.3f}+/-{gate_stats['std']:.3f}"
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}{extra}")
    
    model.load_state_dict(best_model_state)
    return model, best_val_loss, gate_history


def evaluate_ablation_model(model, data, mask_name='test'):
    """Evaluate ablation model."""
    model.eval()
    with torch.no_grad():
        pred, gate, _, _ = model(data.x, data.edge_index, data.gene_mask)
        
        if mask_name == 'test':
            node_mask = data.test_mask
        elif mask_name == 'val':
            node_mask = data.val_mask
        else:
            node_mask = data.train_mask
        
        metrics = evaluate_recovery(pred, data.y_true, data.gene_mask, node_mask,
                                     data.gene_mean, data.gene_std)
        
        gate_stats = {
            'mean': gate[node_mask].mean().item(),
            'std': gate[node_mask].std().item(),
            'min': gate[node_mask].min().item(),
            'max': gate[node_mask].max().item(),
            'spatial_dominant': (gate[node_mask] > 0.5).float().mean().item(),
        }
        
        return metrics, gate_stats


# ============================================================
# Ablation Experiment
# ============================================================

def run_ablation(n_genes, mask_rate, epochs=150):
    """Run ablation study for one condition."""
    print(f"\n{'='*70}")
    print(f"ABLATION: {n_genes} genes, {mask_rate*100:.0f}% mask")
    print(f"{'='*70}\n")
    
    adata, edges_df, splits, metadata = load_data()
    data = prepare_recovery_data(adata, edges_df, splits, mask_rate=mask_rate, n_top_genes=n_genes)
    input_dim = data.num_features
    
    results = {}
    
    # 1. Full Model
    print("\n1. Full Quality-Driven GNN (Ours)")
    model = FullQualityGNN(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='full')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['Full Model (Ours)'] = {'test': test_m, 'test_gate': test_gate, 'gate_history': gate_history}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f}")
    
    # 2. No Gate (Fixed 0.5)
    print("\n2. No Gate (Fixed 0.5 Fusion)")
    model = NoGateFixedFusion(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='fixed')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['No Gate (Fixed 0.5)'] = {'test': test_m, 'test_gate': test_gate}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f} (fixed)")
    
    # 3. No Quality Features
    print("\n3. No Quality Features")
    model = NoQualityFeatures(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='no_quality')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['No Quality Features'] = {'test': test_m, 'test_gate': test_gate, 'gate_history': gate_history}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f}")
    
    # 4. No Attention (Mean Agg)
    print("\n4. No Attention (Mean Aggregation)")
    model = NoAttentionMeanAgg(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='full')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['No Attention (Mean Agg)'] = {'test': test_m, 'test_gate': test_gate, 'gate_history': gate_history}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f}")
    
    # 5. Only Spatial Branch
    print("\n5. Only Spatial Branch (Gate=1)")
    model = OnlySpatialBranch(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='spatial_only')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['Only Spatial'] = {'test': test_m, 'test_gate': test_gate}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f} (fixed=1)")
    
    # 6. Only Non-Spatial Branch
    print("\n6. Only Non-Spatial Branch (Gate=0)")
    model = OnlyNonSpatialBranch(input_dim, hidden_dim=256)
    model, val_loss, gate_history = train_ablation_model(model, data, epochs=epochs, model_type='nospatial_only')
    test_m, test_gate = evaluate_ablation_model(model, data, 'test')
    results['Only Non-Spatial'] = {'test': test_m, 'test_gate': test_gate}
    print(f"  Test R2: {test_m['r2']:.4f}, Gate: {test_gate['mean']:.3f} (fixed=0)")
    
    return results


def main():
    all_ablation_results = {}
    
    # Focus on most informative conditions
    conditions = [
        (50, 0.9),   # Maximum spatial benefit
        (200, 0.7),  # Moderate spatial benefit
    ]
    
    for n_genes, mask_rate in conditions:
        results = run_ablation(n_genes, mask_rate, epochs=150)
        all_ablation_results[f"{n_genes}genes_{mask_rate}"] = results
    
    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for condition, results in all_ablation_results.items():
        print(f"\n--- {condition} ---")
        print(f"{'Model':30s} {'R2':>8s} {'Delta':>8s} {'Gate':>6s}")
        print("-" * 55)
        
        full_r2 = results['Full Model (Ours)']['test']['r2']
        
        for model_name, res in results.items():
            r2 = res['test']['r2']
            delta = r2 - full_r2
            gate = res['test_gate']['mean']
            print(f"{model_name:30s} {r2:>8.4f} {delta:>+8.4f} {gate:>6.3f}")
    
    # Save results
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.dim() == 0 else obj.tolist()
            return super().default(obj)
    
    results_path = EXP_DIR / 'ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_ablation_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")
    
    # ============================================================
    # Plot: Ablation comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.35
    
    for idx, (condition, results) in enumerate(all_ablation_results.items()):
        ax = axes[idx]
        r2_values = [results[m]['test']['r2'] for m in model_names]
        bars = ax.bar(x, r2_values, width, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'])
        ax.set_ylabel('R2 Score')
        ax.set_title(f'Ablation: {condition}')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=7, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Ablation comparison plot saved to {EXP_DIR / 'ablation_comparison.png'}")
    
    # ============================================================
    # Plot: Delta from full model
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions_list = list(all_ablation_results.keys())
    n_conditions = len(conditions_list)
    n_models = len(model_names)
    
    x = np.arange(n_models)
    width = 0.8 / n_conditions
    
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, (condition, results) in enumerate(all_ablation_results.items()):
        full_r2 = results['Full Model (Ours)']['test']['r2']
        deltas = [results[m]['test']['r2'] - full_r2 for m in model_names]
        offset = (idx - n_conditions/2 + 0.5) * width
        ax.bar(x + offset, deltas, width, label=condition, color=colors[idx], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('R2 Delta from Full Model')
    ax.set_title('Performance Drop When Removing Components')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'ablation_delta.png', dpi=150, bbox_inches='tight')
    print(f"Delta plot saved to {EXP_DIR / 'ablation_delta.png'}")


if __name__ == "__main__":
    main()
