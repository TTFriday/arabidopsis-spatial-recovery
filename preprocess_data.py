#!/usr/bin/env python3
"""
Preprocess Stereo-seq data for spatial GNN experiments.
- Load all samples
- Build spatial graphs
- Prepare train/val/test splits
- Visualize spatial structure
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import networkx as nx

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_all_samples():
    """Load all h5ad files and concatenate."""
    h5ad_files = sorted(DATA_DIR.glob("*.h5ad"))
    print(f"Loading {len(h5ad_files)} samples...")
    
    adatas = []
    sample_ids = []
    for i, path in enumerate(h5ad_files):
        adata = ad.read_h5ad(path)
        adata.obs['sample_id'] = path.stem
        adata.obs['sample_idx'] = i
        adatas.append(adata)
        sample_ids.append(path.stem)
        print(f"  {path.stem}: {adata.n_obs} spots, {adata.n_vars} genes")
    
    # Concatenate all samples
    adata_all = ad.concat(adatas, join='outer', merge='same')
    print(f"\nTotal: {adata_all.n_obs} spots, {adata_all.n_vars} genes")
    
    return adata_all, sample_ids

def build_spatial_graph(adata, k=6, sample_key='sample_id'):
    """Build spatial k-NN graph for each sample separately."""
    print(f"\nBuilding spatial k-NN graph (k={k})...")
    
    samples = adata.obs[sample_key].unique()
    all_edges = []
    
    for sample in samples:
        mask = adata.obs[sample_key] == sample
        coords = adata.obsm['spatial'][mask]
        
        # Build k-NN graph for this sample
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Convert to edge list (exclude self-loops)
        n_spots = coords.shape[0]
        for i in range(n_spots):
            for j_idx, j_local in enumerate(indices[i]):
                if j_idx == 0:  # Skip self
                    continue
                # Global indices
                i_global = np.where(mask)[0][i]
                j_global = np.where(mask)[0][j_local]
                dist = distances[i, j_idx]
                all_edges.append((i_global, j_global, dist))
    
    print(f"  Total edges: {len(all_edges)}")
    return all_edges

def visualize_spatial_structure(adata, sample_ids):
    """Visualize spatial coordinates and cell types for each sample."""
    print("\nGenerating spatial visualizations...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, sample in enumerate(sample_ids):
        if idx >= len(axes):
            break
        
        mask = adata.obs['sample_id'] == sample
        coords = adata.obsm['spatial'][mask]
        cell_types = adata.obs['cell_type'][mask]
        
        ax = axes[idx]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cell_types.astype('category').cat.codes, 
                           cmap='tab20', s=10, alpha=0.7)
        ax.set_title(f"{sample}\n({mask.sum()} spots)", fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(PROC_DIR / 'spatial_structure.png', dpi=150, bbox_inches='tight')
    print(f"  Saved to {PROC_DIR / 'spatial_structure.png'}")
    plt.close()

def check_expression_type(adata):
    """Check if expression is raw counts or normalized."""
    print("\nChecking expression matrix...")
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X_sample = X[:1000].toarray()
    else:
        X_sample = X[:1000]
    
    # Check if integer values (raw counts)
    is_integer = np.all(X_sample == X_sample.astype(int))
    
    # Check range
    x_min, x_max, x_mean = X_sample.min(), X_sample.max(), X_sample.mean()
    
    print(f"  Min: {x_min:.2f}, Max: {x_max:.2f}, Mean: {x_mean:.2f}")
    print(f"  Integer values: {is_integer}")
    
    if is_integer:
        print("  → Appears to be RAW COUNTS")
        return 'raw'
    elif x_max < 100:
        print("  → Appears to be NORMALIZED (log or CPM)")
        return 'normalized'
    else:
        print("  → Unknown type, treating as normalized")
        return 'unknown'

def prepare_splits(adata, test_samples=None):
    """Prepare train/val/test splits by sample."""
    print("\nPreparing data splits...")
    
    samples = adata.obs['sample_id'].unique()
    
    if test_samples is None:
        # Use last 2 samples as test
        test_samples = samples[-2:]
        val_samples = samples[-4:-2]
        train_samples = samples[:-4]
    else:
        train_samples = [s for s in samples if s not in test_samples]
        val_samples = train_samples[-2:]
        train_samples = train_samples[:-2]
    
    print(f"  Train samples: {len(train_samples)} ({', '.join(train_samples)})")
    print(f"  Val samples: {len(val_samples)} ({', '.join(val_samples)})")
    print(f"  Test samples: {len(test_samples)} ({', '.join(test_samples)})")
    
    # Create masks
    train_mask = adata.obs['sample_id'].isin(train_samples).values
    val_mask = adata.obs['sample_id'].isin(val_samples).values
    test_mask = adata.obs['sample_id'].isin(test_samples).values
    
    print(f"  Train spots: {train_mask.sum()}")
    print(f"  Val spots: {val_mask.sum()}")
    print(f"  Test spots: {test_mask.sum()}")
    
    return train_mask, val_mask, test_mask

def main():
    # Load data
    adata, sample_ids = load_all_samples()
    
    # Check expression type
    expr_type = check_expression_type(adata)
    
    # Visualize spatial structure
    visualize_spatial_structure(adata, sample_ids)
    
    # Build spatial graph
    edges = build_spatial_graph(adata, k=6)
    
    # Prepare splits
    train_mask, val_mask, test_mask = prepare_splits(adata)
    
    # Save processed data
    print("\nSaving processed data...")
    
    # Save AnnData
    adata.write(PROC_DIR / 'adata_processed.h5ad')
    
    # Save edges
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'distance'])
    edges_df.to_csv(PROC_DIR / 'spatial_edges.csv', index=False)
    
    # Save masks
    masks = {
        'train': train_mask.tolist(),
        'val': val_mask.tolist(),
        'test': test_mask.tolist()
    }
    with open(PROC_DIR / 'data_splits.json', 'w') as f:
        json.dump(masks, f, indent=2)
    
    # Save metadata
    metadata = {
        'n_samples': len(sample_ids),
        'sample_ids': sample_ids,
        'n_spots': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'n_edges': len(edges),
        'expression_type': expr_type,
        'cell_types': list(adata.obs['cell_type'].unique()),
        'n_cell_types': int(adata.obs['cell_type'].nunique())
    }
    with open(PROC_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Processed data saved to {PROC_DIR}")
    print(f"  - adata_processed.h5ad")
    print(f"  - spatial_edges.csv ({len(edges)} edges)")
    print(f"  - data_splits.json")
    print(f"  - metadata.json")
    print(f"\nDataset summary:")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Total spots: {metadata['n_spots']}")
    print(f"  Genes: {metadata['n_genes']}")
    print(f"  Cell types: {metadata['n_cell_types']}")

if __name__ == "__main__":
    main()
