#!/usr/bin/env python3
"""
Inspect Stereo-seq STDS0000104 h5ad files to understand data structure.
"""

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"

def inspect_h5ad(path):
    """Inspect a single h5ad file."""
    adata = ad.read_h5ad(path)
    
    info = {
        "file": path.name,
        "n_obs": adata.n_obs,  # number of spots/cells
        "n_vars": adata.n_vars,  # number of genes
        "obs_keys": list(adata.obs.keys()),
        "var_keys": list(adata.var.keys()),
        "obsm_keys": list(adata.obsm.keys()),
        "layers_keys": list(adata.layers.keys()),
        "uns_keys": list(adata.uns.keys()),
    }
    
    # Check for spatial coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        info["spatial_shape"] = list(coords.shape)
        info["spatial_range_x"] = [float(coords[:, 0].min()), float(coords[:, 0].max())]
        info["spatial_range_y"] = [float(coords[:, 1].min()), float(coords[:, 1].max())]
    
    # Check expression matrix
    if "X" in adata.__dict__:
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        info["X_dtype"] = str(X.dtype)
        info["X_min"] = float(X.min())
        info["X_max"] = float(X.max())
        info["X_mean"] = float(X.mean())
        info["X_nonzero_frac"] = float((X != 0).mean())
    
    # Check for raw counts
    if "raw" in adata.__dict__ and adata.raw is not None:
        info["has_raw"] = True
        info["raw_n_vars"] = adata.raw.n_vars
    
    return info, adata

def main():
    h5ad_files = sorted(DATA_DIR.glob("*.h5ad"))
    print(f"Found {len(h5ad_files)} h5ad files\n")
    
    all_info = []
    for path in h5ad_files:
        print(f"Inspecting {path.name}...")
        info, adata = inspect_h5ad(path)
        all_info.append(info)
        
        print(f"  Spots: {info['n_obs']}, Genes: {info['n_vars']}")
        if "spatial_shape" in info:
            print(f"  Spatial coords: {info['spatial_shape']}")
            print(f"  X range: [{info['spatial_range_x'][0]:.1f}, {info['spatial_range_x'][1]:.1f}]")
            print(f"  Y range: [{info['spatial_range_y'][0]:.1f}, {info['spatial_range_y'][1]:.1f}]")
        print(f"  Expression dtype: {info.get('X_dtype', 'N/A')}")
        print(f"  Non-zero fraction: {info.get('X_nonzero_frac', 'N/A')}")
        print(f"  obs keys: {info['obs_keys']}")
        print(f"  var keys: {info['var_keys']}")
        print(f"  obsm keys: {info['obsm_keys']}")
        print(f"  layers keys: {info['layers_keys']}")
        print()
    
    # Save summary
    summary_path = DATA_DIR.parent / "data_inspection.json"
    summary_path.write_text(json.dumps(all_info, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")
    
    # Check if samples are independent
    print("\n=== Sample Independence Check ===")
    spot_counts = [info["n_obs"] for info in all_info]
    print(f"Spot counts per sample: {spot_counts}")
    print(f"Total spots: {sum(spot_counts)}")
    
    # Check gene overlap
    print("\n=== Gene Overlap Check ===")
    gene_sets = []
    for path in h5ad_files:
        adata = ad.read_h5ad(path)
        genes = set(adata.var_names)
        gene_sets.append(genes)
    
    # Common genes across all samples
    common_genes = set.intersection(*gene_sets)
    print(f"Common genes across all {len(gene_sets)} samples: {len(common_genes)}")
    
    # Total unique genes
    all_genes = set.union(*gene_sets)
    print(f"Total unique genes: {len(all_genes)}")

if __name__ == "__main__":
    main()
