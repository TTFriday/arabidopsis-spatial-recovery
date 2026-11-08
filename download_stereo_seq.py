#!/usr/bin/env python3
"""
Download Stereo-seq STDS0000104 Arabidopsis leaf spatial transcriptomics data.
Files: S1-1 through S1-7, S2-1, S2-3, S2-4 (10 h5ad files)
"""

import urllib.request
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000104/stomics"
FILES = [
    "S1-1_stereoseq", "S1-2_stereoseq", "S1-3_stereoseq", "S1-4_stereoseq", 
    "S1-5_stereoseq", "S1-6_stereoseq", "S1-7_stereoseq",
    "S2-1_stereoseq", "S2-3_stereoseq", "S2-4_stereoseq"
]

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_file(name):
    url = f"{BASE_URL}/{name}.h5ad"
    out_path = OUTPUT_DIR / f"{name}.h5ad"
    
    if out_path.exists():
        size = out_path.stat().st_size
        print(f"  [EXISTS] {name}.h5ad ({size} bytes)")
        return str(out_path), size
    
    print(f"  [DOWNLOAD] {name}.h5ad ...")
    req = urllib.request.Request(url, headers={
        "User-Agent": "arabidopsis-spatial-recovery/1.0 (research data download)"
    })
    
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    
    out_path.write_bytes(data)
    size = len(data)
    print(f"  [OK] {name}.h5ad ({size} bytes)")
    return str(out_path), size

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    results = []
    print(f"Downloading Stereo-seq STDS0000104 data to {OUTPUT_DIR}")
    print(f"Files: {len(FILES)} h5ad files")
    print()
    
    for name in FILES:
        path, size = download_file(name)
        sha = sha256_file(path)
        results.append({
            "name": name,
            "path": f"data/raw/{name}.h5ad",
            "bytes": size,
            "sha256": sha
        })
    
    # Save manifest
    manifest = {
        "dataset": "STDS0000104",
        "source_url": "https://db.cngb.org/stomics/datasets/STDS0000104/data",
        "ftp_base": BASE_URL,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": results,
        "total_bytes": sum(r["bytes"] for r in results),
        "total_files": len(results)
    }
    
    manifest_path = OUTPUT_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    
    print()
    print(f"Downloaded {len(results)} files, total {manifest['total_bytes']} bytes")
    print(f"Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
