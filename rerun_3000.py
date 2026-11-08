#!/usr/bin/env python3
"""Re-run the 3,000-gene conditions with the current pipeline.

The archived results file reused 3,000-gene conditions from an earlier
execution.  This script regenerates them so that every condition in the
manuscript can be traced to one version of the code.  Outputs are written to
a separate file and are not merged automatically.
"""

import json
from pathlib import Path

import numpy as np

import expression_recovery as er

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "recovery_results_3000_rerun.json"


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    if OUT.exists():
        with open(OUT, encoding="utf-8") as fh:
            all_results = json.load(fh)
    else:
        all_results = {}
    for mask_rate in (0.1, 0.3, 0.5):
        key = f"3000genes_{mask_rate}"
        if key in all_results:
            print(f"\n=== 3000 genes / {mask_rate:.1f} mask: already done, skipping ===", flush=True)
            continue
        print(f"\n=== 3000 genes / {mask_rate:.1f} mask ===", flush=True)
        results, _, _ = er.run_single_mask_rate(
            mask_rate, epochs=150, n_top_genes=3000
        )
        all_results[key] = results
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=1, cls=Encoder)
        print(f"updated {OUT}", flush=True)


if __name__ == "__main__":
    main()
