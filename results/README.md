# Derived result tables

These files are derived from the archived JSON in `experiments/` and are kept
separately for convenience. The analysis code writes raw results to
`experiments/`; these tables summarize them for inspection.

- `main_metrics.csv` - one row per method and condition in the 21-condition grid,
  with validation and test metrics.
- `ablation_metrics.csv` - metric vectors for the six ablation variants in the two
  focus conditions.
- `baseline_selection.csv` - per-condition validation-based selection between the
  autoencoder and neighbour-mean baseline, with the selected model's test metrics.
- `rerun_metrics.csv` and `rerun_differences.csv` - comparisons for the 3,000-gene
  same-seed rerun.
- `model_capacity.csv` - parameter counts at the hidden width used in the archived
  runs.
- `sample_manifest.csv` and `data_audit.json` - sample, graph, and split checks.
- `summary.json` - descriptive averages, ranks, winning margins, and repeatability
  summaries.

The metric definitions and the caveats that apply to these summaries are described
in the associated manuscript and its supplementary notes.
