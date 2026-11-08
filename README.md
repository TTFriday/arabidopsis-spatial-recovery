# Spatial context versus adaptive fusion for masked feature recovery

Code, processed data and run records for a leakage-controlled benchmark of
masked gene-expression recovery on spatial graphs, using Stereo-seq data from
*Arabidopsis thaliana* leaves.

## Description

Each tissue section is represented as a directed six-nearest-neighbour graph of
spots, and a fraction of the gene-expression entries is hidden at random. Models
reconstruct the hidden entries from the observed entries of the same spot and,
where applicable, from spatial neighbours. The benchmark asks whether spatial
context helps, whether a learned quality gate that adaptively fuses a node-only and
a spatial representation adds value beyond simpler models, and how much
target-informed preprocessing changes the results.

The benchmark covers 21 conditions (gene panels of 50, 100, 200, 500 and 3,000
genes; masking rates of 10-90%), ten seeds per condition, two preprocessing
protocols, eight methods and a seven-variant ablation of the gated model.

## Dataset information

- Source: China National GeneBank DataBase (CNGBdb), accession
  [STDS0000104](https://db.cngb.org/stomics/datasets/STDS0000104/data).
- Original study: Xia K, Sun H-X, Li J, et al. (2022). The single-cell
  Stereo-seq reveals region-specific cell subtypes and transcriptome profiling in
  *Arabidopsis* leaves. *Developmental Cell* 57(10):1299-1310.e4.
  https://doi.org/10.1016/j.devcel.2022.04.011
- Content used here: ten H5AD sections (S1-1 to S1-7, S2-1, S2-3, S2-4), 6,016
  spots, 19,191 genes (raw counts), spatial coordinates. Cell-type annotations are
  not used by any model.
- `data/processed/` (included):
  - `adata_processed.h5ad` - concatenated count matrix (outer join over genes).
  - `spatial_edges.csv` - directed 6-NN edges within sections (`source`,
    `target`, `distance`); 36,096 edges, no edges between sections.
  - `data_splits.json` - boolean spot masks: training sections S1-1 to S1-6
    (3,370 spots), validation S1-7 and S2-1 (1,172), test S2-3 and S2-4 (1,474).
  - `metadata.json` - sample identifiers and dimensions.
- `data/raw/download_manifest.json` lists the deposited files with sizes and
  SHA256 checksums. The raw H5AD files are not redistributed; the source data
  remain subject to the CNGBdb terms of use.

## Code information

| File | Purpose |
| --- | --- |
| `recovery_benchmark.py` | Runs the benchmark: data preparation for each protocol, masking, all models, training, evaluation; writes one JSON record per run. |
| `summarise_benchmark.py` | Paired statistics across seeds (means, 95% CIs, paired t-tests with Holm correction, Wilcoxon tests), protocol comparison, ablation and cost summaries; `--pack` merges run records into JSON-lines files. |
| `preprocess_data.py` | Builds `data/processed/` from the deposited H5AD files (concatenation, 6-NN graphs, splits). |
| `download_stereo_seq.py` | Downloads the ten deposited H5AD files from CNGBdb. |
| `inspect_data.py` | Prints the structure of the deposited files. |

Methods implemented in `recovery_benchmark.py`: mean fill, neighbour average,
feature propagation, a node-only denoising autoencoder (AE), neighbour-mean plus
MLP (N+MLP), GCN, GAT and the quality-gated dual-branch model (Gated), plus seven
ablation variants of the gated model.

`expression_recovery.py`, `ablation_study.py`, `rerun_3000.py`, `experiments/*.json`
and `results/` belong to an initial single-seed analysis that used target-informed
preprocessing; they are kept for provenance and are superseded by
`recovery_benchmark.py`.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. The reported runs
used Python 3.10.9, PyTorch 2.6.0 (CUDA 12.4), PyTorch Geometric 2.8.1, NumPy
2.2.6, SciPy 1.15.3, pandas 2.3.3 and h5py 3.16.0 on Windows 10 with an NVIDIA
GeForce RTX 3090. A GPU is optional; `--cpu` forces CPU execution.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch Geometric may need a wheel that matches your PyTorch/CUDA build; see the
[installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage instructions

Recompute the summary statistics from the included run records (no training):

```bash
python summarise_benchmark.py
```

Rerun the benchmark (about 15-20 s per condition and seed on an RTX 3090):

```bash
python recovery_benchmark.py main --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py main --protocol leaky --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py propagation --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py propagation --protocol leaky --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py ablation --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9
python summarise_benchmark.py
```

Runs write to `experiments/benchmark/<kind>/<protocol>/`; existing records are
skipped, so an interrupted grid can be resumed. If these folders are absent,
`summarise_benchmark.py` reads the packed files in `experiments/benchmark/records/`.
Summaries are written to `experiments/benchmark/summary/`.

To load a processed matrix without anndata:

```python
import h5py, scipy.sparse as sp
with h5py.File("data/processed/adata_processed.h5ad") as f:
    X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                      shape=tuple(f["X"].attrs["shape"]))
```

To rebuild `data/processed/` from the deposited files:

```bash
python download_stereo_seq.py
python preprocess_data.py
```

## Methodology

1. **Splits.** Whole sections are assigned to training, validation and test, and
   the graph has no edges between sections.
2. **Preprocessing.** In the leakage-controlled protocol (`clean`), the top-G
   genes are ranked by raw-count variance over training spots, counts are
   log1p-transformed and standardised with training-spot means and standard
   deviations. The `leaky` protocol computes the same quantities over all spots,
   including validation and test spots, and is kept only for comparison.
3. **Masking.** Entries are hidden independently with probability p, using a
   generator seeded by (seed, G, p); hidden entries are set to 0 (the training mean)
   in standardised space. All methods and both protocols share the mask of a given
   seed.
4. **Training.** Masked MSE on training spots; Adam (lr 1e-3, weight decay 1e-4),
   150 full-batch epochs, ReduceLROnPlateau; the checkpoint with the lowest masked
   validation MSE is restored. The gated model adds a small gate-spread penalty.
5. **Evaluation.** MSE, MAE, pooled Pearson correlation and R2 on the log1p scale
   over hidden test entries. Differences between methods are paired by seed.

## Citations

If you use this code, please cite the associated article (citation to be added
on publication) and the data source:

- Xia K, et al. (2022). *Developmental Cell* 57(10):1299-1310.e4.
  https://doi.org/10.1016/j.devcel.2022.04.011

Feature propagation follows Rossi E, et al. (2022), *Proceedings of the First
Learning on Graphs Conference*, PMLR 198:11:1-11:16.

## License and contribution guidelines

The code is released under the MIT License (see `LICENSE`). The license does not
cover the third-party Stereo-seq data. Bug reports and suggestions are welcome
through GitHub issues; please include the command, the environment and the error
message.
