#!/usr/bin/env python3
"""Repeated, leakage-controlled benchmark for masked expression recovery.

Each run fits one model on one (protocol, panel size, masking rate, seed)
combination and writes a JSON record. Existing records are skipped, so an
interrupted grid can be resumed with the same command.

Protocols
---------
clean  Gene selection and z-scoring use training spots only; validation and
       test spots never enter preprocessing.
leaky  Gene selection and z-scoring use all spots (the original protocol),
       kept to measure the effect of target-informed preprocessing.

Usage
-----
python recovery_benchmark.py main --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py main --protocol leaky --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py propagation --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py propagation --protocol leaky --seeds 0 1 2 3 4 5 6 7 8 9
python recovery_benchmark.py ablation --protocol clean --seeds 0 1 2 3 4 5 6 7 8 9

'main' trains the learned models and evaluates mean fill and neighbour averaging;
'propagation' evaluates feature propagation; 'ablation' trains the seven
variants of the gated model on six conditions.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, issparse
from scipy.stats import pearsonr
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import softmax as segment_softmax

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "experiments" / "benchmark"

PANELS = [50, 100, 200, 500, 3000]
HIDDEN = 256
EPOCHS = 150
ABLATION_CONDITIONS = [(50, 0.9), (50, 0.5), (100, 0.7), (200, 0.7), (500, 0.3), (3000, 0.3)]


def mask_rates(n_genes):
    return [0.1, 0.3, 0.5, 0.7, 0.9] if n_genes <= 200 else [0.1, 0.3, 0.5]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def read_counts(path):
    """Read the CSR count matrix of an H5AD file without importing anndata."""
    with h5py.File(path, "r") as f:
        group = f["X"]
        shape = tuple(int(v) for v in group.attrs["shape"])
        matrix = csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape)
    return matrix.tocsc()


class Dataset:
    """Raw counts, graph and sample-level splits, loaded once."""

    def __init__(self, device):
        self.X = read_counts(PROC_DIR / "adata_processed.h5ad")
        edges = pd.read_csv(PROC_DIR / "spatial_edges.csv")
        self.edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long, device=device)
        splits = json.loads((PROC_DIR / "data_splits.json").read_text())
        self.train = torch.tensor(splits["train"], dtype=torch.bool, device=device)
        self.val = torch.tensor(splits["val"], dtype=torch.bool, device=device)
        self.test = torch.tensor(splits["test"], dtype=torch.bool, device=device)
        self.n_spots = self.X.shape[0]
        self.device = device

    def gene_variance(self, rows):
        sub = self.X[rows]
        if issparse(sub):
            mean = np.asarray(sub.mean(axis=0)).ravel()
            mean_sq = np.asarray(sub.power(2).mean(axis=0)).ravel()
            return mean_sq - mean ** 2
        return sub.var(axis=0)

    def condition(self, protocol, n_genes, mask_rate, seed):
        """Return standardised targets, masked input and the artificial mask."""
        train_rows = self.train.cpu().numpy()
        fit_rows = train_rows if protocol == "clean" else np.ones(self.n_spots, dtype=bool)
        var = self.gene_variance(fit_rows)
        genes = np.sort(np.argsort(var)[::-1][:n_genes])
        dense = self.X[:, genes]
        dense = dense.toarray() if issparse(dense) else np.asarray(dense)
        Y = torch.log1p(torch.tensor(dense, dtype=torch.float32, device=self.device))
        fit = torch.tensor(fit_rows, device=self.device)
        mu = Y[fit].mean(0, keepdim=True)
        sd = Y[fit].std(0, keepdim=True).clamp(min=1e-6)
        Xn = (Y - mu) / sd
        gen = torch.Generator().manual_seed(1_000_003 * seed + 1_009 * n_genes + int(round(mask_rate * 100)))
        mask = (torch.rand(Xn.shape, generator=gen) < mask_rate).to(self.device)
        x_in = Xn.masked_fill(mask, 0.0)
        return {"x": x_in, "y": Xn, "mask": mask, "mu": mu, "sd": sd, "genes": genes.tolist()}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def mlp_encoder(d_in, h):
    return nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU())


def mlp_decoder(h, d_out):
    return nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, d_out))


def outgoing_mean(values, edge_index, n):
    src, dst = edge_index
    total = torch.zeros(n, values.size(1), device=values.device).index_add_(0, src, values[dst])
    count = torch.zeros(n, 1, device=values.device).index_add_(0, src, torch.ones(src.numel(), 1, device=values.device))
    return total / count.clamp(min=1)


def incoming_mean(values, edge_index, n):
    src, dst = edge_index
    total = torch.zeros(n, values.size(1), device=values.device).index_add_(0, dst, values[src])
    count = torch.zeros(n, 1, device=values.device).index_add_(0, dst, torch.ones(src.numel(), 1, device=values.device))
    return total / count.clamp(min=1)


class AutoEncoder(nn.Module):
    def __init__(self, g, h=HIDDEN):
        super().__init__()
        self.enc, self.dec = mlp_encoder(g, h), mlp_decoder(h, g)

    def forward(self, x, edge_index, mask):
        return self.dec(self.enc(x)), None


class NeighbourMLP(nn.Module):
    """Concatenate the spot with the mean of its six selected neighbours."""

    def __init__(self, g, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * g, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, g))

    def forward(self, x, edge_index, mask):
        return self.net(torch.cat([x, outgoing_mean(x, edge_index, x.size(0))], 1)), None


class GCNImputer(nn.Module):
    def __init__(self, g, h=HIDDEN):
        super().__init__()
        self.c1, self.c2 = GCNConv(g, h), GCNConv(h, h)
        self.dec = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, g))

    def forward(self, x, edge_index, mask):
        z = F.dropout(F.relu(self.c1(x, edge_index)), 0.2, self.training)
        return self.dec(F.relu(self.c2(z, edge_index))), None


class GATImputer(nn.Module):
    def __init__(self, g, h=HIDDEN, heads=4):
        super().__init__()
        self.c1, self.c2 = GATConv(g, h // heads, heads=heads), GATConv(h, h // heads, heads=heads)
        self.dec = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, g))

    def forward(self, x, edge_index, mask):
        z = F.dropout(F.elu(self.c1(x, edge_index)), 0.2, self.training)
        return self.dec(F.elu(self.c2(z, edge_index))), None


class GatedFusion(nn.Module):
    """Dual-branch model with a spot-level quality gate.

    aggregation: 'attention' (segmented softmax over incoming edges) or 'mean'
    (unweighted incoming mean). gate: 'learned', 'fixed' (0.5), 'spatial' (1)
    or 'nonspatial' (0). quality: include the four quality statistics.
    """

    def __init__(self, g, h=HIDDEN, aggregation="attention", gate="learned", quality=True):
        super().__init__()
        self.aggregation, self.gate_mode, self.quality = aggregation, gate, quality
        self.enc_ns, self.enc_sp = mlp_encoder(g, h), mlp_encoder(g, h)
        if aggregation == "attention":
            self.att = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(), nn.Linear(h, 1))
        if gate == "learned":
            d = 2 * h + 1 + (4 if quality else 0)
            self.gate = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(h, 1), nn.Sigmoid())
        self.dec = mlp_decoder(h, g)

    def forward(self, x, edge_index, mask):
        n = x.size(0)
        src, dst = edge_index
        h_ns, h_enc = self.enc_ns(x), self.enc_sp(x)
        if self.aggregation == "attention":
            score = self.att(torch.cat([h_enc[src], h_enc[dst]], 1)).squeeze(1)
            alpha = segment_softmax(score, dst, num_nodes=n)
            h_sp = torch.zeros_like(h_enc).index_add_(0, dst, alpha.unsqueeze(1) * h_enc[src])
        else:
            h_sp = incoming_mean(h_enc, edge_index, n)
        if self.gate_mode == "learned":
            cos = F.cosine_similarity(h_ns, h_sp, dim=1).unsqueeze(1)
            parts = [h_ns, h_sp, cos]
            if self.quality:
                r_self = mask.float().mean(1, keepdim=True)
                r_out = outgoing_mean(r_self, edge_index, n)
                m1 = outgoing_mean(x, edge_index, n)
                m2 = outgoing_mean(x * x, edge_index, n)
                v = (m2 - m1 * m1).clamp(min=0).mean(1, keepdim=True)
                d = F.pairwise_distance(h_ns, h_sp).unsqueeze(1)
                parts += [r_self, r_out, v, d]
            g = self.gate(torch.cat(parts, 1))
        else:
            value = {"fixed": 0.5, "spatial": 1.0, "nonspatial": 0.0}[self.gate_mode]
            g = torch.full((n, 1), value, device=x.device)
        return self.dec(g * h_sp + (1 - g) * h_ns), g


MAIN_MODELS = {
    "AE": lambda g: AutoEncoder(g),
    "N+MLP": lambda g: NeighbourMLP(g),
    "GCN": lambda g: GCNImputer(g),
    "GAT": lambda g: GATImputer(g),
    "Gated": lambda g: GatedFusion(g),
}

ABLATION_MODELS = {
    "Full": (lambda g: GatedFusion(g), True),
    "No gate penalty": (lambda g: GatedFusion(g), False),
    "Fixed fusion (0.5)": (lambda g: GatedFusion(g, gate="fixed"), False),
    "No quality features": (lambda g: GatedFusion(g, quality=False), True),
    "Incoming mean (no attention)": (lambda g: GatedFusion(g, aggregation="mean"), True),
    "Spatial branch only": (lambda g: GatedFusion(g, gate="spatial"), False),
    "Non-spatial branch only": (lambda g: GatedFusion(g, gate="nonspatial"), False),
}


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------

def metrics(pred, cond, rows):
    sel = cond["mask"] & rows.unsqueeze(1)
    genes = sel.nonzero(as_tuple=True)[1]
    scale, shift = cond["sd"][0, genes], cond["mu"][0, genes]
    p = (pred[sel] * scale + shift).double()
    t = (cond["y"][sel] * scale + shift).double()
    err = p - t
    ss_tot = ((t - t.mean()) ** 2).sum()
    pn, tn = p.cpu().numpy(), t.cpu().numpy()
    return {"mse": err.pow(2).mean().item(), "mae": err.abs().mean().item(),
            "pearson": float(pearsonr(pn, tn)[0]), "r2": (1 - err.pow(2).sum() / ss_tot).item(),
            "n_entries": int(sel.sum().item())}


def gate_summary(g, rows):
    if g is None:
        return None
    v = g[rows].squeeze(1)
    return {"mean": v.mean().item(), "sd": v.std().item(), "frac_gt_0.5": (v > 0.5).float().mean().item()}


def fit(model, cond, data, gate_penalty, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    x, y, mask, ei = cond["x"], cond["y"], cond["mask"], data.edge_index
    tr = mask & data.train.unsqueeze(1)
    va = mask & data.val.unsqueeze(1)
    best, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred, g = model(x, ei, mask)
        loss = F.mse_loss(pred[tr], y[tr])
        if gate_penalty and g is not None:
            gt = g[data.train]
            loss = loss + 0.01 * (-gt.std().clamp(min=1e-6) + 0.1 * (gt.mean() - 0.5).abs())
        loss.backward()
        opt.step()
        sched.step(loss.item())
        model.eval()
        with torch.no_grad():
            pred, _ = model(x, ei, mask)
            val = F.mse_loss(pred[va], y[va]).item()
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, best_epoch


def evaluate(model, cond, data):
    model.eval()
    with torch.no_grad():
        pred, g = model(cond["x"], data.edge_index, cond["mask"])
    return {"val": metrics(pred, cond, data.val), "test": metrics(pred, cond, data.test),
            "gate_test": gate_summary(g, data.test), "gate_val": gate_summary(g, data.val)}


def non_learned(cond, data):
    """Training-mean fill and observed-only neighbour averaging."""
    zero = torch.zeros_like(cond["y"])
    observed = (~cond["mask"]).float()
    num = outgoing_mean(cond["x"] * observed, data.edge_index, data.n_spots)
    den = outgoing_mean(observed, data.edge_index, data.n_spots)
    nbr = torch.where(den > 0, num / den.clamp(min=1e-12), zero)
    out = {}
    for name, pred in [("Mean fill", zero), ("Neighbour average", nbr)]:
        out[name] = {"val": metrics(pred, cond, data.val), "test": metrics(pred, cond, data.test),
                     "gate_test": None, "gate_val": None, "parameters": 0, "fit_seconds": 0.0,
                     "best_epoch": None, "best_val_loss_std": None}
    return out


def feature_propagation(cond, data, iterations=40):
    """Feature propagation on the symmetrised graph, observed entries reset each step."""
    n = data.n_spots
    src, dst = data.edge_index
    both = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)
    both = torch.unique(both, dim=1)
    deg = torch.zeros(n, device=both.device).index_add_(0, both[0], torch.ones(both.size(1), device=both.device))
    weight = deg[both[0]].clamp(min=1).rsqrt() * deg[both[1]].clamp(min=1).rsqrt()
    observed = ~cond["mask"]
    x = cond["x"].clone()
    for _ in range(iterations):
        x = torch.zeros_like(x).index_add_(0, both[0], weight.unsqueeze(1) * x[both[1]])
        x = torch.where(observed, cond["x"], x)
    return x


def run_propagation(protocol, n_genes, rate, seed, data):
    path = record_path("propagation", protocol, n_genes, rate, seed)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cond = data.condition(protocol, n_genes, rate, seed)
    pred = feature_propagation(cond, data)
    res = {"val": metrics(pred, cond, data.val), "test": metrics(pred, cond, data.test),
           "gate_test": None, "gate_val": None, "parameters": 0, "fit_seconds": 0.0,
           "best_epoch": None, "best_val_loss_std": None}
    record = {"kind": "propagation", "protocol": protocol, "n_genes": n_genes, "mask_rate": rate,
              "seed": seed, "genes": cond["genes"], "results": {"Feature propagation": res}}
    path.write_text(json.dumps(record, indent=1))


def record_path(kind, protocol, n_genes, rate, seed):
    return OUT_DIR / kind / protocol / f"g{n_genes}_m{int(round(rate * 100)):02d}_s{seed}.json"


def run_condition(kind, protocol, n_genes, rate, seed, data):
    path = record_path(kind, protocol, n_genes, rate, seed)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cond = data.condition(protocol, n_genes, rate, seed)
    results = non_learned(cond, data) if kind == "main" else {}
    specs = ({k: (v, k == "Gated") for k, v in MAIN_MODELS.items()} if kind == "main" else ABLATION_MODELS)
    for index, (name, (factory, penalty)) in enumerate(specs.items()):
        torch.manual_seed(10_007 * seed + 101 * index + n_genes)
        model = factory(n_genes).to(data.device)
        if data.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        best, best_epoch = fit(model, cond, data, penalty)
        if data.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        res = evaluate(model, cond, data)
        res.update({"parameters": sum(p.numel() for p in model.parameters()), "fit_seconds": elapsed,
                    "best_epoch": best_epoch, "best_val_loss_std": best})
        results[name] = res
    record = {"kind": kind, "protocol": protocol, "n_genes": n_genes, "mask_rate": rate, "seed": seed,
              "epochs": EPOCHS, "hidden": HIDDEN, "genes": cond["genes"], "results": results}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    best_name = max((k for k in results), key=lambda k: results[k]["test"]["r2"])
    print(f"{kind}/{protocol} G={n_genes} p={rate} seed={seed}: best={best_name} "
          f"R2={results[best_name]['test']['r2']:.4f}", flush=True)


def environment():
    import scipy, sklearn, torch_geometric
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "h5py": h5py.__version__,
            "pandas": pd.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["main", "ablation", "propagation"])
    parser.add_argument("--protocol", choices=["clean", "leaky"], default="clean")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--panels", type=int, nargs="+", default=PANELS)
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "environment.json").write_text(json.dumps(environment(), indent=1))
    data = Dataset(device)
    if args.kind == "ablation":
        grid = [c for c in ABLATION_CONDITIONS if c[0] in args.panels]
    else:
        grid = [(g, p) for g in args.panels for p in mask_rates(g)]
    for seed in args.seeds:
        for n_genes, rate in grid:
            if args.kind == "propagation":
                run_propagation(args.protocol, n_genes, rate, seed, data)
            else:
                run_condition(args.kind, args.protocol, n_genes, rate, seed, data)


if __name__ == "__main__":
    main()
