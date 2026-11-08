#!/usr/bin/env python3
"""Summarise the repeated benchmark records written by recovery_benchmark.py.

Writes CSV/JSON summaries to experiments/benchmark/summary/. Statistics are
paired across seeds (identical masks, splits and gene panels within a seed);
seeds are technical replicates of masking and initialisation, not new tissue
samples.

python summarise_benchmark.py          # compute summaries
python summarise_benchmark.py --pack   # merge per-run JSON files into records/*.jsonl

Per-run folders (experiments/benchmark/<kind>/<protocol>/) are used when present;
otherwise the packed files in experiments/benchmark/records/ are read.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "experiments" / "benchmark"
OUT = BENCH / "summary"
METRICS = ["mse", "mae", "pearson", "r2"]
LEARNED = ["AE", "N+MLP", "GCN", "GAT", "Gated"]
ALL_METHODS = ["Mean fill", "Neighbour average", "Feature propagation"] + LEARNED
CONTRASTS = [("Gated", "AE"), ("Gated", "N+MLP"), ("N+MLP", "AE"), ("Gated", "Selector"),
             ("GCN", "AE"), ("GAT", "AE")]


def iter_records(kind, protocol):
    """Yield run records from per-run JSON files or, if absent, from packed JSON lines."""
    folder = BENCH / kind / protocol
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            yield json.loads(path.read_text())
        return
    packed = BENCH / "records" / f"{kind}_{protocol}.jsonl"
    if packed.exists():
        with packed.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def pack():
    """Merge per-run JSON files into one JSON-lines file per kind and protocol."""
    out = BENCH / "records"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("main", "propagation", "ablation"):
        for protocol in ("clean", "leaky"):
            folder = BENCH / kind / protocol
            if not folder.exists():
                continue
            lines = [json.dumps(json.loads(p.read_text()), separators=(",", ":"))
                     for p in sorted(folder.glob("*.json"))]
            (out / f"{kind}_{protocol}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"packed {len(lines)} records -> records/{kind}_{protocol}.jsonl")


def load(kind, protocol):
    rows = []
    for rec in iter_records(kind, protocol):
        for method, res in rec["results"].items():
            row = {"n_genes": rec["n_genes"], "mask_rate": rec["mask_rate"], "seed": rec["seed"],
                   "method": method, "parameters": res["parameters"], "fit_seconds": res["fit_seconds"],
                   "best_epoch": res["best_epoch"]}
            for split in ("val", "test"):
                for m in METRICS:
                    row[f"{split}_{m}"] = res[split][m]
            gate = res.get("gate_test")
            row["gate_mean"] = gate["mean"] if gate else np.nan
            row["gate_sd"] = gate["sd"] if gate else np.nan
            row["gate_frac"] = gate["frac_gt_0.5"] if gate else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def add_selector(df):
    """Per condition and seed, choose AE or N+MLP by validation MSE (ties -> AE)."""
    keys = ["n_genes", "mask_rate", "seed"]
    pair = df[df.method.isin(["AE", "N+MLP"])].pivot_table(index=keys, columns="method", values="val_mse")
    choice = np.where(pair["N+MLP"] < pair["AE"], "N+MLP", "AE")
    chosen = pd.DataFrame({"choice": choice}, index=pair.index).reset_index()
    sel = df.merge(chosen, on=keys)
    sel = sel[sel.method == sel.choice].copy()
    sel["method"] = "Selector"
    return pd.concat([df, sel.drop(columns="choice")], ignore_index=True), chosen


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired(diff):
    diff = np.asarray(diff, float)
    n = diff.size
    mean, sd = diff.mean(), diff.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.ttest_1samp(diff, 0.0).pvalue if sd > 0 else (0.0 if mean != 0 else 1.0)
    return mean, sd, mean - half, mean + half, p


def contrast_table(df, metric="test_r2"):
    wide = df.pivot_table(index=["n_genes", "mask_rate", "seed"], columns="method", values=metric)
    rows, overall = [], []
    for a, b in CONTRASTS:
        diff = (wide[a] - wide[b]).rename("diff").reset_index()
        block = []
        for (g, p), grp in diff.groupby(["n_genes", "mask_rate"]):
            mean, sd, lo, hi, pv = paired(grp["diff"])
            block.append({"contrast": f"{a} - {b}", "n_genes": g, "mask_rate": p, "mean": mean, "sd": sd,
                          "ci_low": lo, "ci_high": hi, "p": pv, "n_seeds": len(grp)})
        adj = holm([r["p"] for r in block])
        for r, q in zip(block, adj):
            r["p_holm"] = q
            r["call"] = ("higher" if r["mean"] > 0 else "lower") if q < 0.05 else "ns"
        rows += block
        per_seed = diff.groupby("seed")["diff"].mean()
        mean, sd, lo, hi, pv = paired(per_seed)
        cond_means = np.array([r["mean"] for r in block])
        w = stats.wilcoxon(cond_means).pvalue if np.any(cond_means != 0) else 1.0
        overall.append({"contrast": f"{a} - {b}", "metric": metric, "mean_over_conditions": mean,
                        "ci_low": lo, "ci_high": hi, "p_seed_t": pv, "p_wilcoxon_conditions": w,
                        "n_higher": sum(r["call"] == "higher" for r in block),
                        "n_lower": sum(r["call"] == "lower" for r in block),
                        "n_ns": sum(r["call"] == "ns" for r in block),
                        "n_positive_means": int((cond_means > 0).sum()), "n_conditions": len(block)})
    return pd.DataFrame(rows), pd.DataFrame(overall)


def leakage_table(clean, leaky):
    keys = ["n_genes", "mask_rate", "seed", "method"]
    m = clean.merge(leaky, on=keys, suffixes=("_clean", "_leaky"))
    rows, overall = [], []
    for method in ALL_METHODS:
        sub = m[m.method == method]
        if sub.empty:
            continue
        for (g, p), grp in sub.groupby(["n_genes", "mask_rate"]):
            mean, sd, lo, hi, pv = paired(grp["test_r2_leaky"] - grp["test_r2_clean"])
            rows.append({"method": method, "n_genes": g, "mask_rate": p, "clean_r2": grp["test_r2_clean"].mean(),
                         "leaky_r2": grp["test_r2_leaky"].mean(), "delta": mean, "ci_low": lo, "ci_high": hi, "p": pv})
        per_seed = (sub["test_r2_leaky"] - sub["test_r2_clean"]).groupby(sub["seed"]).mean()
        mean, sd, lo, hi, pv = paired(per_seed)
        mse_ratio = (sub["test_mse_leaky"] / sub["test_mse_clean"]).mean()
        overall.append({"method": method, "mean_delta_r2": mean, "ci_low": lo, "ci_high": hi, "p_seed_t": pv,
                        "mean_mse_ratio_leaky_over_clean": mse_ratio})
    return pd.DataFrame(rows), pd.DataFrame(overall)


def ranking(df, metric="test_r2"):
    means = df.groupby(["n_genes", "mask_rate", "method"])[metric].mean().unstack("method")
    learned = means[[c for c in LEARNED if c in means]]
    winners = learned.idxmax(axis=1)
    ranks = learned.rank(axis=1, ascending=False).mean()
    return winners.value_counts().to_dict(), ranks.to_dict(), means


def ablation_table(abl):
    wide = abl.pivot_table(index=["n_genes", "mask_rate", "seed"], columns="method", values="test_r2")
    variants = [c for c in wide.columns if c != "Full"]
    rows, overall = [], []
    for v in variants:
        diff = (wide[v] - wide["Full"]).rename("diff").reset_index()
        block = []
        for (g, p), grp in diff.groupby(["n_genes", "mask_rate"]):
            mean, sd, lo, hi, pv = paired(grp["diff"])
            block.append({"variant": v, "n_genes": g, "mask_rate": p, "full_r2": wide.loc[(g, p), "Full"].mean(),
                          "variant_r2": wide.loc[(g, p), v].mean(), "mean": mean, "ci_low": lo, "ci_high": hi, "p": pv})
        for r, q in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = q
        rows += block
        per_seed = diff.groupby("seed")["diff"].mean()
        mean, sd, lo, hi, pv = paired(per_seed)
        overall.append({"variant": v, "mean_delta_r2": mean, "ci_low": lo, "ci_high": hi, "p_seed_t": pv,
                        "n_conditions_higher": sum(r["mean"] > 0 and r["p_holm"] < 0.05 for r in block),
                        "n_conditions_lower": sum(r["mean"] < 0 and r["p_holm"] < 0.05 for r in block)})
    return pd.DataFrame(rows), pd.DataFrame(overall)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.concat([load("main", "clean"), load("propagation", "clean")], ignore_index=True)
    leaky = pd.concat([load("main", "leaky"), load("propagation", "leaky")], ignore_index=True)
    complete = clean.groupby(["n_genes", "mask_rate", "seed"])["method"].transform("nunique") == len(ALL_METHODS)
    clean = clean[complete]
    if not leaky.empty:
        complete = leaky.groupby(["n_genes", "mask_rate", "seed"])["method"].transform("nunique") == len(ALL_METHODS)
        leaky = leaky[complete]
    abl = load("ablation", "clean")
    summary = {"records": {"clean_runs": int(clean.shape[0]), "leaky_runs": int(leaky.shape[0]),
                           "ablation_runs": int(abl.shape[0])}}
    clean_sel, choices = add_selector(clean)
    clean_sel.to_csv(OUT / "main_runs_clean.csv", index=False)
    choices.to_csv(OUT / "selector_choices.csv", index=False)
    cond = clean_sel.groupby(["n_genes", "mask_rate", "method"])[[f"test_{m}" for m in METRICS]].agg(["mean", "std"])
    cond.columns = [f"{a}_{b}" for a, b in cond.columns]
    cond.reset_index().to_csv(OUT / "main_conditions_clean.csv", index=False)
    by_cond, overall = contrast_table(clean_sel)
    by_cond.to_csv(OUT / "contrasts_r2_clean.csv", index=False)
    overall.to_csv(OUT / "contrasts_overall_clean.csv", index=False)
    for metric in ("test_mse", "test_mae", "test_pearson"):
        _, o = contrast_table(clean_sel, metric)
        o.to_csv(OUT / f"contrasts_overall_clean_{metric}.csv", index=False)
    wins, ranks, means = ranking(clean)
    summary["clean"] = {"winners": wins, "mean_rank": ranks,
                        "grand_mean_r2": clean_sel.groupby("method")["test_r2"].mean().to_dict(),
                        "grand_mean_mae": clean_sel.groupby("method")["test_mae"].mean().to_dict(),
                        "grand_mean_pearson": clean_sel.groupby("method")["test_pearson"].mean().to_dict(),
                        "selector_choice_counts": choices["choice"].value_counts().to_dict()}
    if not leaky.empty:
        leaky_sel, _ = add_selector(leaky)
        wins_l, ranks_l, _ = ranking(leaky)
        by_cond_l, overall_l = contrast_table(leaky_sel)
        overall_l.to_csv(OUT / "contrasts_overall_leaky.csv", index=False)
        by_cond_l.to_csv(OUT / "contrasts_r2_leaky.csv", index=False)
        summary["leaky"] = {"winners": wins_l, "mean_rank": ranks_l,
                            "grand_mean_r2": leaky_sel.groupby("method")["test_r2"].mean().to_dict()}
        lk, lk_overall = leakage_table(clean, leaky)
        lk.to_csv(OUT / "leakage_by_condition.csv", index=False)
        lk_overall.to_csv(OUT / "leakage_overall.csv", index=False)
    if not abl.empty:
        ab, ab_overall = ablation_table(abl)
        ab.to_csv(OUT / "ablation_by_condition.csv", index=False)
        ab_overall.to_csv(OUT / "ablation_overall.csv", index=False)
        gates = abl.groupby(["n_genes", "mask_rate", "method"])[["gate_mean", "gate_frac"]].mean().reset_index()
        gates.to_csv(OUT / "ablation_gates.csv", index=False)
    gate = clean[clean.method == "Gated"].groupby(["n_genes", "mask_rate"])[["gate_mean", "gate_sd", "gate_frac"]]
    gate.agg(["mean", "std"]).to_csv(OUT / "gate_clean.csv")
    cost = clean[clean.method.isin(LEARNED)].groupby(["method", "n_genes"]).agg(
        parameters=("parameters", "first"), fit_seconds_median=("fit_seconds", "median"),
        best_epoch_median=("best_epoch", "median")).reset_index()
    cost.to_csv(OUT / "cost.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    print(overall.to_string())


if __name__ == "__main__":
    import sys
    if "--pack" in sys.argv:
        pack()
    else:
        main()
