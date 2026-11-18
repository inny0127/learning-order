"""Exp 11 Part B analysis: order spread of prequential code length vs model size.

Usage: python analyze11.py --runs results/runs/order --out results
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp05_order_compression"))
from analyze05 import spearman  # noqa: E402

PARAMS = {"m1": 0.94e6, "m4": 4.23e6, "m16": 15.8e6, "m64": 61.7e6, "m128": 119.4e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/order")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(a.runs, "*.json")):
        r = json.load(open(p))
        g, b, tpg = r["groups"], np.array(r["bits_per_group"]), r["tokens_per_group"]
        code = b[[g.index(k) for k in ("stories", "facts", "dupes")]].sum() / (tpg * 3)
        by[r["args"]["size"]][r["args"]["order"]].append(
            {"code": code, "final_val": r["curve"][-1]["val_loss"], "ens_val": r["path_ensemble"]["val_loss"],
             "final_fact": r["curve"][-1]["fact_acc"], "ens_fact": r["path_ensemble"]["fact_acc"]})
    rows, lines = {}, ["| Size | Params | Runs | Mean code | Order spread Δ | ρ(code, ensemble story loss) | ρ(code, final story loss) |",
                       "|---|---|---|---|---|---|---|"]
    for s in PARAMS:
        if s not in by:
            continue
        means = {o: np.mean([x["code"] for x in v]) for o, v in by[s].items()}
        allr = [x for v in by[s].values() for x in v]
        d = (max(means.values()) - min(means.values())) / np.mean(list(means.values()))
        re = spearman([x["code"] for x in allr], [x["ens_val"] for x in allr])
        rf = spearman([x["code"] for x in allr], [x["final_val"] for x in allr])
        rows[s] = {"params": PARAMS[s], "n": len(allr), "mean_code": float(np.mean(list(means.values()))), "delta": float(d),
                   "rho_ens": re, "rho_final": rf, "order_means": {k: float(v) for k, v in means.items()}}
        lines.append(f"| {s} | {PARAMS[s] / 1e6:.1f}M | {len(allr)} | {rows[s]['mean_code']:.3f} | {d:.4f} | {re:.2f} | {rf:.2f} |")
    ck = {}
    ss = [s for s in PARAMS if s in rows]
    if len(ss) >= 3:
        D = [rows[s]["delta"] for s in ss]
        N = [rows[s]["params"] for s in ss]
        slope, icpt = np.polyfit(np.log(N), np.log(D), 1)
        pred = slope * np.log(N) + icpt
        r2 = 1 - np.sum((np.log(D) - pred) ** 2) / np.sum((np.log(D) - np.mean(np.log(D))) ** 2)
        ck["H-B1 delta decreases monotonically"] = bool(all(D[i + 1] < D[i] for i in range(len(D) - 1)))
        ck["H-B2 power law R2 >= 0.9"] = {"exponent": float(slope), "r2": float(r2), "holds": bool(r2 >= 0.9)}
        ck["H-B3 rho(code, ensemble) >= 0.8 at every size"] = {"min": min(rows[s]["rho_ens"] for s in ss),
                                                              "holds": bool(min(rows[s]["rho_ens"] for s in ss) >= 0.8)}
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
        ax[0].loglog(N, D, "o-", color="#2a78d6")
        ax[0].set(xlabel="model parameters", ylabel="order spread of code length Δ", title=f"Order sensitivity vs scale (slope {slope:.2f})")
        ax[1].semilogx(N, [rows[s]["rho_ens"] for s in ss], "o-", color="#1baf7a", label="path ensemble")
        ax[1].semilogx(N, [rows[s]["rho_final"] for s in ss], "o-", color="#d9483b", label="final model")
        ax[1].set(xlabel="model parameters", ylabel="Spearman ρ(code, story loss)", title="Compression predicts the path at every size?")
        ax[1].legend(frameon=False)
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "scale.png"), dpi=150)
    md = "\n".join(lines)
    open(os.path.join(a.out, "table.md"), "w").write(md + "\n")
    json.dump({"rows": rows, "checks": ck}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(md)
    print(json.dumps(ck, indent=1, default=float))


if __name__ == "__main__":
    main()
