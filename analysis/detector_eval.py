"""Offline detector evaluation against ORACLE criticality labels (Section 10, Ablation 4).

critical = regret > delta, with regret from Q_ref (approximate, finite Monte Carlo).
For each signal combination: PR-AUC (average precision), precision/recall/F1 at the
threshold sweep, and at trigger budgets: share of total regret captured.

Usage: python analysis/detector_eval.py labels.jsonl --out report/detector [--delta 0.2]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMBOS = {"entropy": (1, 0, 0), "disagreement": (0, 1, 0), "risk": (0, 0, 1),
          "entropy+disagreement": (0.5, 0.5, 0), "all (0.4/0.3/0.3)": (0.4, 0.3, 0.3),
          "random": None}


def score(rows, w, rng):
    if w is None:
        return rng.random(len(rows))
    s = np.array([[r["signals"]["entropy"], r["signals"]["disagreement"], r["signals"]["risk"]] for r in rows])
    return s @ np.array(w) / sum(w)


def average_precision(y, s):
    order = np.argsort(-s, kind="stable")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.6, help="'confident' = top normalized score >= this")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(line) for line in open(a.labels)]
    rng = np.random.default_rng(0)
    reg = np.array([r["regret"] for r in rows])
    y = (reg > a.delta).astype(float)
    conf_wrong = np.array([r["fast_conf"] >= a.conf for r in rows]) & (y > 0)
    res = {"n_states": len(rows), "delta": a.delta, "base_rate": float(y.mean()),
           "n_critical": int(y.sum()), "n_confidently_wrong": int(conf_wrong.sum()),
           "by_method_states": {m: sum(r["method"] == m for r in rows) for m in sorted({r["method"] for r in rows})},
           "combos": {}}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    budgets = [0.05, 0.1, 0.2, 0.4, 1.0]
    for name, w in COMBOS.items():
        s = score(rows, w, rng)
        c = {"pr_auc": average_precision(y, s), "thresholds": {}, "budgets": {}}
        for th in (0.2, 0.4, 0.6, 0.8):
            t = s >= th
            tp = float((t * y).sum())
            p = tp / t.sum() if t.sum() else float("nan")
            rc = tp / y.sum() if y.sum() else float("nan")
            c["thresholds"][th] = {"trigger_rate": float(t.mean()), "precision": p, "recall": rc,
                                   "f1": 2 * p * rc / (p + rc) if p and rc and p + rc else 0.0,
                                   "regret_captured": float(reg[t].sum() / reg.sum()) if reg.sum() else float("nan"),
                                   "conf_wrong_recall": float((t & conf_wrong).sum() / conf_wrong.sum()) if conf_wrong.sum() else float("nan")}
        order = np.argsort(-s, kind="stable")
        caps = []
        for b in budgets:
            k = max(1, int(round(b * len(rows))))
            top = order[:k]
            cap = float(reg[top].sum() / reg.sum()) if reg.sum() else float("nan")
            c["budgets"][b] = {"regret_captured": cap, "recall": float(y[top].sum() / y.sum()) if y.sum() else float("nan")}
            caps.append(cap)
        res["combos"][name] = c
        ax[0].plot(budgets, caps, marker="o", label=name)
        prec = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
        rec = np.cumsum(y[order]) / max(1, y.sum())
        ax[1].plot(rec, prec, label=f"{name} (AP={c['pr_auc']:.2f})")
    ax[0].set_xlabel("trigger budget (fraction of decisions)")
    ax[0].set_ylabel("share of total regret captured")
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].axhline(y.mean(), color="gray", ls=":", lw=1)
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.suptitle(f"Detector vs oracle criticality (regret > {a.delta}); n={len(rows)} states, base rate {y.mean():.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "detector_pr_and_budget.png"), dpi=130)
    json.dump(res, open(os.path.join(a.out, "detector_eval.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "combos"}, indent=1))
    for n, c in res["combos"].items():
        print(f"{n:24s} AP={c['pr_auc']:.3f}  regret@10%={c['budgets'][0.1]['regret_captured']:.3f} "
              f"@20%={c['budgets'][0.2]['regret_captured']:.3f}")


if __name__ == "__main__":
    main()
