"""Experiment 2 Track A: per-decision latency, paired by seed, plus compute per decision.

Episode wall-clock is confounded by episode length (methods reach different outcomes),
so the primary latency metric is each seed's MEAN DECISION LATENCY, compared pairwise
against the reference on common seeds. Compute is model-seconds PER DECISION, including
wasted speculative calls.

Usage: python analysis/specgraph_latency.py RUN_DIR [RUN_DIR...] --ref B0_no_spec --out DIR
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dreamcache.experiments.metrics import episode_metrics, load  # noqa: E402
from dreamcache.experiments.statistics import bootstrap_ci, paired_diff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="B0_no_spec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by = defaultdict(dict)
    for r in a.runs:
        eps, _ = load(os.path.join(r, "episodes.jsonl"))
        for e in eps:
            m = episode_metrics(e)
            m["lat_per_decision"] = float(np.mean(m["_latencies"])) if m["_latencies"] else np.nan
            m["model_s_per_decision"] = m["llm_seconds"] / max(1, m["steps"])
            by[e["method"]].setdefault(e["seed"], m)
    ref = by[a.ref]
    rows = []
    for meth, seeds in by.items():
        lat = {s: m["lat_per_decision"] for s, m in seeds.items()}
        cpd = {s: m["model_s_per_decision"] for s, m in seeds.items()}
        d = paired_diff(lat, {s: m["lat_per_decision"] for s, m in ref.items()})
        dc = paired_diff(cpd, {s: m["model_s_per_decision"] for s, m in ref.items()})
        ci = bootstrap_ci(list(lat.values()))
        g = lambda k: float(np.nanmean([m.get(k, np.nan) for m in seeds.values()]))  # noqa: E731
        rows.append({"method": meth, "n": len(seeds), "success": g("success"),
                     "lat_per_decision": ci[0], "lat_lo": ci[1], "lat_hi": ci[2],
                     "d_lat_vs_ref": d, "rel_lat_change": d["diff"] / np.nanmean(list(ref_lat := {s: m["lat_per_decision"] for s, m in ref.items()}.values())),
                     "model_s_per_decision": float(np.mean(list(cpd.values()))), "d_compute_vs_ref": dc,
                     "hit_per_step": g("prefetch_hit_rate"), "reuse_of_launched": g("spec_reuse_rate"),
                     "late_of_launched": g("spec_late_rate"), "unused_of_launched": g("spec_unused_rate"),
                     "wasted_model_s_per_ep": g("wasted_llm_seconds"), "predict_overhead_s_per_ep": g("predict_overhead_s")})
    rows.sort(key=lambda r: r["lat_per_decision"])
    json.dump(rows, open(os.path.join(a.out, "latency_paired.json"), "w"), indent=2, default=float)
    lines = [f"Reference: {a.ref}. Paired over common seeds; 95% bootstrap CIs.", "",
             "| method | n | success | decision latency s [95% CI] | Δ vs ref (paired) [95% CI], p | rel. | model-s / decision | Δ compute / decision | hit / step | launched: reused / late / unused | wasted model-s / ep |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, dc = r["d_lat_vs_ref"], r["d_compute_vs_ref"]
        f = lambda x: "—" if x != x else f"{x:.2f}"  # noqa: E731
        lines.append(f"| {r['method']} | {r['n']} | {r['success']:.2f} | {r['lat_per_decision']:.3f} [{r['lat_lo']:.3f}, {r['lat_hi']:.3f}] | "
                     f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], p={d['p']:.3f} | {r['rel_lat_change']:+.1%} | "
                     f"{r['model_s_per_decision']:.3f} | {dc['diff']:+.3f} | {f(r['hit_per_step'])} | "
                     f"{f(r['reuse_of_launched'] - r['late_of_launched'] if r['reuse_of_launched'] == r['reuse_of_launched'] else float('nan'))} / "
                     f"{f(r['late_of_launched'])} / {f(r['unused_of_launched'])} | {f(r['wasted_model_s_per_ep'])} |")
    open(os.path.join(a.out, "latency_paired.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        ax.errorbar(r["model_s_per_decision"], r["lat_per_decision"],
                    yerr=[[r["lat_per_decision"] - r["lat_lo"]], [r["lat_hi"] - r["lat_per_decision"]]], fmt="o", capsize=3)
        ax.annotate(r["method"], (r["model_s_per_decision"], r["lat_per_decision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("model-seconds per decision (incl. wasted speculation)")
    ax.set_ylabel("mean decision latency (s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "latency_vs_compute.png"), dpi=130)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    spec = [r for r in rows if r["reuse_of_launched"] == r["reuse_of_launched"]]
    xs = np.arange(len(spec))
    reused = [r["reuse_of_launched"] - r["late_of_launched"] for r in spec]
    late = [r["late_of_launched"] for r in spec]
    unused = [r["unused_of_launched"] for r in spec]
    ax.bar(xs, reused, label="reused (finished in time)", color="#55A868")
    ax.bar(xs, late, bottom=reused, label="reused late (partial saving)", color="#8172B2")
    ax.bar(xs, unused, bottom=np.array(reused) + np.array(late), label="unused (wasted)", color="#C44E52")
    ax.set_xticks(xs, [r["method"] for r in spec], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("fraction of launched speculations")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "speculation_fates.png"), dpi=130)


if __name__ == "__main__":
    main()
