"""Summarize run directories: tables with 95% bootstrap CIs, paired diffs, plots.

Usage: python analysis/summarize.py RUN_DIR [RUN_DIR ...] --out report/<name> [--ref A_greedy]
"""
from __future__ import annotations

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

COLS = ["success", "failure", "return", "steps", "trigger_rate", "llm_calls", "gen_tokens",
        "llm_seconds", "sim_steps", "sim_cpu_seconds", "compute_seconds", "episode_seconds",
        "decision_latency_mean", "verify_accept_rate", "draft_accepted_mean", "draft_calls_per_step", "prefetch_hit_rate", "actor_calls_per_step",
        "spec_reuse_rate", "spec_late_rate", "spec_unused_rate", "wasted_llm_seconds", "predict_overhead_s"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="A_greedy")
    ap.add_argument("--compare", default="B_always")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows, errors = [], 0
    for r in a.runs:
        eps, errs = load(os.path.join(r, "episodes.jsonl"))
        errors += len(errs)
        rows += [episode_metrics(e) for e in eps]
    by = defaultdict(list)
    for m in rows:
        by[(m["env"], m["method"])].append(m)

    summary = []
    for (env, meth), ms in sorted(by.items()):
        s = {"env": env, "method": meth, "n": len(ms)}
        for c in COLS:
            vals = [m[c] for m in ms if c in m and not (isinstance(m[c], float) and np.isnan(m[c]))]
            mean, lo, hi = bootstrap_ci(vals)
            s[c], s[c + "_lo"], s[c + "_hi"] = mean, lo, hi
        lat = np.concatenate([m["_latencies"] for m in ms]) if ms else np.array([0.0])
        s.update({f"lat_p{q}": float(np.percentile(lat, q)) for q in (50, 95, 99)})
        for refname in (a.ref, a.compare):
            ref = by.get((env, refname))
            if ref and meth != refname:
                for c in ("success", "compute_seconds", "episode_seconds"):
                    d = paired_diff({m["seed"]: m[c] for m in ms}, {m["seed"]: m[c] for m in ref})
                    s[f"d_{c}_vs_{refname}"] = d
        summary.append(s)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    lines = [f"Episodes with runner errors (excluded, listed in episodes.jsonl): {errors}\n",
             "| env | method | n | success [95% CI] | collision | steps | trigger rate | LLM calls/ep | model-s/ep | sim CPU-s/ep | episode s | lat p50/p95 (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['env'].replace('MiniGrid-','')} | {s['method']} | {s['n']} | "
                     f"{s['success']:.2f} [{s['success_lo']:.2f}, {s['success_hi']:.2f}] | {s['failure']:.2f} | "
                     f"{s['steps']:.1f} | {s['trigger_rate']:.2f} | {s['llm_calls']:.1f} | {s['llm_seconds']:.1f} | "
                     f"{s['sim_cpu_seconds']:.2f} | {s['episode_seconds']:.1f} | {s['lat_p50']:.2f}/{s['lat_p95']:.2f} |")
    lines += ["", "Paired differences on common seeds (method minus reference; 95% bootstrap CI; bootstrap p):", "",
              "| env | method | vs | Δ success | Δ compute-s | Δ episode-s |", "|---|---|---|---|---|---|"]
    for s in summary:
        for refname in (a.ref, a.compare):
            k = f"d_success_vs_{refname}"
            if k in s:
                f = lambda d: f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p']:.3f}"  # noqa: E731
                lines.append(f"| {s['env'].replace('MiniGrid-','')} | {s['method']} | {refname} | {f(s[k])} | "
                             f"{f(s[f'd_compute_seconds_vs_{refname}'])} | {f(s[f'd_episode_seconds_vs_{refname}'])} |")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    for env in sorted({s["env"] for s in summary}):
        ss = [s for s in summary if s["env"] == env]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, x, xl in ((axes[0], "trigger_rate", "speculation (trigger) rate"),
                          (axes[1], "compute_seconds", "compute per episode (model-s + sim CPU-s)")):
            for s in ss:
                ax.errorbar(s[x], s["success"], yerr=[[s["success"] - s["success_lo"]], [s["success_hi"] - s["success"]]],
                            fmt="o", capsize=3)
                ax.annotate(s["method"], (s[x], s["success"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(xl)
            ax.set_ylabel("success rate")
            ax.grid(alpha=0.3)
        axes[1].set_xscale("symlog", linthresh=0.1)
        fig.suptitle(env)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"success_vs_compute_{env.replace('MiniGrid-','')}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
