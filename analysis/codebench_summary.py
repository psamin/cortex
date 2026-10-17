"""Track B analysis: paired latency vs `none`, reuse breakdown, wasted compute, timeline.

Usage: python analysis/codebench_summary.py RUN_DIR --out DIR
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dreamcache.experiments.statistics import bootstrap_ci, paired_diff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(line) for line in open(os.path.join(a.run, "tasks.jsonl"))]
    rep = [r for r in rows if r["run"] == "replay"]
    by = defaultdict(list)
    for r in rep:
        by[r["policy"]].append(r)
    base = {(r["task"], r["rep"]): r for r in by.get("none", [])}
    table = []
    for pol, rs in by.items():
        tool_crit = [sum(s["tool_s"] for s in r["steps"] if s["op"] in ("run_tests", "lint", "deps_check")) for r in rs]
        think = [sum(s["think_s"] for s in r["steps"]) for r in rs]
        fates = Counter(s["fate"] for r in rs for s in r["steps"] if s["op"] in ("run_tests", "lint", "deps_check"))
        spec = Counter()
        for r in rs:
            for k in ("launched", "reused", "reused_late", "unused"):
                spec[k] += r["spec"].get(k, 0)
        d = paired_diff({(r["task"], r["rep"]): r["wall_s"] for r in rs}, {k: v["wall_s"] for k, v in base.items()})
        dt = paired_diff({(r["task"], r["rep"]): t for r, t in zip(rs, tool_crit)},
                         {k: sum(s["tool_s"] for s in v["steps"] if s["op"] in ("run_tests", "lint", "deps_check"))
                          for k, v in base.items()})
        same_ops = np.mean([[s["op"] for s in r["steps"]] == [s["op"] for s in base[(r["task"], r["rep"])]["steps"]]
                            for r in rs if (r["task"], r["rep"]) in base])
        w = bootstrap_ci([r["wall_s"] for r in rs])
        table.append({"policy": pol, "n": len(rs), "success": float(np.mean([r["success"] for r in rs])),
                      "wall_mean": w[0], "wall_lo": w[1], "wall_hi": w[2],
                      "wall_p50": float(np.median([r["wall_s"] for r in rs])),
                      "wall_p95": float(np.percentile([r["wall_s"] for r in rs], 95)),
                      "think_mean": float(np.mean(think)), "tool_critical_mean": float(np.mean(tool_crit)),
                      "d_wall_vs_none": d, "d_toolcrit_vs_none": dt, "identical_op_sequence_rate": float(same_ops),
                      "tool_fates": dict(fates), "spec": dict(spec),
                      "spec_cpu_s": float(sum(r["spec"].get("spec_cpu_s", 0) for r in rs)),
                      "wasted_cpu_s": float(sum(r["spec"].get("wasted_cpu_s", 0) for r in rs)),
                      "copy_overhead_s": float(sum(r["spec"].get("copy_overhead_s", 0) for r in rs))})
    order = ["none", "memo", "all", "random", "top1_frequency", "top1_global", "top1_personalized",
             "topk2_personalized", "utility_personalized"]
    table.sort(key=lambda t: order.index(t["policy"]) if t["policy"] in order else 99)
    json.dump(table, open(os.path.join(a.out, "codebench_summary.json"), "w"), indent=2)
    lines = ["| policy | n | success | wall s (mean [95% CI]) | p50 / p95 | think s | tool s on critical path | Δ wall vs none [95% CI] | reused / late / unused of launched | spec CPU-s | wasted CPU-s | same op sequence |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        d, s = t["d_wall_vs_none"], t["spec"]
        lines.append(f"| {t['policy']} | {t['n']} | {t['success']:.2f} | {t['wall_mean']:.2f} [{t['wall_lo']:.2f}, {t['wall_hi']:.2f}] | "
                     f"{t['wall_p50']:.2f} / {t['wall_p95']:.2f} | {t['think_mean']:.2f} | {t['tool_critical_mean']:.2f} | "
                     f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] | {s.get('reused', 0)} / {s.get('reused_late', 0)} / "
                     f"{s.get('unused', 0)} of {s.get('launched', 0)} | {t['spec_cpu_s']:.1f} | {t['wasted_cpu_s']:.1f} | "
                     f"{t['identical_op_sequence_rate']:.2f} |")
    open(os.path.join(a.out, "codebench_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # latency vs speculative compute
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in table:
        ax.scatter(t["spec_cpu_s"] / max(1, t["n"]), t["wall_mean"])
        ax.annotate(t["policy"], (t["spec_cpu_s"] / max(1, t["n"]), t["wall_mean"]), fontsize=7)
    ax.set_xlabel("speculative CPU-s per task")
    ax.set_ylabel("mean task wall-clock (s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "codebench_latency_vs_compute.png"), dpi=130)
    # per-task paired improvements for the best learned policy
    best = min((t for t in table if t["policy"] != "none"), key=lambda t: t["wall_mean"], default=None)
    if best:
        rs = {(r["task"], r["rep"]): r["wall_s"] for r in by[best["policy"]]}
        diffs = sorted(base[k]["wall_s"] - v for k, v in rs.items() if k in base)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(range(len(diffs)), diffs)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("wall-clock saved (s)")
        ax.set_xlabel(f"eval task (sorted), policy={best['policy']} vs none")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "codebench_per_task_savings.png"), dpi=130)
    # timeline (needs event timestamps; present in runs after the timeline instrumentation commit)
    for pol in ("none", best["policy"] if best else None):
        r = next((r for r in by.get(pol, []) if "t_think" in r["steps"][0]), None) if pol else None
        if r is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 2.6))
        for s in r["steps"]:
            ax.barh(1, s["t_tool"] - s["t_think"], left=s["t_think"], color="#4C72B0")
            col = "#55A868" if s["fate"].startswith("reused") else "#DD8452"
            ax.barh(1, max(0.01, s["t_end"] - s["t_tool"]), left=s["t_tool"], color=col)
            ax.text(s["t_think"], 1.42, s["op"], fontsize=6, rotation=30)
        for e in r["spec"].get("events", []):
            col = {"reused": "#55A868", "reused_late": "#8172B2", "unused": "#C44E52"}.get(e["fate"], "gray")
            ax.barh(0, e["finish"] - e["start"], left=e["start"], color=col, alpha=0.8)
        ax.set_yticks([0, 1], ["speculative workers", "critical path"])
        ax.set_xlabel("seconds since task start (blue = LLM think, orange = tool on critical path, "
                      "green = reused speculative result, red = wasted speculation)")
        ax.set_title(f"{r['task']} — policy {pol}")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"timeline_{pol}.png"), dpi=130)


if __name__ == "__main__":
    main()
