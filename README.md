# DreamCache

**Selective speculative execution for LLM agents.** A small LLM acts on most steps.
A difficulty detector looks only at what the agent can observe and decides when a step
is hard. On hard steps, the agent clones the simulator into K sandboxes, rolls each
candidate action forward H steps, scores the outcomes, and executes the best one.

The repo also includes speculative-decoding-style alternatives (a small model drafts,
a big model verifies), model cascades, self-consistency voting, and UCT-MCTS, all
evaluated on MiniGrid.

- Results and analysis: [report/REPORT.md](report/REPORT.md)
- How we collaborate: [CONTRIBUTING.md](CONTRIBUTING.md) (issue first, declare the files you'll touch)

## Quick start (CPU, no LLM)
```bash
uv sync --extra dev
uv run pytest                      # clone-correctness + pipeline tests, ~2 s
scripts/run_local.sh               # surrogate-policy smoke run (no model inference)
```

## Cluster (GT Futurama, all LLM inference happens here)
```bash
# GT VPN on. Syncs code, submits one GPU job that starts a private Ollama on 127.0.0.1
scripts/run_cluster.sh configs/smoke.yaml
DC_SBATCH_ARGS="--gres=gpu:h200:1 -t 02:00:00" scripts/run_cluster.sh configs/val_dyn8_core.yaml
# results land in ~/dreamcache/results/<timestamp>-<config>/ (manifest.json + episodes.jsonl)
uv run python analysis/summarize.py results/<run> --out report/<name>
```
GPU sizing: see [configs/gpus.md](configs/gpus.md).

## Layout
| Path | What |
|---|---|
| `src/dreamcache/environments/` | MiniGrid adapter, verified deep-copy cloning, observation-only text and features |
| `src/dreamcache/policies/` | Ollama client and LLM policy (decide / propose / verify / draft), heuristics, surrogate |
| `src/dreamcache/detection/` | entropy, disagreement, observable risk → difficulty score |
| `src/dreamcache/speculation/` | sandbox rollouts, evaluator, executor (CRN, parallel), UCT-MCTS |
| `src/dreamcache/agents/` | `Agent` (all conditions), oracle Q_ref |
| `src/dreamcache/experiments/` | runner, metrics, bootstrap statistics, oracle state labeling |
| `analysis/` | summary tables and plots, detector evaluation |
