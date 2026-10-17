# GPU deployment notes (measured, Futurama, Oct 2026)

| GPU | Model | VRAM used (Ollama, ctx 4096, NUM_PARALLEL=8) | p50 latency, 1 req | throughput @16 concurrent |
|---|---|---|---|---|
| L4 24 GB | qwen3:4b (Q4_K_M) | 7.4 GB | 0.76–0.83 s | 3.0 calls/s |
| L4 24 GB | qwen3:4b + qwen3:8b both loaded | 17.0 GB | 8b: 1.11 s | 8b: 3.0 calls/s |

Measured from `scripts/bench_ollama.py` (job 10453), using real MiniGrid prompts (~260 prompt tokens, ~37 generated tokens, `think: false`).

Presets: `configs/deploy/{l40,small_gpu,gpu_8gb,cpu}.yaml`.

Recommendations:
- **L40 48 GB** (down during this work): qwen3:4b + qwen3:8b with NUM_PARALLEL 16. A 14b verifier also fits.
- **16–24 GB** (L4): qwen3:4b alone, or 4b + 8b (17 GB measured). Keep `OLLAMA_CONTEXT_LENGTH=4096`.
- **8 GB**: qwen3:4b only, `OLLAMA_NUM_PARALLEL=2`, context 2048. Expect about 1 call/s.
- **CPU**: tests only (`fast: {kind: surrogate}`). No LLM inference on laptops.

Throughput scales sub-linearly with concurrency: 1.2 → 3.0 calls/s going from 1 to 16 concurrent requests, while p50 latency rises from 0.8 s to 4.7 s. Ten parallel requests are not ten times faster.
