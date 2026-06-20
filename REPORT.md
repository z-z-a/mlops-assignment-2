# Report: LLM inference + o11y

## 1. Serving configuration (Phase 1)

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (served as the FP8 checkpoint `…-2507-FP8`)
**Hardware:** 1× H100 80GB

### Workload profile that drove the choices
- MoE 30B total / 3B active → compute per token is small, but the full 30B of weights sits in VRAM, so **memory (KV cache), not FLOPs, is the constraint**.
- 2507-**Instruct** is the non-thinking variant → no `<think>` overhead eating the latency budget.
- Prompts 1.5–3K tokens, short SQL outputs → **prefill-dominated**.
- The DB schema + system prompt prefix repeats on every call → prefix caching pays off.
- 10 RPS × 2–3 calls ≈ 20–30 req/s, P95 < 5s → needs concurrency headroom.

### Flags

| Flag | Value | Justification |
|---|---|---|
| FP8 checkpoint (`…-2507-FP8`) | FP8 | Halves weights to ~30GB, freeing ~45GB of the 80GB for KV cache, and uses Hopper FP8 tensor cores. Fallback: BF16 model + `--quantization fp8`. |
| `--served-model-name` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Keeps the agent's `.env` `VLLM_MODEL` valid while serving the FP8 repo. |
| `--max-model-len` | `8192` | Workload is ≤3K prompt + short SQL; native 256K would waste the KV budget. Tight ceiling = far more concurrent sequences. |
| `--gpu-memory-utilization` | `0.92` | Dedicated single-model box, so claim almost all of the 80GB for KV cache headroom. |
| `--enable-prefix-caching` | on | Schema + system prompt prefix (~1.5–3K tokens) is identical across requests; cache once → big TTFT/prefill win. Highest-leverage flag for this agent. |
| `--enable-chunked-prefill` | on | Large prompts; chunking interleaves prefill with ongoing decodes so a 3K prefill doesn't stall others' inter-token latency. |
| `--max-num-batched-tokens` | `8192` | Prefill-heavy workload benefits from a large prefill batch; Phase 6 tuning knob (TTFT vs throughput). |
| `--max-num-seqs` | `128` | Bounds concurrency so queueing stays predictable under 20–30 req/s; Phase 6 tuning knob. |
| TP=1, CUDA graphs on | — | One GPU so no tensor parallelism; keep CUDA graphs (no `--enforce-eager`) for low decode latency. |

**Agent-side sampling:** SQL generation uses **temperature 0** (greedy, deterministic — best for SQL correctness) and **`max_tokens` ~256–512**.

### Memory sanity check
KV per token ≈ `2 × layers × kv_heads × head_dim × dtype_bytes` ≈ ~96 KB/token (BF16, ~48 layers / GQA 4 KV heads / head_dim 128 — confirm from `config.json`). A 3K-token prompt ≈ ~290MB; with ~45GB free that's ~150 concurrent sequences — ample for 20–30 req/s with short outputs.

### To revisit in Phase 6
- `--kv-cache-dtype fp8` — doubles KV capacity; A/B against eval pass rate first.
- Guided decoding (`xgrammar`) on the `verify` node's `{ok, issue}` JSON to kill parse failures.

### Manual verification
See `screenshots/vllm_manual_query.png` — vLLM serving + a manual query returning SQL.

---
