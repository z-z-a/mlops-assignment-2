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

## 2. Baseline eval results (Phase 5)

30 BIRD questions, scored by **execution accuracy** (canonicalized row sets:
rows sorted, cells stringified, `NULL`→`""`), against the real
`Qwen3-30B-A3B-Instruct-2507` FP8 endpoint. Per-iteration SQL is reconstructed
from the agent's `history`, so we can ask "what would the pass rate be if we had
stopped after iter k?" with carry-forward for early-terminating runs.

| Metric | Value |
|---|---|
| Overall pass rate | **36.7%** (11/30) |
| Pass rate @ iter 0 (generate only) | 33.3% (10/30) |
| Pass rate @ iter 1 (after 1st revise) | 33.3% |
| Pass rate @ iter 2 (after 2nd revise) | 36.7% |
| Iteration distribution (1 / 2 / 3 calls) | 21 / 1 / 8 |
| Agent errors | 0 |
| Mean latency per agent run | 1.05 s |

**Where it fails.** The misses cluster into a few patterns, and only the last
one is something the current verify→revise loop can detect:

- **Value-encoding mismatches (largest bucket).** Correct logic, wrong literal:
  `gender='m'` vs gold `'M'` (financial); `element='Ca'` vs `'ca'` (toxicology);
  `department='Art and Design'` vs the stored `'Art and Design Department'`
  (student_club); `'Gladiator'/'banned'` vs `'gladiator'/'Banned'` (card_games).
  The query runs and returns plausible rows, so it looks fine.
- **BIRD "evidence" / domain knowledge not in the schema.** "Normal IgG" =
  `IGG BETWEEN 900 AND 2000`, "no eye colour" = `colour.id = 1`, crimes-1995 =
  column `A15`. The model can't infer these from `CREATE TABLE` text alone.
- **Output shape.** Wrong/extra columns (thrombosis `SELECT p.*` vs gold
  `DISTINCT ID`; codebase "well-finished" returned post columns instead of the
  IIF string), and one column-**order** mismatch (california address:
  Street,City,Zip,State vs gold Street,City,State,Zip) — semantically the same
  data, counted wrong by the position-sensitive comparison. A minor harness
  strictness note, not an agent bug.
- **Datetime trailing precision.** Stored timestamps carry a `.0`
  (`'…:08.0'`); exact-equality filters return 0 rows until switched to `LIKE`.

`screenshots/grafana_eval_run.png` shows the dashboard during the run (KV cache,
running/waiting, and token rate all reacting to the ~70-request burst).

## 3. Hitting the SLO (Phase 6)

_Pending — load test + iteration log._

## 4. Agent value — did the loop earn its keep?

**Barely, at current prompts.** The verify→revise loop moved overall accuracy
from **33.3% (iter 0) to 36.7% (iter 2)** — a net **+1 question (+3.3 pts)** —
at a real latency cost (3-iteration runs take ~2–3.5 s vs ~0.4 s for single-pass
ones). Verify accepted 21/30 questions on the first try but only 10 of those
were actually correct, so it has a **high false-accept rate**: it sees a count
or a name come back and calls it plausible, blind to wrong value-encodings that
silently return the wrong rows. Of the 9 questions that did revise, **only 1
recovered**; the other 8 spun to the cap re-emitting cosmetically different
queries (toggling `=`↔`LIKE`, flipping literal case) without diagnosing the real
defect. Tellingly, the single win was an *empty-result → `LIKE`* fix — exactly
the failure class verify can actually observe. On the upside there were **zero
regressions** (revise never broke a correct answer), so the loop is *safe* — it
just doesn't yet have the signal it needs to be *effective*. To make it earn its
keep, verify needs grounding it currently lacks: e.g. show it the distinct
values of filtered columns (catches `'m'` vs `'M'`), assert the result shape
against what the question asks, and feed BIRD evidence into the generate prompt.
That is the highest-leverage quality work, ahead of any further serving tuning.

---
