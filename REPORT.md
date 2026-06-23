# Report: LLM inference + o11y
By Zvi Amolsky

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

> **SLO:** P95 end-to-end agent latency < 5 s at 10+ RPS over a 5-minute window.

Load generated with `load_test/driver.py` (open-loop, samples
`load_test/perf_pool.jsonl`). The driver's `latency_p*` is the SLO metric (full
`/answer` round trip); Grafana / `/metrics` gauges are the per-vLLM-call
diagnosis lens. Note: driver percentiles are computed over **`ok` requests
only**, so timeouts/errors are excluded — the true tail is worse than the
reported p95 whenever failures are non-trivial.

### Run log

| Run | Config change | ok / 3000 | timeouts | http 500 | conn err | p50 | p95 (ok) | p99 (ok) | vLLM during run | SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1 uvicorn worker, sync endpoint, MAX_ITER=3 | 1159 | 998 | 227 | 616 | 9.6s | 70s | 74s | KV 3%, 0 preempt, 0 waiting (idle) | ❌ hard miss |
| Iter 1 | **8 workers** (client still built per call) | 2609 | 4 | 383 | 4 | 4.1s | 86s | 100s | running 29, KV 5%, 0 preempt, 0 waiting | ❌ miss |
| Iter 2 | + **shared LLM client** (`@lru_cache`) | 2617 | 2 | 381 | 0 | **1.4s** | **7.8s** | 13.5s | running 28, KV 3%, 0 preempt, 0 waiting | ❌ miss (close) |
| Iter 3 | + **schema NULL-FK fix** (`european_football_2`) | 2989 | 5 | **2** | 4 | 1.5s | 8.3s | 14.2s | running 36, KV 3%, 0 preempt, 0 waiting | ❌ miss (close) |
| Iter 4 | + **MAX_ITERATIONS 3→2** | 2979 | 2 | 3 | 16 | 1.4s | 7.4s | 16.1s | running 27, KV 3%, 0 preempt, 0 waiting | ❌ miss (close) |
| Iter 5 | + **24 workers** (8→24; oversubscribe, CPU ~60% idle) | 2994 | 1 | 4 | 1 | 1.2s | **4.38s** | 6.99s | running 31, KV 5%, 0 preempt, 0 waiting | ✅ **HIT** |
| **Final** | MAX=2, 24 workers, **+ schema example values** | 2981 | 7 | 11 | 1 | 1.3s | **4.67s** | 7.6s | running 24, KV 6%, 0 preempt, 0 waiting | ✅ **HIT** |

**Final config: FP8 vLLM (Phase 1 flags) + 24 uvicorn workers + shared LLM
client + schema NULL-FK fix + column example values + `MAX_ITERATIONS=2`.**
SLO met (p95 **4.67 s** < 5 s @ 10 RPS over 5 min) *and* accuracy **43.3%** (up
from the 36.7% baseline). The annotated schema added ~0.3 s of p95 (more prefill
tokens) but stayed well under budget — vLLM remained idle throughout, so the
serving layer was never the constraint. (HTTP errors 11/3000 ≈ 0.4%, negligible.)

### Iterations (saw → hypothesized → changed → result)

**Baseline.** 10 RPS / 300 s → p50 **9.6 s**, 61% of requests failed (998
timeouts + 227 HTTP 500 + 616 connection errors). True p95 ≥120 s (timeouts
excluded from the reported 70 s). **SLO: hard miss.**

**Iteration 1.**
- *Saw:* total queueing collapse while vLLM sat at **3% KV, 0 preemptions, 0
  waiting** — vLLM bored throughout; 616 connection errors against a single
  agent process.
- *Hypothesized:* the bottleneck is the **agent server**, not serving — a single
  sync `uvicorn` worker whose blocking `graph.invoke` runs in FastAPI's ~40-thread
  pool under one GIL; 616 connection errors against one process.
- *Changed:* ran `uvicorn … --workers 8` (no other change).
- *Result:* connection collapse **fixed** — timeouts 998→4, connection errors
  616→4, ok 1159→2609, p50 9.6→**4.1 s**. **But still a miss:** p95 **86 s**,
  and HTTP 500s rose to 383. vLLM remained idle (**running 29, KV 5%, 0
  preempt**). With 8×40=320 thread slots but only ~29 concurrent vLLM calls
  (~3.6/worker), the agent is now **CPU/GIL-bound on per-call framework
  overhead**, not I/O or serving. Service rate still < 10 runs/s → open-loop
  pile-up → 86 s tail.

**Iteration 2.**
- *Saw:* per-worker concurrency capped at ~3.6 with a fresh `ChatOpenAI` (and a
  new httpx connection pool + TCP/TLS) constructed on every node call.
- *Hypothesized:* per-call client construction is avoidable per-request overhead.
- *Changed:* cached the `ChatOpenAI` client with `@lru_cache(maxsize=1)` so the
  pool is reused across calls (per worker).
- *Result:* **dominant fix.** p50 4.1→**1.4 s**, p95 86→**7.8 s** (~11×), p99
  100→13.5 s, connection errors → 0. This *refuted* the Iteration-1 "CPU/GIL-bound
  framework overhead" read: the real cost was rebuilding the client/pool per call,
  not the worker count. vLLM still idle (running 21, **KV 3%, 0 preempt**) — the
  remaining latency is still agent-side, not serving. **SLO still missed but
  close: p95 7.8 s vs 5 s target.** Two open issues: (1) HTTP 500s persist at
  **381 (~12.7%)**, essentially unchanged from Iter 1 — a separate, input-dependent
  bug (driver seeds RNG, so the same questions fail each run); (2) a long tail
  (max 111 s) likely driven by those failing requests retrying before they 500.

**Iteration 3.**
- *Saw:* the 381 HTTP 500s were all `european_football_2` →
  `AttributeError: 'NoneType' … 'replace'`, deterministic per DB. Root cause: the
  provided `render_schema` calls `_q(fk[4])`, but SQLite's `PRAGMA
  foreign_key_list` returns `NULL` for the "to" column when a FK references the
  parent's PK implicitly — which this DB does. (Not context length; the eval set
  has no `european_football_2`, so § 2 saw 0 errors.)
- *Hypothesized:* fixing the crash removes 12.7% hard failures *and* the
  retry-storms they caused (openai client retries 2× across 2–3 nodes before
  throwing), which were inflating the tail.
- *Changed:* `agent/schema.py` — render the FK without the column when the target
  is `None`.
- *Result:* HTTP 500s **381→2**, ok 2617→**2989** (99.6%). `max` latency 111→73 s
  (retry outliers gone). p95 ticked **7.8→8.3 s** because the formerly-failing
  requests now run and add real load. Still agent-bound (vLLM idle: running 36,
  KV 3%, 0 preempt); p50 1.5 s / p95 8.3 s is open-loop pile-up — service rate
  ~8–9 runs/s < 10 RPS arrival. **SLO still missed (close).**

**Iteration 4.**
- *Saw:* a persistent ceiling — every iteration since the shared-client fix
  plateaus at ~8–9 served runs/s with vLLM idle (KV ~3%, 0 preempt), regardless
  of per-run trimming.
- *Hypothesized:* the 3rd iteration's 6-call runs are part of the tail; removing
  them should help p95.
- *Changed:* `MAX_ITERATIONS` 3→2.
- *Result:* p95 8.3→**7.4 s** (small gain), but p99 14.2→16.1 s and max 73→114 s
  — noisy tail, throughput still ~8.7 runs/s. **SLO still missed.** Diminishing
  returns confirm the bottleneck is not work-per-run but the **agent's fixed
  service capacity**: vLLM has ~30× headroom while the LangGraph/LangChain Python
  path (sync threadpool across 8 workers) caps throughput below the 10-RPS arrival
  rate → open-loop backlog → tail.

**Iteration 5.**
- *Saw:* throughput plateaued at ~8–9 runs/s with vLLM idle, but the agent box
  (`nproc` = **16**) ran at only **~60% CPU idle** under load — *not* CPU-bound.
  Each agent run is I/O-bound (mostly waiting on vLLM with the GIL released), so
  worker processes ≈ cores can't keep the box busy; throughput is
  concurrency-limited, not compute-limited.
- *Hypothesized:* oversubscribing workers past the core count raises in-flight
  concurrency, drains the open-loop backlog, and lowers p95 — and there's ample
  CPU headroom to absorb it.
- *Changed:* `uvicorn … --workers 24` (up from 8).
- *Result:* **SLO HIT.** p95 7.4→**4.38 s** at 10 RPS over 5 min, p99 6.99 s,
  p50 1.2 s, **timeouts → 0**, 99.8% ok. running 24, KV 5% (still ~30× KV
  headroom). Oversubscribing gave the concurrency the I/O-bound runs needed
  without touching CPU.

### Verdict

**SLO met: P95 4.67 s < 5 s at 10 RPS over a 5-minute window, ~99% success**
(baseline was full collapse — p50 9.6 s, 61% failures). The SLO was first cleared
at p95 4.38 s (Iter 5); the final config's annotated schema added ~0.3 s of
prefill but stayed well under budget. The entire gap was the **agent serving
layer, never the model or vLLM** — vLLM sat at ~6% KV with 0 preemptions through
every run. Changes that moved the needle, in order of impact:

1. **Shared `ChatOpenAI` client** (`@lru_cache`) — ~11× p95 win (86→7.8 s);
   per-call client construction was the dominant cost.
2. **Scale uvicorn workers 1→24** — fixed connection collapse, then concurrency
   (runs are I/O-bound on vLLM, so oversubscribing past the 16 cores paid off;
   CPU stayed ~60% idle).
3. **Schema NULL-FK crash fix** — removed a 12.7% hard-failure rate
   (`european_football_2`).
4. **MAX_ITERATIONS 3→2** — trimmed the multi-call tail to fit the latency
   budget, at a quantified cost of one eval question

The quality impact of the `MAX_ITERATIONS=2` cut — and the change that actually
moved accuracy — are analysed in § 4.

> Root-cause arc of Phase 6: the SLO miss was **never the 30B model or vLLM**
> (idle at 3% KV throughout) — it was the agent serving layer: per-call client
> construction (Iter 2, the 11× win), a schema crash (Iter 3), and GIL-bound
> under-provisioned workers (Iter 4–5). vLLM's Phase 1 config was right all along.

_Screenshots: `screenshots/grafana_before.png` (baseline collapse),
`grafana_after.png` (final healthy run)._

## 4. Agent value — did the loop earn its keep?

**Marginally — and the schema beat it.** The verify→revise loop moved baseline
accuracy from **33.3% (iter 0) to 36.7% (iter 2)** — net **+1 question** — and the
gain is fragile. Verify accepted 21/30 on the first try but only 10 were actually
correct (**high false-accept rate**): it sees a count or a name and calls it
plausible, blind to wrong value-encodings that silently return wrong rows. Of the
9 revising questions, **only 1 recovered** (an *empty-result → `LIKE`* fix — the
one failure class verify can actually observe); the other 8 spun to the cap
re-emitting cosmetic variants (toggling `=`↔`LIKE`, flipping literal case). There
were **zero regressions**, so the loop is *safe* but not *effective* — and under
load its 3rd iteration adds tail latency the SLO budget can't afford, so the final
config caps it at 2, costing that one question (36.7%→33.3%).

**The accuracy lever was the schema, not the loop.** The dominant failure mode
(§ 2) was value-encoding — the model guessing string literals from column
names/types alone. Annotating each column in `render_schema` with 3 real example
values (`-- e.g. 'M', 'F'`) lets it copy the exact stored literal and pick the
right column. Holding the SLO-compliant `MAX_ITERATIONS=2`:

| Config | Pass rate | Iterations (1 / 2) | Mean latency |
|---|---|---|---|
| bare schema | 33.3% (10/30) | 21 / 9 | 0.91 s |
| **+ column example values** | **43.3% (13/30)** | **26 / 4** | **0.80 s** |

That's **+3 questions (+10 pts)** — more than recovering the SLO-cut question and
beating the 36.7% baseline. It is also *latency-friendly*: better first attempts
mean fewer revises fire (mean latency dropped), and the per-DB schema prefix is a
vLLM prefix-cache hit, so the extra tokens are near-free. **Conclusion: at this
prompt/model scale, content grounding in generation beats the verify→revise loop
for accuracy by a wide margin** (+3 free vs +1 at the cost of the SLO).
`results/eval_after_tuning.json` is the 43.3% final-config run.

## 5. What I'd do with more time

- **Ground the verifier like generation.** Give `verify` the same distinct-column
  values plus an explicit result-shape check (expected column count/types vs the
  question). It would catch the residual value-encoding and `SELECT *`/wrong-column
  misses instead of rubber-stamping plausible-but-wrong rows.
- **Inject BIRD "evidence" into the generate prompt.** Misses like "normal IgG =
  900–2000", "no eye colour = `colour.id` 1", "crimes-1995 = column `A15`" are
  domain mappings not inferable from `CREATE TABLE`. A small per-DB evidence file
  (or retrieved hint) would unlock the `thrombosis`/`superhero`/`financial` cases.
- **Smarter example-value sampling.** Skip free-text / high-cardinality / ID
  columns (no literal to match), prefer low-cardinality categoricals, optionally
  show value counts — keeps the prompt lean while maximizing signal.
- **Async agent endpoint.** Runs are I/O-bound on vLLM, yet the sync endpoint
  needs 24 worker processes to reach 10 RPS. `async def` + `graph.ainvoke` + async
  nodes would hit the same concurrency in one process — less memory, no GIL
  fan-out, headroom well past 10 RPS.
- **Afford a smarter 3rd iteration.** The cap to 2 was forced by latency. A
  cheaper verify (guided JSON, a small draft model, or short-circuiting hard
  errors without an LLM call) could fit a 3rd iteration in budget and recover the
  lost question without breaking the SLO.

---
