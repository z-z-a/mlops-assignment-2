"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question, capturing correctness at every iteration."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    # Gold result once (the ground truth row set).
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # Ask the agent. Tag the run so it's findable in Langfuse (Phase 4/6).
    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={
                "question": q_text,
                "db": db_id,
                "tags": {"tags": "phase5,baseline", "run_id": "eval-baseline"},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        agent_error = data.get("error")
    except Exception as e:  # noqa: BLE001
        return {
            "db_id": db_id, "question": q_text, "gold_sql": gold_sql,
            "gold_ok": gold_ok, "gold_error": gold_err,
            "final_sql": "", "candidates": [], "num_iterations": 0,
            "correct_per_iter": [], "final_correct": False,
            "agent_error": f"{type(e).__name__}: {e}",
            "latency_s": time.monotonic() - t0,
        }
    latency = time.monotonic() - t0

    # Ordered SQL candidates, one per generate/revise step.
    history = data.get("history", [])
    candidates = [h["sql"] for h in history if "sql" in h]
    if not candidates and data.get("sql"):
        candidates = [data["sql"]]  # fallback if history is empty

    # Score each candidate by executed row set against gold.
    correct_per_iter: list[bool] = []
    for sql in candidates:
        ok, rows, _ = run_sql(db_id, sql)
        correct_per_iter.append(bool(ok and gold_ok and matches(gold_rows, rows)))

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "final_sql": candidates[-1] if candidates else "",
        "candidates": candidates,
        "num_iterations": len(candidates),
        "correct_per_iter": correct_per_iter,
        "final_correct": correct_per_iter[-1] if correct_per_iter else False,
        "agent_ok": data.get("ok"),
        "agent_error": agent_error,
        "latency_s": latency,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    overall = sum(1 for r in results if r["final_correct"]) / n

    max_iters = max((r["num_iterations"] for r in results), default=0)
    pass_rate_by_iteration = []
    for k in range(max_iters):
        correct = 0
        for r in results:
            cpi = r["correct_per_iter"]
            if not cpi:
                continue  # agent error -> incorrect at every iteration
            idx = k if k < len(cpi) else len(cpi) - 1  # carry last forward
            correct += int(cpi[idx])
        pass_rate_by_iteration.append({"iteration": k, "pass_rate": correct / n})

    iter_dist: dict[int, int] = {}
    for r in results:
        iter_dist[r["num_iterations"]] = iter_dist.get(r["num_iterations"], 0) + 1

    return {
        "n": n,
        "overall_pass_rate": round(overall, 4),
        "pass_rate_by_iteration": pass_rate_by_iteration,
        "iteration_distribution": dict(sorted(iter_dist.items())),
        "agent_errors": sum(1 for r in results if r.get("agent_error")),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / n, 3),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
