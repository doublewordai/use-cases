"""Phase 1 baseline against a closed model on OpenAI, realtime.

Runs the golden set through the OpenAI client concurrently and collects results
as they return. Realtime only, not OpenAI's batch or flex tier. Writes
results/incumbent.jsonl in the same shape score.py reads, and returns the token
counts and cost so you can put the closed model next to the open candidates.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import BadRequestError, OpenAI

from .golden import SYSTEM, TRIAGE_SCHEMA, load_golden


def run(model, out="results/incumbent.jsonl", concurrency=8, price_in=5.0, price_out=30.0, limit=None, reasoning_effort=None):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
    tickets = load_golden()
    if limit:
        tickets = tickets[:limit]

    def one(ticket):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket["text"]}]
        kwargs = dict(model=model, messages=messages, response_format=TRIAGE_SCHEMA)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        started = time.time()
        try:
            resp = client.chat.completions.create(max_completion_tokens=8192, **kwargs)
        except BadRequestError:
            resp = client.chat.completions.create(**kwargs)
        return ticket["id"], resp.model_dump(), time.time() - started

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    records, in_tok, out_tok, latencies = {}, 0, 0, []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, t): t for t in tickets}
        for fut in as_completed(futures):
            tid = futures[fut]["id"]
            try:
                tid, body, dt = fut.result()
                latencies.append(dt)
            except Exception as exc:
                body = {"choices": [{"message": {"content": None}}], "usage": {}, "error": str(exc)[:200]}
            records[tid] = body
            usage = body.get("usage") or {}
            in_tok += usage.get("prompt_tokens", 0)
            out_tok += usage.get("completion_tokens", 0)

    with open(out, "w") as f:
        for tid, body in records.items():
            f.write(json.dumps({"custom_id": tid, "response": {"body": body}}) + "\n")

    cost = in_tok * price_in / 1e6 + out_tok * price_out / 1e6
    avg_latency_ms = round(1000 * sum(latencies) / len(latencies)) if latencies else None
    return {
        "n": len(tickets),
        "out": out,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost": round(cost, 4),
        "avg_latency_ms": avg_latency_ms,
    }
