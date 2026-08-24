"""Phase 3: shadow the candidate against the incumbent on live calls.

Doubleword does not mirror traffic. This orchestrates it. The incumbent answers
on the hot path, which is what a user gets, and the candidate runs on the async
(flex) tier from a background worker, so the shadow call adds no user-facing
latency. Both are logged for offline comparison. If a Phoenix collector is
reachable the calls are traced there too, and if not the demo still runs.

The incumbent defaults to the closed flagship on OpenAI (gpt-5.6-sol). Point
FLAGSHIP_BASE_URL / FLAGSHIP_API_KEY / FLAGSHIP_MODEL at a different provider to
shadow against that instead.
"""
import json
import os
import threading
from pathlib import Path

from openai import BadRequestError, OpenAI

from .golden import SYSTEM, TRIAGE_SCHEMA, load_golden

try:
    if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
        from phoenix.otel import register

        register(project_name="agent-migration-shadow", auto_instrument=True)
except Exception:
    pass


def _messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def run(incumbent="gpt-5.6-sol", candidate="moonshotai/kimi-k3", n=5, out="results/shadow_log.jsonl"):
    flagship_key_env = "FLAGSHIP_API_KEY" if os.environ.get("FLAGSHIP_API_KEY") else "OPENAI_API_KEY"
    flagship = OpenAI(api_key=os.environ[flagship_key_env], base_url=os.environ.get("FLAGSHIP_BASE_URL") or None)
    incumbent_model = os.environ.get("FLAGSHIP_MODEL", incumbent)
    doubleword = OpenAI(api_key=os.environ["DOUBLEWORD_API_KEY"], base_url="https://api.doubleword.ai/v1")

    tickets = load_golden()[:n]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    open(out, "w").close()
    lock = threading.Lock()

    def hot_answer(text):
        kwargs = dict(model=incumbent_model, messages=_messages(text), response_format=TRIAGE_SCHEMA)
        try:
            resp = flagship.chat.completions.create(reasoning_effort="high", max_completion_tokens=8192, **kwargs)
        except BadRequestError:
            resp = flagship.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def shadow_call(ticket, incumbent_answer):
        result = doubleword.chat.completions.create(
            model=candidate,
            messages=_messages(ticket["text"]),
            response_format=TRIAGE_SCHEMA,
            extra_body={"service_tier": "flex"},
            max_tokens=4096,
            temperature=0,
        )
        with lock, open(out, "a") as f:
            f.write(json.dumps({
                "id": ticket["id"],
                "incumbent_model": incumbent_model,
                "incumbent": incumbent_answer,
                "candidate_model": candidate,
                "candidate": result.choices[0].message.content,
            }) + "\n")

    workers = []
    for ticket in tickets:
        answer = hot_answer(ticket["text"])  # served to the user now
        worker = threading.Thread(target=shadow_call, args=(ticket, answer))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return n, out
