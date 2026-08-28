"""Build a model-agnostic batch file from the golden dataset.

Each ticket becomes a /v1/chat/completions request with the triage schema pinned
but no model set. Stamp the model per candidate with `dw files prepare --model`,
so the incumbent and every candidate run off the same prepared file.
"""
import json
from pathlib import Path

from .golden import SYSTEM, TRIAGE_SCHEMA, load_golden

# Reasoning models spend part of the budget thinking before the JSON. Give them
# headroom, or they hit finish_reason=length and return empty content. A large
# reasoning incumbent can want a few thousand tokens on a hard ticket.
DEFAULT_MAX_TOKENS = 4096


def build(output="batches/batch.jsonl", max_tokens=DEFAULT_MAX_TOKENS, reasoning_effort="minimal"):
    tickets = load_golden()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in tickets:
            body = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": t["text"]},
                ],
                "response_format": TRIAGE_SCHEMA,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            # Triage is a classification task, not a reasoning one. Minimal effort
            # keeps reasoning models from spending output tokens thinking, which is
            # what makes the open candidate cheap enough to matter.
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            request = {"custom_id": t["id"], "method": "POST", "url": "/v1/chat/completions", "body": body}
            f.write(json.dumps(request) + "\n")
    return len(tickets), str(out)
