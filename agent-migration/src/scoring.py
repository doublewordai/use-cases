"""Score a batch results file against the golden labels.

Reports the KPIs that catch a bad migration: schema compliance, category
accuracy (the headline, since categories are labelled by construction), priority
and needs_human accuracy, and the token counts behind the cost. Handles the two
result shapes the dw CLI can emit and treats truncated or empty content as a miss.
"""
import json
from pathlib import Path

from .golden import CATEGORIES, PRIORITIES, load_golden, parse_triage


def _content(record):
    body = record.get("response_body") or record.get("response", {}).get("body", {})
    choices = body.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    usage = body.get("usage", {})
    return content, usage


def score(results_path, label=None):
    label = label or results_path
    gold = {t["id"]: t for t in load_golden()}

    outputs, in_tok, out_tok = {}, 0, 0
    for line in open(results_path):
        if not line.strip():
            continue
        record = json.loads(line)
        content, usage = _content(record)
        outputs[record["custom_id"]] = content
        in_tok += usage.get("prompt_tokens", 0)
        out_tok += usage.get("completion_tokens", 0)

    n = len(gold)
    schema_ok = enum_ok = category_ok = priority_ok = needs_human_ok = 0
    for tid, g in gold.items():
        try:
            obj = parse_triage(outputs.get(tid))
            if all(k in obj for k in ("category", "priority", "needs_human", "summary")):
                schema_ok += 1
            if obj.get("category") in CATEGORIES and obj.get("priority") in PRIORITIES:
                enum_ok += 1
            category_ok += obj.get("category") == g["category"]
            priority_ok += obj.get("priority") == g["priority"]
            needs_human_ok += bool(obj.get("needs_human")) == g["needs_human"]
        except Exception:
            pass  # unparseable or truncated counts as a miss on every metric

    metrics = {
        "label": label,
        "n": n,
        "schema_valid": schema_ok,
        "enum_valid": enum_ok,
        "category_accuracy": round(category_ok / n, 3),
        "priority_accuracy": round(priority_ok / n, 3),
        "needs_human_accuracy": round(needs_human_ok / n, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"scored-{Path(results_path).stem}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
