"""The triage task: schema, prompt, and the golden dataset it is scored on.

The agent reads a support ticket and returns a structured triage. The golden
dataset (data/golden.jsonl) is synthetic and labelled by construction: each
ticket is generated to fit a known category, priority, and needs_human flag, so
a candidate model can be scored without a human-labelled reference set.
"""
import json
from pathlib import Path

CATEGORIES = ["billing", "bug", "feature_request", "account", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]

SYSTEM = (
    "You are a support-triage agent. For each ticket, respond with only a JSON "
    "object, no reasoning and no code fence, with these keys:\n"
    "- category: one of billing, bug, feature_request, account, other.\n"
    "- priority: one of low, medium, high, urgent. Urgent means a service is down "
    "or many users are blocked. High means a broken feature or a money or access "
    "problem for one customer. Medium means degraded but usable. Low means a "
    "question, feedback, or a request.\n"
    "- needs_human: true only when category is billing, bug, or account and "
    "priority is medium, high, or urgent. Otherwise false.\n"
    "- summary: one sentence, at most 20 words.\n"
    "Return the JSON object only."
)

# response_format value. Doubleword honors this server-side, so the candidate is
# constrained to valid categories, priorities, and JSON.
TRIAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "priority": {"type": "string", "enum": PRIORITIES},
                "needs_human": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["category", "priority", "needs_human", "summary"],
            "additionalProperties": False,
        },
    },
}

GOLDEN_PATH = Path("data/golden.jsonl")


def load_golden(path=GOLDEN_PATH):
    """Return the golden tickets as a list of dicts with text and gold labels."""
    return [json.loads(line) for line in open(path) if line.strip()]


def parse_triage(content):
    """Parse a model's triage output, tolerating a stray code fence."""
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:].strip() if s.lower().startswith("json") else s.strip()
    return json.loads(s)
