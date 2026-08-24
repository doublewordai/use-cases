"""Generate the synthetic golden dataset.

Each ticket is seeded with a category, priority, and needs_human flag, then a
model writes ticket text to match. The seed is the gold label, so scoring needs
no human-labelled reference. Output is data/golden.jsonl, which ships frozen.
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

# (scenario, priority, needs_human) seeds per category. Category is guaranteed by
# the scenario; priority and needs_human are written to match the seed.
SEEDS = {
    "billing": [
        ("charged twice for the same subscription", "high", True),
        ("wrong VAT rate on the latest invoice", "medium", True),
        ("wants a refund after cancelling", "medium", True),
        ("card declined but still got a receipt", "high", True),
        ("asking which plan they are on", "low", False),
        ("annual invoice PDF will not download", "low", False),
        ("upgrade did not apply the new price", "medium", True),
        ("disputing a charge they do not recognise", "high", True),
    ],
    "bug": [
        ("export button returns a 500 error", "high", True),
        ("dashboard chart renders blank", "medium", True),
        ("mobile app crashes on the settings screen", "high", True),
        ("search returns no results for a known term", "medium", True),
        ("timezone shown is off by an hour", "low", False),
        ("CSV import silently drops rows", "high", True),
        ("notification emails arrive twice", "low", False),
        ("API returns 403 for every key since this morning", "urgent", True),
    ],
    "feature_request": [
        ("would like a dark mode", "low", False),
        ("wants CSV export on the reports page", "low", False),
        ("asking for a Slack integration", "medium", False),
        ("requests bulk editing of records", "low", False),
        ("wants two-factor authentication options", "medium", False),
        ("suggests keyboard shortcuts", "low", False),
        ("wants a public API for their workflow", "medium", False),
        ("asks for more granular permissions", "low", False),
    ],
    "account": [
        ("locked out after changing their email", "high", True),
        ("cannot reset their password", "medium", True),
        ("wants to transfer ownership to a colleague", "medium", True),
        ("needs to add a teammate to the workspace", "low", False),
        ("asking to delete their account and data", "medium", True),
        ("two-factor device lost, needs to regain access", "high", True),
        ("wants to change their username", "low", False),
        ("single sign-on stopped working for the team", "urgent", True),
    ],
    "other": [
        ("just says thanks, the product is great", "low", False),
        ("asking when the next webinar is", "low", False),
        ("wants a case study to share internally", "low", False),
        ("general question about the roadmap", "low", False),
        ("asking if there is a student discount", "low", False),
        ("feedback that onboarding felt smooth", "low", False),
        ("wants merch or stickers", "low", False),
        ("asking where to follow product updates", "low", False),
    ],
}

PER_CATEGORY = 30


def _seed_rows():
    rows = []
    rnd = random.Random(7)
    for category, seeds in SEEDS.items():
        for i in range(PER_CATEGORY):
            scenario, priority, needs_human = rnd.choice(seeds)
            rows.append({
                "id": f"{category[:4]}-{i:02d}",
                "category": category,
                "priority": priority,
                "needs_human": needs_human,
                "scenario": scenario,
            })
    return rows


def generate(model, out=Path("data/golden.jsonl")):
    client = OpenAI(api_key=os.environ["DOUBLEWORD_API_KEY"], base_url="https://api.doubleword.ai/v1")
    rows = _seed_rows()

    def write_ticket(row):
        escalation = "needs a human agent" if row["needs_human"] else "can be handled without a human"
        prompt = (
            f"Write a realistic one to three sentence customer support ticket. "
            f"The customer {row['scenario']}. It should read as {row['priority']} priority and {escalation}. "
            f"Write only the ticket text, first person, no subject line, no greeting."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.9,
        )
        return (resp.choices[0].message.content or "").strip().strip('"')

    with ThreadPoolExecutor(max_workers=16) as pool:
        texts = list(pool.map(write_ticket, rows))

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out, "w") as f:
        for row, text in zip(rows, texts):
            if not text:
                continue
            f.write(json.dumps({
                "id": row["id"],
                "text": text,
                "category": row["category"],
                "priority": row["priority"],
                "needs_human": row["needs_human"],
            }) + "\n")
            kept += 1
    return kept, str(out)
