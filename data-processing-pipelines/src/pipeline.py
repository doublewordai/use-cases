"""Data download and processing functions for the pipeline.

Downloads real company data from SEC EDGAR (public, no API key required)
and processes it through normalize -> enrich -> deduplicate stages.
"""

import csv
import json
import re

import requests
from thefuzz import fuzz

from .prompts import DEDUP_PROMPT, ENRICH_PROMPT, NORMALIZE_PROMPT

SEC_EDGAR_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_USER_AGENT = "DoublewordAI research@doubleword.ai"


def download_sec_data(limit: int = 500) -> list[dict]:
    """Download company records from SEC EDGAR.

    The SEC EDGAR company tickers dataset contains ~10,000 public company
    records with naturally inconsistent formatting (mixed casing, varied
    abbreviations, different legal suffix styles).

    Args:
        limit: Maximum number of records to download (default 500)

    Returns:
        List of company record dicts
    """
    response = requests.get(
        SEC_EDGAR_URL,
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    fields = data["fields"]  # ['cik', 'name', 'ticker', 'exchange']
    rows = data["data"][:limit]

    records = []
    for i, row in enumerate(rows):
        record = {field: row[j] for j, field in enumerate(fields)}
        record["id"] = f"sec-{i:06d}"
        record["cik"] = str(record["cik"])
        records.append(record)

    return records


def load_csv_data(path: str, limit: int = 500) -> list[dict]:
    """Load company records from a user-provided CSV file.

    The CSV should have at least a 'name' column. Other columns are preserved
    as-is in the record.

    Args:
        path: Path to CSV file
        limit: Maximum records to load

    Returns:
        List of company record dicts
    """
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            record = dict(row)
            if "id" not in record:
                record["id"] = f"csv-{i:06d}"
            if "name" not in record:
                # Try common column names
                for col in ["company_name", "company", "Name", "Company"]:
                    if col in record:
                        record["name"] = record[col]
                        break
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")
    if "name" not in records[0]:
        raise ValueError(
            f"CSV must have a 'name' column (or 'company_name', 'company'). "
            f"Found columns: {list(records[0].keys())}"
        )

    return records


def build_normalize_requests(records: list[dict], model: str) -> list[dict]:
    """Build batch requests for the normalization stage."""
    requests_data = []
    for record in records:
        user_content = f"Company name: {record['name']}"
        if record.get("ticker"):
            user_content += f"\nTicker: {record['ticker']}"
        if record.get("exchange"):
            user_content += f"\nExchange: {record['exchange']}"
        if record.get("address"):
            user_content += f"\nAddress: {record['address']}"

        requests_data.append(
            {
                "custom_id": f"normalize-{record['id']}",
                "model": model,
                "messages": [
                    {"role": "system", "content": NORMALIZE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 512,
                "temperature": 0,
            }
        )
    return requests_data


def build_enrich_requests(records: list[dict], model: str) -> list[dict]:
    """Build batch requests for the enrichment stage."""
    requests_data = []
    for record in records:
        name = record.get("normalized_name", record["name"])
        user_content = f"Company name: {name}"
        if record.get("ticker"):
            user_content += f"\nTicker: {record['ticker']}"
        if record.get("exchange"):
            user_content += f"\nExchange: {record['exchange']}"
        if record.get("city"):
            user_content += f"\nLocation: {record['city']}, {record.get('state', '')}"
        if record.get("website"):
            user_content += f"\nWebsite: {record['website']}"

        requests_data.append(
            {
                "custom_id": f"enrich-{record['id']}",
                "model": model,
                "messages": [
                    {"role": "system", "content": ENRICH_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 512,
                "temperature": 0,
            }
        )
    return requests_data


def generate_dedup_candidates(
    records: list[dict],
    threshold: int = 70,
) -> list[tuple]:
    """Use fuzzy matching to find candidate duplicate pairs above threshold."""
    candidates = []
    names = [(i, r.get("normalized_name", r["name"])) for i, r in enumerate(records)]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = fuzz.ratio(names[i][1].lower(), names[j][1].lower())
            if score >= threshold:
                candidates.append(
                    (
                        records[names[i][0]],
                        records[names[j][0]],
                        score,
                    )
                )

    return candidates


def build_dedup_requests(pairs: list[tuple], model: str) -> list[dict]:
    """Build batch requests for the deduplication stage."""
    requests_data = []
    for idx, (rec_a, rec_b, score) in enumerate(pairs):
        user_content = (
            f"Record A:\n  Name: {rec_a.get('normalized_name', rec_a['name'])}"
        )
        if rec_a.get("ticker"):
            user_content += f"\n  Ticker: {rec_a['ticker']}"
        if rec_a.get("city"):
            user_content += f"\n  Location: {rec_a['city']}, {rec_a.get('state', '')}"

        user_content += (
            f"\n\nRecord B:\n  Name: {rec_b.get('normalized_name', rec_b['name'])}"
        )
        if rec_b.get("ticker"):
            user_content += f"\n  Ticker: {rec_b['ticker']}"
        if rec_b.get("city"):
            user_content += f"\n  Location: {rec_b['city']}, {rec_b.get('state', '')}"

        user_content += f"\n\nFuzzy match score: {score}/100"

        requests_data.append(
            {
                "custom_id": f"dedup-{idx:06d}",
                "model": model,
                "messages": [
                    {"role": "system", "content": DEDUP_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 256,
                "temperature": 0,
            }
        )
    return requests_data


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
