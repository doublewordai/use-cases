"""Ensemble voting logic for structured extraction."""

import json
import re
from collections import Counter
from typing import Any, Optional

from .schema import EVAL_FIELDS

# Fields that are numeric
NUMERIC_FIELDS = {"total", "subtotal", "tax"}


def normalize_string(s: Optional[str]) -> Optional[str]:
    """Normalize a string for comparison."""
    if s is None:
        return None
    # Lowercase, strip whitespace, remove extra spaces
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_number(n: Any, precision: int = 2) -> Optional[float]:
    """Normalize a number for comparison, handling currency symbols."""
    if n is None:
        return None
    if isinstance(n, (int, float)):
        return round(float(n), precision)
    if isinstance(n, str):
        # Remove whitespace and currency symbols
        n = n.strip()
        if not n:
            return None
        # Strip common currency prefixes/suffixes (USD, MYR, etc.)
        n = re.sub(r"^[$£€¥₹]|RM\s*|USD\s*|MYR\s*", "", n, flags=re.IGNORECASE)
        n = n.replace(",", "").strip()
        if not n:
            return None
        try:
            return round(float(n), precision)
        except ValueError:
            return None
    return None


def normalize_date(d: Optional[str]) -> Optional[str]:
    """Normalize date for comparison.

    Extracts day, month, year components and returns canonical YYYY-MM-DD format.
    Handles ambiguous formats by checking if values are valid days/months.
    """
    if d is None:
        return None

    d = d.strip()

    # Try YYYY-MM-DD first (unambiguous ISO format)
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", d)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{month}-{day}"

    # Try to extract components from ambiguous formats
    # Matches: DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, DD.MM.YYYY, etc.
    match = re.match(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})", d)
    if match:
        a, b, year_str = match.groups()
        a, b = int(a), int(b)

        # Expand 2-digit year
        if len(year_str) == 2:
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        else:
            year = int(year_str)

        # Disambiguate day/month:
        # - If a > 12, it must be the day (DD/MM format)
        # - If b > 12, it must be the day (MM/DD format)
        # - If both <= 12, assume DD/MM (more common globally, matches SROIE)
        if a > 12:
            day, month = a, b
        elif b > 12:
            month, day = a, b
        else:
            # Ambiguous - default to DD/MM (common in SROIE dataset)
            day, month = a, b

        return f"{year:04d}-{month:02d}-{day:02d}"

    # If no pattern matches, return lowercase stripped version
    return d.lower().strip()


def normalize_value(field: str, value: Any) -> Any:
    """Normalize a value based on field type."""
    if value is None:
        return None

    if field in ("total", "subtotal", "tax"):
        return normalize_number(value)
    elif field == "date":
        return normalize_date(value)
    elif field in ("vendor_name", "vendor_address", "payment_method"):
        return normalize_string(value)
    else:
        return value


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the text
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"(\{.*\})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def vote_on_field(values: list[Any], field: str) -> tuple[Any, float, float]:
    """
    Vote on a single field across multiple extractions.

    Returns (consensus_value, agreement_rate, extraction_rate):
        - agreement_rate: What fraction of non-null extractions agree on the consensus value
        - extraction_rate: What fraction of runs successfully extracted this field (non-null)
    """
    # Normalize all values
    normalized = [normalize_value(field, v) for v in values]

    # Filter out None values for voting
    non_null = [v for v in normalized if v is not None]

    if not non_null:
        return None, 0.0, 0.0

    extraction_rate = len(non_null) / len(values)

    # For numbers, we need to handle floating point comparison
    if field in ("total", "subtotal", "tax"):
        # Round to 2 decimal places and count
        rounded = [round(v, 2) for v in non_null]
        counter = Counter(rounded)
    else:
        counter = Counter(non_null)

    # Get most common value
    most_common, count = counter.most_common(1)[0]
    agreement_rate = count / len(non_null)  # Agreement among successful extractions

    return most_common, agreement_rate, extraction_rate


def ensemble_vote(extractions: list[dict]) -> tuple[dict, dict[str, dict]]:
    """
    Perform ensemble voting across multiple extractions.

    Returns (consensus_extraction, field_stats) where field_stats contains:
        - agreement_rate: Fraction of non-null values that match consensus
        - extraction_rate: Fraction of runs that extracted this field
    """
    if not extractions:
        return {}, {}

    consensus = {}
    field_stats = {}

    for field in EVAL_FIELDS:
        values = [e.get(field) for e in extractions]
        consensus_value, agreement_rate, extraction_rate = vote_on_field(values, field)
        consensus[field] = consensus_value
        field_stats[field] = {
            "agreement_rate": agreement_rate,
            "extraction_rate": extraction_rate,
        }

    return consensus, field_stats


def calculate_field_accuracy(
    predicted: dict, ground_truth: dict, fields: list[str] = None
) -> dict[str, bool]:
    """
    Calculate per-field accuracy.
    Returns dict mapping field -> is_correct.
    """
    if fields is None:
        fields = EVAL_FIELDS

    results = {}
    for field in fields:
        pred_val = normalize_value(field, predicted.get(field))
        true_val = normalize_value(field, ground_truth.get(field))

        # Both None counts as correct
        if pred_val is None and true_val is None:
            results[field] = True
        # One None and one not is incorrect
        elif pred_val is None or true_val is None:
            results[field] = False
        # Compare normalized values
        else:
            results[field] = pred_val == true_val

    return results
