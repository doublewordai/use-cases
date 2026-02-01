"""
CWE Classification module.

Classifies vulnerable code snippets into CWE categories.
"""

import json
import sqlite3
from pathlib import Path

# Target CWE classes with descriptions
CWE_CLASSES = {
    "CWE-125": "Out-of-bounds Read - Reading memory outside allocated buffer",
    "CWE-787": "Out-of-bounds Write - Writing memory outside allocated buffer",
    "CWE-119": "Buffer Overflow - Operations on memory buffer without proper bounds",
    "CWE-190": "Integer Overflow - Arithmetic operation exceeds integer limits",
    "CWE-476": "NULL Pointer Dereference - Dereferencing a pointer that is NULL",
    "CWE-416": "Use After Free - Using memory after it has been freed",
}

CWE_LIST = list(CWE_CLASSES.keys())

CLASSIFICATION_PROMPT = """Analyze this C/C++ code for security vulnerabilities.

This code is known to contain a vulnerability. Classify which type it is.

Categories:
{categories}

Code:
```c
{code}
```

Respond with JSON: {{"cwe": "CWE-XXX", "confidence": "high/medium/low", "reasoning": "brief explanation"}}

You must choose one of: {cwe_list}"""


def format_classification_prompt(code: str) -> str:
    """Format the classification prompt with code."""
    categories = "\n".join(f"- {cwe}: {desc}" for cwe, desc in CWE_CLASSES.items())
    return CLASSIFICATION_PROMPT.format(
        categories=categories,
        code=code,
        cwe_list=", ".join(CWE_LIST)
    )


def load_classification_samples(
    db_path: str,
    max_per_cwe: int = 100,
    min_code_length: int = 100,
    max_code_length: int = 3000,
) -> list[dict]:
    """
    Load vulnerable code samples for CWE classification.

    Only loads samples from the target CWE classes.
    Returns balanced samples across classes.
    """
    from .preprocess import preprocess_code

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    samples = []

    for cwe in CWE_LIST:
        query = """
        SELECT DISTINCT
            m.method_change_id,
            m.name as method_name,
            m.code,
            f.programming_language,
            fixes.cve_id,
            cwe_class.cwe_id
        FROM method_change m
        JOIN file_change f ON m.file_change_id = f.file_change_id
        JOIN commits c ON f.hash = c.hash
        JOIN fixes ON c.hash = fixes.hash
        JOIN cwe_classification cwe_class ON fixes.cve_id = cwe_class.cve_id
        WHERE m.before_change = 'True'
          AND m.code IS NOT NULL
          AND LENGTH(m.code) >= ?
          AND LENGTH(m.code) <= ?
          AND LOWER(f.programming_language) IN ('c', 'c++', 'cpp')
          AND cwe_class.cwe_id = ?
        LIMIT ?
        """

        cursor = conn.execute(query, [min_code_length, max_code_length, cwe, max_per_cwe])

        for row in cursor:
            code = preprocess_code(row['code'], strip_comments=True)

            samples.append({
                "id": f"{row['cve_id']}_{row['method_change_id']}",
                "code": code,
                "cwe": row['cwe_id'],
                "cve_id": row['cve_id'],
                "method_name": row['method_name'],
            })

    conn.close()
    return samples


def parse_classification_response(content: str) -> dict:
    """Parse the model's classification response."""
    try:
        parsed = json.loads(content)
        cwe = parsed.get("cwe", "").upper()
        # Normalize CWE format
        if not cwe.startswith("CWE-"):
            cwe = f"CWE-{cwe.replace('CWE', '')}"
        return {
            "predicted_cwe": cwe if cwe in CWE_LIST else None,
            "confidence": parsed.get("confidence", "unknown"),
            "reasoning": parsed.get("reasoning", ""),
            "parse_error": None,
        }
    except json.JSONDecodeError as e:
        return {
            "predicted_cwe": None,
            "confidence": "unknown",
            "reasoning": "",
            "parse_error": str(e),
        }


def analyze_classification_results(
    results_path: Path,
    samples: list[dict],
    id_mapping: dict,
) -> dict:
    """Analyze classification results and compute metrics."""
    from collections import Counter

    # Load results
    results = {}
    with open(results_path) as f:
        for line in f:
            obj = json.loads(line)
            results[obj["custom_id"]] = obj

    # Build ground truth lookup
    ground_truth = {s["id"]: s["cwe"] for s in samples}
    reverse_mapping = {v: k for k, v in id_mapping.items()}

    # Analyze predictions
    correct = 0
    total = 0
    confusion = {}  # {actual: {predicted: count}}
    by_confidence = {"high": [], "medium": [], "low": [], "unknown": []}

    predictions = []

    for sample in samples:
        short_id = reverse_mapping.get(sample["id"])
        if not short_id or short_id not in results:
            continue

        result = results[short_id]
        actual_cwe = sample["cwe"]

        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            parsed = parse_classification_response(content)
            predicted_cwe = parsed["predicted_cwe"]
            confidence = parsed["confidence"]
        except (KeyError, TypeError):
            predicted_cwe = None
            confidence = "unknown"

        total += 1
        is_correct = (predicted_cwe == actual_cwe)
        if is_correct:
            correct += 1

        # Track confusion matrix
        if actual_cwe not in confusion:
            confusion[actual_cwe] = Counter()
        confusion[actual_cwe][predicted_cwe or "NONE"] += 1

        # Track by confidence
        by_confidence[confidence].append(is_correct)

        predictions.append({
            "sample_id": sample["id"],
            "actual": actual_cwe,
            "predicted": predicted_cwe,
            "correct": is_correct,
            "confidence": confidence,
        })

    # Compute per-class metrics
    per_class = {}
    for cwe in CWE_LIST:
        tp = sum(1 for p in predictions if p["actual"] == cwe and p["predicted"] == cwe)
        fp = sum(1 for p in predictions if p["actual"] != cwe and p["predicted"] == cwe)
        fn = sum(1 for p in predictions if p["actual"] == cwe and p["predicted"] != cwe)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[cwe] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Compute calibration by confidence
    calibration = {}
    for conf, results_list in by_confidence.items():
        if results_list:
            calibration[conf] = {
                "count": len(results_list),
                "accuracy": sum(results_list) / len(results_list),
            }

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "calibration": calibration,
        "predictions": predictions,
    }
