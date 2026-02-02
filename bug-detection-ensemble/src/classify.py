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
    "CWE-20": "Improper Input Validation - Not validating or incorrectly validating input",
    "CWE-190": "Integer Overflow - Arithmetic operation exceeds integer limits",
    "CWE-476": "NULL Pointer Dereference - Dereferencing a pointer that is NULL",
    "CWE-416": "Use After Free - Using memory after it has been freed",
    "CWE-362": "Race Condition - Concurrent execution with shared resource",
    "CWE-120": "Classic Buffer Overflow - Copying data without checking size",
    "CWE-400": "Resource Exhaustion - Not limiting resource consumption",
    "CWE-200": "Information Exposure - Exposing sensitive information",
    "CWE-295": "Improper Certificate Validation - Not properly validating certificates",
    "CWE-415": "Double Free - Freeing memory that has already been freed",
    "CWE-617": "Reachable Assertion - Assertion can be triggered by attacker",
    "CWE-22": "Path Traversal - Not neutralizing path elements like ../",
    "CWE-59": "Symlink Following - Following symbolic links to unintended files",
    "CWE-401": "Memory Leak - Not releasing memory after use",
    "CWE-835": "Infinite Loop - Loop with unreachable exit condition",
    "CWE-763": "Invalid Pointer Dereference - Dereferencing an invalid pointer",
    "CWE-78": "OS Command Injection - Executing commands with untrusted input",
    "CWE-674": "Uncontrolled Recursion - Recursion without proper termination",
    "CWE-772": "Missing Resource Release - Not releasing resources after use",
    "CWE-122": "Heap Buffer Overflow - Buffer overflow in heap memory",
    "CWE-269": "Improper Privilege Management - Not properly managing privileges",
}

CWE_LIST = list(CWE_CLASSES.keys())

# Groupings for coarser classification
CWE_GROUPS = {
    "Memory Safety": ["CWE-125", "CWE-787", "CWE-119", "CWE-120", "CWE-122"],
    "Pointer/Lifetime": ["CWE-476", "CWE-416", "CWE-415", "CWE-763"],
    "Integer": ["CWE-190"],
    "Resource": ["CWE-400", "CWE-401", "CWE-772"],
    "Input Validation": ["CWE-20", "CWE-22", "CWE-78"],
    "Concurrency": ["CWE-362"],
    "Control Flow": ["CWE-617", "CWE-835", "CWE-674"],
    "Other": ["CWE-59", "CWE-295", "CWE-269", "CWE-200"],
}

# Reverse mapping: CWE -> group
CWE_TO_GROUP = {cwe: group for group, cwes in CWE_GROUPS.items() for cwe in cwes}

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
    max_per_cwe: int | None = None,
    min_code_length: int = 100,
    max_code_length: int = 3000,
) -> list[dict]:
    """
    Load vulnerable code samples for CWE classification.

    Only loads samples from the target CWE classes.

    Args:
        db_path: Path to CVEfixes database
        max_per_cwe: Maximum samples per CWE (None for all)
        min_code_length: Minimum code length in characters
        max_code_length: Maximum code length in characters
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
        """
        params = [min_code_length, max_code_length, cwe]

        if max_per_cwe is not None:
            query += " LIMIT ?"
            params.append(max_per_cwe)

        cursor = conn.execute(query, params)

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

        # Track by confidence (normalize variations)
        conf_normalized = confidence.lower().replace("-", "_")
        if conf_normalized not in by_confidence:
            by_confidence[conf_normalized] = []
        by_confidence[conf_normalized].append(is_correct)

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

    # Compute grouped accuracy
    grouped_correct = 0
    grouped_total = 0
    per_group = {}

    for p in predictions:
        actual_group = CWE_TO_GROUP.get(p["actual"])
        predicted_group = CWE_TO_GROUP.get(p["predicted"]) if p["predicted"] else None

        if actual_group:
            grouped_total += 1
            if actual_group == predicted_group:
                grouped_correct += 1

    # Per-group metrics
    for group_name in CWE_GROUPS:
        group_preds = [p for p in predictions if CWE_TO_GROUP.get(p["actual"]) == group_name]
        if group_preds:
            group_correct = sum(
                1 for p in group_preds
                if CWE_TO_GROUP.get(p["predicted"]) == group_name
            )
            per_group[group_name] = {
                "accuracy": group_correct / len(group_preds),
                "support": len(group_preds),
            }

    return {
        "accuracy": correct / total if total > 0 else 0,
        "grouped_accuracy": grouped_correct / grouped_total if grouped_total > 0 else 0,
        "total": total,
        "correct": correct,
        "grouped_correct": grouped_correct,
        "per_class": per_class,
        "per_group": per_group,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "calibration": calibration,
        "predictions": predictions,
    }
