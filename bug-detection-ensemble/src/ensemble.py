"""
Ensemble aggregation logic for bug detection.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class BugFinding:
    """A bug found by a prompt."""
    prompt_id: str
    bug_type: str
    description: str
    line: Optional[int] = None


@dataclass
class EnsembleResult:
    """Aggregated result for a code sample."""
    sample_id: str
    votes_bug: int
    votes_no_bug: int
    total_votes: int
    confidence: float  # 0.0 (split) to 1.0 (unanimous)
    predicted_has_bug: bool
    findings: list[BugFinding]
    prompt_results: dict[str, dict]  # prompt_id -> result

    @property
    def vote_ratio(self) -> float:
        """Ratio of bug votes to total votes."""
        if self.total_votes == 0:
            return 0.0
        return self.votes_bug / self.total_votes

    @property
    def is_split(self) -> bool:
        """Whether the vote is split (neither unanimous nor near-unanimous)."""
        return 0.3 < self.vote_ratio < 0.7


def parse_llm_response(response_text: str) -> dict:
    """Parse JSON response from LLM."""
    try:
        # Try to parse as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                return json.loads(response_text[start:end].strip())
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end > start:
                return json.loads(response_text[start:end].strip())

        # Try to find JSON object in text
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])

        # Give up, assume no bug
        return {"has_bug": False, "bugs": []}


def aggregate_results(
    sample_id: str,
    results: dict[str, dict],  # custom_id -> batch result
    threshold: float = 0.5
) -> EnsembleResult:
    """
    Aggregate results from multiple prompts for a single sample.

    Args:
        sample_id: The code sample ID
        results: Dict mapping custom_id to batch result
        threshold: Vote threshold to predict bug (default 0.5 = majority)
    """
    votes_bug = 0
    votes_no_bug = 0
    findings = []
    prompt_results = {}

    for custom_id, result in results.items():
        # Extract prompt_id from custom_id (format: sample_id__prompt_id)
        if "__" not in custom_id:
            continue
        parts = custom_id.split("__")
        # Note: results dict should already be pre-filtered for this sample
        # so we don't need to check sample_id match here

        prompt_id = parts[1]

        # Get the response content
        response = result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            continue

        content = choices[0].get("message", {}).get("content", "")
        parsed = parse_llm_response(content)

        prompt_results[prompt_id] = {
            "raw_content": content,
            "parsed": parsed,
        }

        has_bug = parsed.get("has_bug", False)

        if has_bug:
            votes_bug += 1
            for bug in parsed.get("bugs", []):
                findings.append(BugFinding(
                    prompt_id=prompt_id,
                    bug_type=bug.get("type", "unknown"),
                    description=bug.get("description", ""),
                    line=bug.get("line"),
                ))
        else:
            votes_no_bug += 1

    total_votes = votes_bug + votes_no_bug

    if total_votes == 0:
        confidence = 0.0
        predicted_has_bug = False
    else:
        vote_ratio = votes_bug / total_votes
        # Confidence is how far from 0.5 the vote is
        confidence = abs(vote_ratio - 0.5) * 2
        predicted_has_bug = vote_ratio >= threshold

    return EnsembleResult(
        sample_id=sample_id,
        votes_bug=votes_bug,
        votes_no_bug=votes_no_bug,
        total_votes=total_votes,
        confidence=confidence,
        predicted_has_bug=predicted_has_bug,
        findings=findings,
        prompt_results=prompt_results,
    )


def calculate_metrics(
    results: list[EnsembleResult],
    ground_truth: dict[str, bool]  # sample_id -> has_bug
) -> dict:
    """Calculate precision, recall, F1 for ensemble predictions."""
    tp = fp = tn = fn = 0

    for result in results:
        actual = ground_truth.get(result.sample_id, False)
        predicted = result.predicted_has_bug

        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def analyze_vote_splits(
    results: list[EnsembleResult],
    ground_truth: dict[str, bool]
) -> dict:
    """Analyze whether vote splits correlate with bug difficulty."""
    split_results = []
    unanimous_results = []

    for result in results:
        actual = ground_truth.get(result.sample_id, False)
        correct = (result.predicted_has_bug == actual)

        entry = {
            "sample_id": result.sample_id,
            "vote_ratio": result.vote_ratio,
            "confidence": result.confidence,
            "correct": correct,
            "actual_has_bug": actual,
        }

        if result.is_split:
            split_results.append(entry)
        else:
            unanimous_results.append(entry)

    split_accuracy = (
        sum(1 for r in split_results if r["correct"]) / len(split_results)
        if split_results else 0.0
    )
    unanimous_accuracy = (
        sum(1 for r in unanimous_results if r["correct"]) / len(unanimous_results)
        if unanimous_results else 0.0
    )

    return {
        "split_count": len(split_results),
        "unanimous_count": len(unanimous_results),
        "split_accuracy": split_accuracy,
        "unanimous_accuracy": unanimous_accuracy,
        "split_samples": split_results,
        "unanimous_samples": unanimous_results,
    }


def categorize_findings(results: list[EnsembleResult]) -> dict[str, list]:
    """Group findings by bug type across all samples."""
    by_type = defaultdict(list)

    for result in results:
        for finding in result.findings:
            by_type[finding.bug_type].append({
                "sample_id": result.sample_id,
                "prompt_id": finding.prompt_id,
                "description": finding.description,
                "line": finding.line,
            })

    return dict(by_type)
