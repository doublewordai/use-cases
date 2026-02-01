"""
Cost-performance curve analysis for bug detection ensemble.

Analyzes how performance changes with different numbers of prompts,
enabling cost-optimization decisions.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ensemble import aggregate_results, calculate_metrics, EnsembleResult
from .prompts import REVIEW_PROMPTS, PROMPT_SUBSETS


@dataclass
class SubsetResult:
    """Results for a subset of prompts."""
    num_prompts: int
    prompt_ids: list[str]
    subset_name: str
    metrics: dict
    cost_fraction: float  # Fraction of full ensemble cost


@dataclass
class CalibrationBin:
    """A bin for calibration analysis."""
    confidence_min: float
    confidence_max: float
    predictions: int
    correct: int
    accuracy: float


def analyze_prompt_subsets(
    raw_results: dict[str, dict],
    ground_truth: dict[str, bool],
    threshold: float = 0.5,
) -> list[SubsetResult]:
    """
    Analyze performance of different prompt subsets.

    Uses predefined subsets from prompts.py:
    - security_only: Security-focused prompts only (3)
    - security_logic: Security + logic prompts (7)
    - core: Most effective prompts (10)
    - full: All prompts (20)

    Args:
        raw_results: Dict mapping custom_id to batch result
        ground_truth: Dict mapping sample_id to has_bug
        threshold: Vote threshold for predictions

    Returns:
        List of SubsetResult for each subset
    """
    results = []

    for subset_name, prompt_ids in PROMPT_SUBSETS.items():
        # Filter results to only include these prompts
        filtered_results = {}
        for custom_id, result in raw_results.items():
            if "__" not in custom_id:
                continue
            parts = custom_id.split("__")
            prompt_id = parts[1]
            if prompt_id in prompt_ids:
                filtered_results[custom_id] = result

        # Get unique sample IDs
        sample_ids = set()
        for custom_id in filtered_results.keys():
            sample_ids.add(custom_id.split("__")[0])

        # Aggregate results for each sample
        ensemble_results = []
        for sample_id in sample_ids:
            sample_results = {
                k: v for k, v in filtered_results.items()
                if k.startswith(f"{sample_id}__")
            }
            result = aggregate_results(sample_id, sample_results, threshold)
            ensemble_results.append(result)

        # Calculate metrics
        metrics = calculate_metrics(ensemble_results, ground_truth)

        results.append(SubsetResult(
            num_prompts=len(prompt_ids),
            prompt_ids=prompt_ids,
            subset_name=subset_name,
            metrics=metrics,
            cost_fraction=len(prompt_ids) / len(REVIEW_PROMPTS),
        ))

    # Sort by number of prompts
    results.sort(key=lambda x: x.num_prompts)

    return results


def find_elbow_point(subset_results: list[SubsetResult], metric: str = "recall") -> SubsetResult:
    """
    Find the "elbow" point where adding more prompts has diminishing returns.

    Uses the point with best metric-to-cost ratio improvement.

    Args:
        subset_results: List of subset results (sorted by num_prompts)
        metric: Metric to optimize (recall, f1, precision)

    Returns:
        The SubsetResult at the elbow point
    """
    if len(subset_results) < 2:
        return subset_results[0] if subset_results else None

    # Calculate efficiency (metric per unit cost) for each subset
    efficiencies = []
    for result in subset_results:
        value = result.metrics.get(metric, 0)
        cost = result.cost_fraction
        efficiency = value / cost if cost > 0 else 0
        efficiencies.append((result, efficiency))

    # Find best efficiency
    best = max(efficiencies, key=lambda x: x[1])
    return best[0]


def calculate_calibration(
    results: list[EnsembleResult],
    ground_truth: dict[str, bool],
    num_bins: int = 5,
) -> list[CalibrationBin]:
    """
    Calculate calibration - do confidence scores predict accuracy?

    Bins predictions by confidence and checks accuracy per bin.
    Well-calibrated models have higher accuracy at higher confidence.

    Args:
        results: Ensemble results with confidence scores
        ground_truth: Ground truth labels
        num_bins: Number of confidence bins

    Returns:
        List of calibration bins from low to high confidence
    """
    bins = []
    bin_size = 1.0 / num_bins

    for i in range(num_bins):
        conf_min = i * bin_size
        conf_max = (i + 1) * bin_size

        # Find predictions in this confidence range
        in_bin = [
            r for r in results
            if conf_min <= r.confidence < conf_max
        ]

        if not in_bin:
            bins.append(CalibrationBin(
                confidence_min=conf_min,
                confidence_max=conf_max,
                predictions=0,
                correct=0,
                accuracy=0.0,
            ))
            continue

        correct = sum(
            1 for r in in_bin
            if r.predicted_has_bug == ground_truth.get(r.sample_id, False)
        )

        bins.append(CalibrationBin(
            confidence_min=conf_min,
            confidence_max=conf_max,
            predictions=len(in_bin),
            correct=correct,
            accuracy=correct / len(in_bin),
        ))

    return bins


def calculate_triage_stats(
    results: list[EnsembleResult],
    ground_truth: dict[str, bool],
    high_conf_threshold: float = 0.7,
) -> dict:
    """
    Calculate statistics for a triage workflow.

    Triage categories:
    - Auto-flag: Unanimous "bug" (>= high_conf_threshold and predicted bug)
    - Auto-approve: Unanimous "clean" (>= high_conf_threshold and predicted clean)
    - Human review: Split vote (< high_conf_threshold)

    Args:
        results: Ensemble results
        ground_truth: Ground truth labels
        high_conf_threshold: Confidence threshold for auto-decisions

    Returns:
        Dict with triage statistics
    """
    auto_flag = []
    auto_approve = []
    human_review = []

    for r in results:
        actual = ground_truth.get(r.sample_id, False)
        correct = r.predicted_has_bug == actual

        if r.confidence >= high_conf_threshold:
            if r.predicted_has_bug:
                auto_flag.append((r, correct, actual))
            else:
                auto_approve.append((r, correct, actual))
        else:
            human_review.append((r, correct, actual))

    def calc_stats(items):
        if not items:
            return {"count": 0, "accuracy": 0.0, "false_negatives": 0, "false_positives": 0}

        correct = sum(1 for _, c, _ in items if c)
        fn = sum(1 for r, c, actual in items if actual and not r.predicted_has_bug)
        fp = sum(1 for r, c, actual in items if not actual and r.predicted_has_bug)

        return {
            "count": len(items),
            "accuracy": correct / len(items),
            "false_negatives": fn,
            "false_positives": fp,
        }

    return {
        "auto_flag": calc_stats(auto_flag),
        "auto_approve": calc_stats(auto_approve),
        "human_review": calc_stats(human_review),
        "total_samples": len(results),
        "auto_rate": (len(auto_flag) + len(auto_approve)) / len(results) if results else 0,
    }


def generate_curve_data(subset_results: list[SubsetResult]) -> dict:
    """
    Generate data for plotting cost-performance curves.

    Returns:
        Dict with x (cost/prompts) and y (various metrics) arrays
    """
    return {
        "num_prompts": [r.num_prompts for r in subset_results],
        "cost_fraction": [r.cost_fraction for r in subset_results],
        "recall": [r.metrics["recall"] for r in subset_results],
        "precision": [r.metrics["precision"] for r in subset_results],
        "f1": [r.metrics["f1"] for r in subset_results],
        "accuracy": [r.metrics["accuracy"] for r in subset_results],
        "subset_names": [r.subset_name for r in subset_results],
    }


def format_calibration_table(calibration: list[CalibrationBin]) -> str:
    """Format calibration bins as a markdown table."""
    lines = [
        "| Confidence Range | Predictions | Correct | Accuracy |",
        "|------------------|-------------|---------|----------|",
    ]
    for b in calibration:
        lines.append(
            f"| {b.confidence_min:.0%}-{b.confidence_max:.0%} | "
            f"{b.predictions} | {b.correct} | {b.accuracy:.1%} |"
        )
    return "\n".join(lines)


def format_triage_table(triage: dict) -> str:
    """Format triage stats as a markdown table."""
    af = triage["auto_flag"]
    aa = triage["auto_approve"]
    hr = triage["human_review"]

    lines = [
        "| Category | Count | Accuracy | False Neg | False Pos |",
        "|----------|-------|----------|-----------|-----------|",
        f"| Auto-flag (bug) | {af['count']} | {af['accuracy']:.1%} | {af['false_negatives']} | {af['false_positives']} |",
        f"| Auto-approve (clean) | {aa['count']} | {aa['accuracy']:.1%} | {aa['false_negatives']} | {aa['false_positives']} |",
        f"| Human review | {hr['count']} | {hr['accuracy']:.1%} | {hr['false_negatives']} | {hr['false_positives']} |",
    ]
    return "\n".join(lines)


def format_subset_comparison(subset_results: list[SubsetResult]) -> str:
    """Format subset comparison as a markdown table."""
    lines = [
        "| Subset | Prompts | Cost | Recall | Precision | F1 |",
        "|--------|---------|------|--------|-----------|-----|",
    ]
    for r in subset_results:
        lines.append(
            f"| {r.subset_name} | {r.num_prompts} | {r.cost_fraction:.0%} | "
            f"{r.metrics['recall']:.1%} | {r.metrics['precision']:.1%} | {r.metrics['f1']:.1%} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with mock data
    print("=== Analysis Module Demo ===")
    print("\nThis module provides:")
    print("- analyze_prompt_subsets(): Compare different prompt combinations")
    print("- calculate_calibration(): Check if confidence predicts accuracy")
    print("- calculate_triage_stats(): Stats for auto-flag/approve/review workflow")
    print("- find_elbow_point(): Find optimal cost-performance tradeoff")

    print("\nExample calibration table:")
    mock_bins = [
        CalibrationBin(0.0, 0.2, 10, 5, 0.5),
        CalibrationBin(0.2, 0.4, 15, 9, 0.6),
        CalibrationBin(0.4, 0.6, 20, 14, 0.7),
        CalibrationBin(0.6, 0.8, 25, 20, 0.8),
        CalibrationBin(0.8, 1.0, 30, 27, 0.9),
    ]
    print(format_calibration_table(mock_bins))
