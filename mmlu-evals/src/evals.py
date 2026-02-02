"""MMLU evaluation dataset loading and scoring."""

import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class EvalQuestion:
    """A single MMLU evaluation question."""

    id: str
    question: str
    choices: list[str]
    answer: str  # The correct answer letter (A, B, C, or D)
    subject: str


# MMLU subject categories
SUBJECT_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

# Create reverse mapping: subject -> category
SUBJECT_TO_CATEGORY = {}
for category, subjects in SUBJECT_CATEGORIES.items():
    for subject in subjects:
        SUBJECT_TO_CATEGORY[subject] = category


def load_mmlu(
    split: str = "test",
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalQuestion]:
    """Load MMLU dataset.

    MMLU contains ~14K test questions across 57 subjects spanning STEM,
    humanities, social sciences, and other domains.

    Args:
        split: Dataset split ("test", "validation", "dev")
        subjects: List of subjects to load (default: all)
        limit: Maximum questions to load (default: all)

    Returns:
        List of EvalQuestion objects
    """
    # Get all subjects if not specified
    if subjects is None:
        subjects = list(SUBJECT_TO_CATEGORY.keys())

    questions = []
    question_count = 0

    for subject in subjects:
        if limit and question_count >= limit:
            break

        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
        except Exception:
            # Some subjects might not be available
            continue

        for i, item in enumerate(dataset):
            if limit and question_count >= limit:
                break

            # MMLU format: question, choices (list), answer (int 0-3)
            answer_idx = item["answer"]
            answer_letter = chr(ord("A") + answer_idx)

            questions.append(
                EvalQuestion(
                    id=f"mmlu-{subject}-{i:05d}",
                    question=item["question"],
                    choices=item["choices"],
                    answer=answer_letter,
                    subject=subject,
                )
            )
            question_count += 1

    return questions


def build_eval_prompt(question: str, choices: list[str]) -> list[dict]:
    """Build prompt messages for MMLU evaluation."""
    # Format choices as A, B, C, D
    choices_text = "\n".join(
        f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers multiple choice questions. "
                "Analyze the question carefully, then provide your answer as a single "
                "letter (A, B, C, or D) on the last line in the format: ANSWER: X"
            ),
        },
        {
            "role": "user",
            "content": f"{question}\n\n{choices_text}",
        },
    ]


def extract_model_answer(response_text: str) -> str | None:
    """Extract the answer letter from model response.

    Tries multiple formats:
    1. Explicit "ANSWER: X" format (most reliable)
    2. Standalone letter at end
    3. "The answer is X" patterns
    """
    response_text = response_text.strip()

    # 1. Try "ANSWER: X" format
    matches = re.findall(r"ANSWER:\s*([A-Da-d])", response_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 2. Try "the answer is X" patterns
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*([A-Da-d])",
        r"(?:correct\s+)?answer\s*[:=]\s*([A-Da-d])",
        r"\b([A-Da-d])\s*(?:is\s+(?:the\s+)?(?:correct|right)\s+answer)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 3. Try standalone letter at end of response
    # Look for a single letter on its own line or at the very end
    end_match = re.search(r"\n\s*([A-Da-d])\s*\.?\s*$", response_text)
    if end_match:
        return end_match.group(1).upper()

    # 4. If response is just a single letter
    if len(response_text) == 1 and response_text.upper() in "ABCD":
        return response_text.upper()

    return None


def score_answer(model_answer: str | None, correct_answer: str) -> bool:
    """Check if model answer matches correct answer."""
    if model_answer is None:
        return False
    return model_answer.upper() == correct_answer.upper()


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))

    # Group by subject
    by_subject = {}
    for r in results:
        subject = r.get("subject", "unknown")
        if subject not in by_subject:
            by_subject[subject] = {"total": 0, "correct": 0}
        by_subject[subject]["total"] += 1
        if r.get("correct", False):
            by_subject[subject]["correct"] += 1

    subject_scores = {}
    for subject, counts in by_subject.items():
        subject_scores[subject] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
            "total": counts["total"],
            "correct": counts["correct"],
        }

    # Group by category
    by_category = {}
    for r in results:
        subject = r.get("subject", "unknown")
        category = SUBJECT_TO_CATEGORY.get(subject, "Other")
        if category not in by_category:
            by_category[category] = {"total": 0, "correct": 0}
        by_category[category]["total"] += 1
        if r.get("correct", False):
            by_category[category]["correct"] += 1

    category_scores = {}
    for category, counts in by_category.items():
        category_scores[category] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
            "total": counts["total"],
            "correct": counts["correct"],
        }

    return {
        "overall_accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "by_subject": subject_scores,
        "by_category": category_scores,
    }
