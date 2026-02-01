"""Evaluation dataset loading and scoring."""

import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class EvalQuestion:
    """A single evaluation question."""

    id: str
    question: str
    answer: str
    category: str | None = None


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[EvalQuestion]:
    """Load GSM8K dataset (grade school math).

    GSM8K contains 8.5K high-quality, linguistically diverse grade school
    math word problems. Each problem requires 2-8 steps to solve.
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    questions = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        # Extract the final numeric answer from the answer string
        # GSM8K format: "step by step reasoning\n#### final_answer"
        answer_text = item["answer"]
        final_answer = extract_gsm8k_answer(answer_text)

        questions.append(
            EvalQuestion(
                id=f"gsm8k-{i:05d}",
                question=item["question"],
                answer=final_answer,
                category="math",
            )
        )

    return questions


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract final numeric answer from GSM8K answer format."""
    # GSM8K answers end with "#### <number>"
    match = re.search(r"####\s*(.+)$", answer_text)
    if match:
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def build_eval_prompt(question: str) -> list[dict]:
    """Build prompt messages for evaluation."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that solves math problems. "
                "Show your reasoning step by step, then provide your final answer "
                "on the last line in the format: ANSWER: <number>"
            ),
        },
        {"role": "user", "content": question},
    ]


def extract_model_answer(response_text: str) -> str | None:
    """Extract the final answer from model response.

    Tries multiple formats in order of reliability:
    1. Explicit "ANSWER: X" format (most reliable)
    2. LaTeX \\boxed{X} format (common in math)
    3. "The answer is X" or "= X" at end of response

    Does NOT fall back to grabbing arbitrary numbers, as that often
    matches intermediate reasoning steps rather than final answers.
    """
    # Regex for numbers including scientific notation
    num_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"

    def clean_number(s: str) -> str | None:
        """Clean and normalize a number string."""
        s = re.sub(r"[\*\$\\{}]", "", s)  # Remove markdown/LaTeX formatting
        s = s.strip().rstrip(".")
        s = s.replace(",", "")
        match = re.search(num_pattern.replace(",", ""), s)
        return match.group(0) if match else None

    # 1. Try "ANSWER: X" format (most reliable - we asked for this format)
    matches = re.findall(r"ANSWER:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
    if matches:
        result = clean_number(matches[-1])
        if result:
            return result

    # 2. Try boxed format: \boxed{48}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if boxed_match:
        result = clean_number(boxed_match.group(1))
        if result:
            return result

    # 3. Try "the answer is X" or "answer: X" patterns
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*(" + num_pattern + r")",
        r"(?:final\s+)?answer\s*[:=]\s*(" + num_pattern + r")",
        r"=\s*(" + num_pattern + r")\s*$",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            result = clean_number(match.group(1))
            if result:
                return result

    # No reliable answer found - return None rather than guessing
    return None


def score_answer(model_answer: str | None, correct_answer: str) -> bool:
    """Check if model answer matches correct answer."""
    if model_answer is None:
        return False

    # Normalize both answers
    model_norm = model_answer.strip().replace(",", "").lower()
    correct_norm = correct_answer.strip().replace(",", "").lower()

    # Direct match
    if model_norm == correct_norm:
        return True

    # Try numeric comparison (handles "5.0" == "5")
    try:
        model_num = float(model_norm)
        correct_num = float(correct_norm)
        return abs(model_num - correct_num) < 1e-6
    except ValueError:
        pass

    return False


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))

    # Group by category
    by_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if r.get("correct", False):
            by_category[cat]["correct"] += 1

    category_scores = {}
    for cat, counts in by_category.items():
        category_scores[cat] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
            "total": counts["total"],
            "correct": counts["correct"],
        }

    return {
        "overall_accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "by_category": category_scores,
    }
