"""CLI for running model evaluations via the Doubleword CLI.

This module handles data preparation and scoring. Batch submission,
monitoring, and result retrieval are done via the `dw` CLI — see the
README for the full workflow.
"""

import json
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from .evals import (
    EvalQuestion,
    build_eval_prompt,
    compute_metrics,
    extract_model_answer,
    load_gsm8k,
    score_answer,
)


@click.group()
def cli():
    """Model evaluation using the Doubleword CLI.

    Prepare batch files, then use `dw stream` to submit and retrieve results.
    """
    pass


@cli.command()
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Number of questions to include (default: all 1319)",
)
@click.option(
    "--output",
    "-o",
    default="batches/batch.jsonl",
    help="Output JSONL batch file path",
)
def prepare(limit: int | None, output: str):
    """Download GSM8K and generate a batch-ready JSONL file.

    The output file has no model set — use `dw files prepare --model <name>`
    to set the model before submitting.
    """
    click.echo("Loading GSM8K dataset...")
    questions = load_gsm8k(split="test", limit=limit)
    click.echo(f"Loaded {len(questions)} questions")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write batch-ready JSONL (model intentionally omitted)
    with open(output_path, "w") as f:
        for q in questions:
            line = {
                "custom_id": q.id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": build_eval_prompt(q.question),
                    "temperature": 0,
                    "max_tokens": 1024,
                },
            }
            f.write(json.dumps(line) + "\n")

    # Also save the ground truth for scoring later
    ground_truth_path = output_path.parent / "ground_truth.json"
    ground_truth = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "category": q.category,
        }
        for q in questions
    ]
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    click.echo(f"Created {output_path} ({len(questions)} requests)")
    click.echo(f"Ground truth saved to {ground_truth_path}")
    click.echo("\nRun `dw project info` for next steps.")


@cli.command()
@click.option(
    "--results",
    "-r",
    required=True,
    help="Results JSONL file (from `dw stream` or `dw batches results`)",
)
@click.option(
    "--dataset",
    "-d",
    default="batches/ground_truth.json",
    help="Ground truth dataset file",
)
@click.option(
    "--output",
    "-o",
    default="results/",
    help="Output directory for scored results",
)
def score(results: str, dataset: str, output: str):
    """Score batch results against ground truth."""
    results_path = Path(results)
    dataset_path = Path(dataset)

    if not results_path.exists():
        raise click.ClickException(f"Results file not found: {results_path}")
    if not dataset_path.exists():
        raise click.ClickException(
            f"Dataset file not found: {dataset_path}\n"
            "Run `dw project run prepare` first to generate ground truth."
        )

    # Load ground truth
    with open(dataset_path) as f:
        questions_data = json.load(f)
    questions = {q["id"]: q for q in questions_data}

    # Load results (JSONL)
    batch_results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                batch_results[obj["custom_id"]] = obj

    click.echo(f"Loaded {len(batch_results)} results from {results_path}")

    # Score each result
    scored = []
    errors = []

    for custom_id, result in tqdm(batch_results.items(), desc="Scoring"):
        q = questions.get(custom_id)
        if not q:
            errors.append({"id": custom_id, "error": "Question not found in dataset"})
            continue

        if result.get("error"):
            errors.append({"id": custom_id, "error": result["error"]})
            scored.append(
                {
                    "id": custom_id,
                    "question": q["question"],
                    "correct_answer": q["answer"],
                    "model_answer": None,
                    "model_response": None,
                    "correct": False,
                    "category": q.get("category"),
                    "error": result["error"],
                }
            )
            continue

        # Handle both formats:
        # - Doubleword dw stream: {"response_body": {"choices": [...]}}
        # - OpenAI batch format:  {"response": {"body": {"choices": [...]}}}
        response_body = result.get("response_body") or result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            errors.append({"id": custom_id, "error": "No choices in response"})
            continue

        model_response = choices[0].get("message", {}).get("content", "")
        model_answer = extract_model_answer(model_response)
        is_correct = score_answer(model_answer, q["answer"])

        scored.append(
            {
                "id": custom_id,
                "question": q["question"],
                "correct_answer": q["answer"],
                "model_answer": model_answer,
                "model_response": model_response,
                "correct": is_correct,
                "category": q.get("category"),
            }
        )

    # Compute metrics
    metrics = compute_metrics(scored)

    # Count tokens from results
    input_tokens = 0
    output_tokens = 0
    for r in batch_results.values():
        rb = r.get("response_body") or r.get("response", {}).get("body", {})
        usage = rb.get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)

    # Save scored results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_file = output_dir / f"scored_{timestamp}.json"
    with open(scored_file, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "token_counts": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "results": scored,
                "errors": errors,
            },
            f,
            indent=2,
        )

    click.echo(f"\nSaved scored results: {scored_file}")

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 50)
    click.echo(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    click.echo(f"Correct: {metrics['correct']}/{metrics['total']}")

    if metrics["by_category"]:
        click.echo("\nBy Category:")
        for cat, scores in metrics["by_category"].items():
            click.echo(
                f"  {cat}: {scores['accuracy']:.1%} ({scores['correct']}/{scores['total']})"
            )

    click.echo(f"\nTokens Used:")
    click.echo(f"  Input: {input_tokens:,}")
    click.echo(f"  Output: {output_tokens:,}")

    if errors:
        click.echo(f"\nErrors: {len(errors)}")


def main():
    cli()
