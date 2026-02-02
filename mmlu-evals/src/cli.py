"""CLI for running MMLU evaluations via batch API."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
from tqdm import tqdm

from .batch import (
    count_tokens,
    create_batch,
    create_batch_file,
    download_results,
    get_client,
    parse_results,
    upload_batch_file,
    wait_for_batch,
)
from .evals import (
    EvalQuestion,
    SUBJECT_CATEGORIES,
    build_eval_prompt,
    compute_metrics,
    extract_model_answer,
    load_mmlu,
    score_answer,
)

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


@click.group()
def cli():
    """MMLU evaluation using Doubleword Batch API.

    Run large-scale evaluations affordably against MMLU benchmark.
    """
    pass


@cli.command()
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Number of questions to include (default: all ~14K)",
)
@click.option(
    "--subjects",
    "-s",
    default=None,
    help="Comma-separated list of subjects (default: all)",
)
@click.option(
    "--category",
    "-c",
    default=None,
    type=click.Choice(["STEM", "Humanities", "Social Sciences", "Other"]),
    help="Load only subjects from this category",
)
@click.option(
    "--output",
    "-o",
    default="data/mmlu_sample.json",
    help="Output file path",
)
def prepare(limit: int | None, subjects: str | None, category: str | None, output: str):
    """Prepare evaluation dataset (download MMLU sample)."""
    # Parse subjects
    subject_list = None
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",")]
    elif category:
        subject_list = SUBJECT_CATEGORIES.get(category, [])

    desc = f"limit: {limit}" if limit else "all"
    if subject_list:
        desc += f", {len(subject_list)} subjects"
    click.echo(f"Loading MMLU dataset ({desc})...")

    questions = load_mmlu(split="test", subjects=subject_list, limit=limit)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "id": q.id,
            "question": q.question,
            "choices": q.choices,
            "answer": q.answer,
            "subject": q.subject,
        }
        for q in questions
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Print subject breakdown
    subjects_count = {}
    for q in questions:
        subjects_count[q.subject] = subjects_count.get(q.subject, 0) + 1

    click.echo(f"Saved {len(questions)} questions to {output_path}")
    click.echo(f"Subjects: {len(subjects_count)}")


@cli.command()
def list_subjects():
    """List available MMLU subjects by category."""
    for category, subjects in SUBJECT_CATEGORIES.items():
        click.echo(f"\n{category} ({len(subjects)} subjects):")
        for subject in sorted(subjects):
            click.echo(f"  - {subject}")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="data/mmlu_sample.json",
    help="Input dataset file",
)
@click.option(
    "--output",
    "-o",
    default="data/",
    help="Output directory for batch files",
)
@click.option(
    "--provider",
    "-p",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
    help="API provider (default: doubleword)",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    help="Model alias (30b, 235b, gpt5-nano, gpt5-mini, gpt5.2) or full name",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Create batch file but don't submit",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for batch completion",
)
@click.option(
    "--realtime",
    is_flag=True,
    help="Use realtime API instead of batch (faster but more expensive)",
)
@click.option(
    "--concurrency",
    "-c",
    default=10,
    help="Number of concurrent requests in realtime mode (default: 10)",
)
def run(
    input_path: str,
    output: str,
    provider: str,
    model: str,
    dry_run: bool,
    wait: bool,
    realtime: bool,
    concurrency: int,
):
    """Run evaluation batch against specified model."""
    # Resolve model alias to full name
    model = MODELS.get(model, model)

    # Load questions
    input_file = Path(input_path)
    if not input_file.exists():
        raise click.ClickException(
            f"Dataset not found: {input_file}\nRun 'prepare' command first."
        )

    with open(input_file) as f:
        questions_data = json.load(f)

    questions = [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            choices=q["choices"],
            answer=q["answer"],
            subject=q.get("subject", "unknown"),
        )
        for q in questions_data
    ]

    click.echo(f"Loaded {len(questions)} questions from {input_file}")

    # Build batch requests
    requests_data = []
    for q in questions:
        requests_data.append(
            {
                "custom_id": q.id,
                "model": model,
                "messages": build_eval_prompt(q.question, q.choices),
                "temperature": 0,
                "max_tokens": 512,
            }
        )

    # Create batch file
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    batch_file = output_dir / f"batch_{model_slug}_{timestamp}.jsonl"

    create_batch_file(requests_data, batch_file)
    click.echo(f"Created batch file: {batch_file}")

    if dry_run:
        click.echo("Dry run - batch not submitted")
        return

    # Get client
    client, provider_name = get_client(provider)
    click.echo(f"Using provider: {provider_name}")

    # Realtime mode - process requests directly with concurrency
    if realtime:
        click.echo(
            f"\nProcessing {len(requests_data)} requests in realtime "
            f"(concurrency={concurrency})..."
        )
        results_file = output_dir / f"results_{model_slug}_{timestamp}.jsonl"

        def process_request(req):
            """Process a single request."""
            try:
                params = {
                    "model": req["model"],
                    "messages": req["messages"],
                }
                if provider_name == "openai":
                    params["max_completion_tokens"] = req.get("max_tokens", 512)
                else:
                    params["max_tokens"] = req.get("max_tokens", 512)
                    params["temperature"] = req.get("temperature", 0)

                response = client.chat.completions.create(**params)
                return {
                    "custom_id": req["custom_id"],
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": response.choices[0].message.content
                                    }
                                }
                            ],
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                            },
                        }
                    },
                }
            except Exception as e:
                return {
                    "custom_id": req["custom_id"],
                    "error": {"message": str(e)},
                }

        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(process_request, req): req for req in requests_data
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                results.append(future.result())

        # Sort results by custom_id to maintain order
        results.sort(key=lambda x: x["custom_id"])

        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        click.echo(f"Saved results: {results_file}")

        # Save metadata
        metadata = {
            "provider": provider_name,
            "model": model,
            "mode": "realtime",
            "num_questions": len(questions),
            "timestamp": timestamp,
            "input_file": str(input_file),
            "results_file": str(results_file),
        }
        metadata_file = output_dir / f"batch_{model_slug}_{timestamp}_meta.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        click.echo(f"Saved metadata: {metadata_file}")
        return

    # Batch mode - submit to batch API

    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)
    click.echo(f"  File ID: {file_id}")

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"  Batch ID: {batch_id}")

    # Save batch metadata
    metadata = {
        "batch_id": batch_id,
        "file_id": file_id,
        "provider": provider_name,
        "model": model,
        "num_questions": len(questions),
        "timestamp": timestamp,
        "input_file": str(input_file),
        "batch_file": str(batch_file),
    }
    metadata_file = output_dir / f"batch_{model_slug}_{timestamp}_meta.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    click.echo(f"Saved metadata: {metadata_file}")

    if not wait:
        click.echo(
            f"\nBatch submitted. Check status with:\n"
            f"  python -m src.cli status --batch-id {batch_id}"
        )
        return

    # Wait for completion
    click.echo("\nWaiting for batch completion...")
    batch = wait_for_batch(client, batch_id, poll_interval=15)

    if batch.status != "completed":
        raise click.ClickException(f"Batch failed with status: {batch.status}")

    click.echo("Batch completed!")

    # Download results
    results_file = output_dir / f"results_{model_slug}_{timestamp}.jsonl"
    download_results(batch.output_file_id, results_file, provider=provider_name)
    click.echo(f"Downloaded results: {results_file}")

    # Update metadata with results file
    metadata["results_file"] = str(results_file)
    metadata["output_file_id"] = batch.output_file_id
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


@cli.command()
@click.option("--batch-id", required=True, help="Batch ID to check")
@click.option(
    "--provider",
    "-p",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
    help="API provider (default: doubleword)",
)
def status(batch_id: str, provider: str):
    """Check status of a running batch."""
    client, _ = get_client(provider)
    batch = client.batches.retrieve(batch_id)

    click.echo(f"Batch ID: {batch.id}")
    click.echo(f"Status: {batch.status}")
    click.echo(
        f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
    )

    if batch.output_file_id:
        click.echo(f"Output file: {batch.output_file_id}")


@cli.command()
@click.option(
    "--results",
    "-r",
    required=True,
    help="Results JSONL file from batch",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    help="Original dataset file (auto-detected from metadata if not specified)",
)
@click.option(
    "--output",
    "-o",
    default="results/",
    help="Output directory for scored results",
)
def score(results: str, dataset: str | None, output: str):
    """Score batch results against ground truth."""
    results_path = Path(results)

    # Try to auto-detect dataset from metadata file
    if dataset is None:
        # Look for corresponding metadata file
        meta_pattern = results_path.name.replace("results_", "batch_").replace(
            ".jsonl", "_meta.json"
        )
        meta_path = results_path.parent / meta_pattern
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                dataset = meta.get("input_file")
                if dataset:
                    click.echo(f"Auto-detected dataset from metadata: {dataset}")

        if dataset is None:
            dataset = "data/mmlu_sample.json"
            click.echo(f"No metadata found, using default: {dataset}")

    dataset_path = Path(dataset)

    if not results_path.exists():
        raise click.ClickException(f"Results file not found: {results_path}")
    if not dataset_path.exists():
        raise click.ClickException(f"Dataset file not found: {dataset_path}")

    # Load dataset (ground truth)
    with open(dataset_path) as f:
        questions_data = json.load(f)
    questions = {q["id"]: q for q in questions_data}

    # Load results
    batch_results = parse_results(results_path)
    click.echo(f"Loaded {len(batch_results)} results")

    # Score each result
    scored = []
    errors = []

    for custom_id, result in tqdm(batch_results.items(), desc="Scoring"):
        q = questions.get(custom_id)
        if not q:
            errors.append({"id": custom_id, "error": "Question not found in dataset"})
            continue

        # Check for API error
        if result.get("error"):
            errors.append({"id": custom_id, "error": result["error"]})
            scored.append(
                {
                    "id": custom_id,
                    "question": q["question"],
                    "choices": q["choices"],
                    "correct_answer": q["answer"],
                    "model_answer": None,
                    "model_response": None,
                    "correct": False,
                    "subject": q.get("subject"),
                    "error": result["error"],
                }
            )
            continue

        # Extract model response
        response_body = result.get("response", {}).get("body", {})
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
                "choices": q["choices"],
                "correct_answer": q["answer"],
                "model_answer": model_answer,
                "model_response": model_response,
                "correct": is_correct,
                "subject": q.get("subject"),
            }
        )

    # Compute metrics
    metrics = compute_metrics(scored)
    token_counts = count_tokens(batch_results)

    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_file = output_dir / f"scored_{timestamp}.json"
    with open(scored_file, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "token_counts": token_counts,
                "results": scored,
                "errors": errors,
            },
            f,
            indent=2,
        )

    click.echo(f"\nSaved scored results: {scored_file}")

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("MMLU EVALUATION RESULTS")
    click.echo("=" * 50)
    click.echo(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    click.echo(f"Correct: {metrics['correct']}/{metrics['total']}")

    if metrics.get("by_category"):
        click.echo("\nBy Category:")
        for cat in ["STEM", "Humanities", "Social Sciences", "Other"]:
            if cat in metrics["by_category"]:
                scores = metrics["by_category"][cat]
                click.echo(
                    f"  {cat}: {scores['accuracy']:.1%} "
                    f"({scores['correct']}/{scores['total']})"
                )

    click.echo(f"\nTokens Used:")
    click.echo(f"  Input: {token_counts['input_tokens']:,}")
    click.echo(f"  Output: {token_counts['output_tokens']:,}")

    if errors:
        click.echo(f"\nErrors: {len(errors)}")


def main():
    cli()


if __name__ == "__main__":
    main()
