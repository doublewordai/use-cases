"""CLI for synthetic data generation via batch API."""

import json
from pathlib import Path

import click

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
from .generator import (
    build_conversation_requests,
    build_quality_requests,
    build_scenario_requests,
    format_for_training,
    load_seed_topics,
    parse_json_response,
)
from .prompts import SUPPORT_TOPICS

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


def _extract_content(result: dict) -> str:
    """Extract message content from a batch result."""
    try:
        return result["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def _run_stage(
    client,
    provider: str,
    requests_data: list[dict],
    stage_name: str,
    output_dir: Path,
) -> dict[str, dict]:
    """Run a complete batch stage: create file, upload, wait, download, parse."""
    click.echo(f"\n--- Stage: {stage_name} ---")
    click.echo(f"Requests: {len(requests_data)}")

    batch_file = output_dir / f"{stage_name}_input.jsonl"
    create_batch_file(requests_data, batch_file)

    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)
    click.echo(f"  File ID: {file_id}")

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"  Batch ID: {batch_id}")

    click.echo("Waiting for batch to complete...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed":
        raise click.ClickException(f"Batch failed with status: {batch.status}")

    results_file = output_dir / f"{stage_name}_output.jsonl"
    download_results(batch.output_file_id, results_file, provider)
    click.echo(f"Results saved: {results_file}")

    return parse_results(results_file)


@click.group()
def cli():
    """Synthetic data generation using the Doubleword Batch API.

    Generate training data for fine-tuning customer support models.
    """
    pass


@cli.command()
@click.option(
    "-n", "--count", default=1000, help="Number of samples to generate (default: 1000)"
)
@click.option(
    "-m",
    "--model",
    default=DEFAULT_MODEL,
    help="Model alias (30b, 235b, gpt5-nano, gpt5-mini, gpt5.2) or full name",
)
@click.option(
    "-p",
    "--provider",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
@click.option("-o", "--output", default="results/", help="Output directory")
@click.option(
    "--domain",
    default="customer support",
    help="Domain for generated data (default: 'customer support')",
)
@click.option(
    "--product",
    default="SaaS platform",
    help="Product type for scenarios (default: 'SaaS platform')",
)
@click.option(
    "--seed-file",
    default=None,
    help="CSV or JSONL with a 'topic' column to use as scenario seeds",
)
@click.option("--dry-run", is_flag=True, help="Create batch files but don't submit")
def run(
    count: int,
    model: str,
    provider: str,
    output: str,
    domain: str,
    product: str,
    seed_file: str,
    dry_run: bool,
):
    """Run the three-stage generation pipeline (scenarios -> conversations -> quality).

    By default generates customer support training data for a SaaS platform.
    Use --domain and --product to customize, or --seed-file to provide your
    own topic seeds.
    """
    model_id = MODELS.get(model, model)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load topics
    if seed_file:
        topics = load_seed_topics(seed_file)
        click.echo(f"Loaded {len(topics)} topics from {seed_file}")
    else:
        topics = SUPPORT_TOPICS

    click.echo(f"Model: {model_id}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Domain: {domain}")
    click.echo(f"Product: {product}")
    click.echo(f"Topics: {len(topics)}")
    click.echo(f"Count: {count}")

    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    # Stage 1: Generate scenarios
    scenario_requests = build_scenario_requests(
        count, model_id, topics=topics, domain=domain, product=product
    )
    click.echo(f"\nStage 1: {len(scenario_requests)} scenario requests")

    batch_file = output_dir / "stage1_scenarios_input.jsonl"
    create_batch_file(scenario_requests, batch_file)

    if dry_run:
        click.echo("Dry run - batch files created but not submitted.")
        return

    client, _ = get_client(provider)

    scenario_results = _run_stage(
        client,
        provider,
        scenario_requests,
        "stage1_scenarios",
        output_dir,
    )
    tokens = count_tokens(scenario_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    scenarios = []
    for custom_id in sorted(scenario_results.keys()):
        content = _extract_content(scenario_results[custom_id])
        if content:
            try:
                scenario = parse_json_response(content)
                scenarios.append(scenario)
            except (json.JSONDecodeError, ValueError):
                click.echo(f"  Warning: failed to parse {custom_id}")

    with open(output_dir / "scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    click.echo(f"Parsed scenarios: {len(scenarios)}")

    # Stage 2: Generate conversations
    conv_requests = build_conversation_requests(scenarios, model_id, domain=domain)
    conv_results = _run_stage(
        client,
        provider,
        conv_requests,
        "stage2_conversations",
        output_dir,
    )
    tokens = count_tokens(conv_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    conversations = []
    for custom_id in sorted(conv_results.keys()):
        content = _extract_content(conv_results[custom_id])
        if content:
            try:
                conversation = parse_json_response(content)
                conversations.append(conversation)
            except (json.JSONDecodeError, ValueError):
                click.echo(f"  Warning: failed to parse {custom_id}")

    with open(output_dir / "conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
    click.echo(f"Parsed conversations: {len(conversations)}")

    # Stage 3: Quality scoring
    quality_requests = build_quality_requests(conversations, model_id)
    quality_results = _run_stage(
        client,
        provider,
        quality_requests,
        "stage3_quality",
        output_dir,
    )
    tokens = count_tokens(quality_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    scores = []
    for custom_id in sorted(quality_results.keys()):
        content = _extract_content(quality_results[custom_id])
        if content:
            try:
                score = parse_json_response(content)
                scores.append(score)
            except (json.JSONDecodeError, ValueError):
                click.echo(f"  Warning: failed to parse {custom_id}")

    with open(output_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    click.echo(f"Quality scores: {len(scores)}")

    # Summary
    if scores:
        avg_overall = sum(s.get("overall", 0) for s in scores) / len(scores)
        passing = sum(1 for s in scores if s.get("overall", 0) >= 3.5)
        click.echo(f"\n{'=' * 60}")
        click.echo("Pipeline Complete")
        click.echo(f"{'=' * 60}")
        click.echo(f"Scenarios generated: {len(scenarios)}")
        click.echo(f"Conversations generated: {len(conversations)}")
        click.echo(f"Average quality score: {avg_overall:.2f}/5.0")
        click.echo(
            f"Passing (>=3.5): {passing}/{len(scores)} ({passing / len(scores) * 100:.1f}%)"
        )
        click.echo(f"Tokens: {total_tokens}")


@cli.command()
@click.option("--batch-id", required=True, help="Batch ID to check")
@click.option(
    "-p",
    "--provider",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
def status(batch_id: str, provider: str):
    """Check batch job status."""
    client, _ = get_client(provider)
    batch = client.batches.retrieve(batch_id)

    click.echo(f"Batch ID: {batch.id}")
    click.echo(f"Status: {batch.status}")
    click.echo(
        f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
    )


@cli.command()
@click.option("-o", "--output", default="results/", help="Results directory")
def analyze(output: str):
    """Analyze quality scores from a completed pipeline run."""
    output_dir = Path(output)

    scores_path = output_dir / "scores.json"
    if not scores_path.exists():
        raise click.ClickException(
            f"No scores found at {scores_path}. Run the pipeline first."
        )

    with open(scores_path) as f:
        scores = json.load(f)

    total = len(scores)
    if total == 0:
        click.echo("No scores to analyze.")
        return

    passing = sum(1 for s in scores if s.get("overall", 0) >= 3.5)
    avg_naturalness = sum(s.get("naturalness", 0) for s in scores) / total
    avg_helpfulness = sum(s.get("helpfulness", 0) for s in scores) / total
    avg_guidelines = sum(s.get("guidelines", 0) for s in scores) / total
    avg_overall = sum(s.get("overall", 0) for s in scores) / total

    click.echo("=== Quality Analysis ===")
    click.echo(f"\nTotal generated: {total}")
    click.echo(f"Pass rate (>=3.5): {passing}/{total} ({passing / total * 100:.1f}%)")
    click.echo(f"\nAverage scores:")
    click.echo(f"  Naturalness:  {avg_naturalness:.2f}/5.0")
    click.echo(f"  Helpfulness:  {avg_helpfulness:.2f}/5.0")
    click.echo(f"  Guidelines:   {avg_guidelines:.2f}/5.0")
    click.echo(f"  Overall:      {avg_overall:.2f}/5.0")

    # Topic distribution from scenarios
    scenarios_path = output_dir / "scenarios.json"
    if scenarios_path.exists():
        with open(scenarios_path) as f:
            scenarios = json.load(f)

        topic_counts: dict[str, int] = {}
        for s in scenarios:
            topic = s.get("topic", "Unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        click.echo(f"\nTopic distribution:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
            click.echo(f"  {topic}: {count}")

        diff_counts: dict[str, int] = {}
        for s in scenarios:
            diff = s.get("difficulty", "Unknown")
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
        click.echo(f"\nDifficulty distribution:")
        for diff, count in sorted(diff_counts.items()):
            click.echo(f"  {diff}: {count}")


@cli.command()
@click.option("-o", "--output", default="results/", help="Results directory")
@click.option("--min-score", default=3.5, help="Minimum quality score (default: 3.5)")
def export(output: str, min_score: float):
    """Export filtered training data in OpenAI fine-tuning JSONL format."""
    output_dir = Path(output)

    conversations_path = output_dir / "conversations.json"
    scores_path = output_dir / "scores.json"

    if not conversations_path.exists():
        raise click.ClickException(f"No conversations found at {conversations_path}")
    if not scores_path.exists():
        raise click.ClickException(f"No scores found at {scores_path}")

    with open(conversations_path) as f:
        conversations = json.load(f)
    with open(scores_path) as f:
        scores = json.load(f)

    training_data = format_for_training(conversations, scores, min_score=min_score)

    export_path = output_dir / "training_data.jsonl"
    with open(export_path, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")

    click.echo(f"Exported {len(training_data)} training examples to {export_path}")
    click.echo(
        f"Filtered from {len(conversations)} conversations (min score: {min_score})"
    )


def main():
    cli()
