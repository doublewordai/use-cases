"""CLI for running async research agents via the batch API with web search.

Each research round interleaves realtime web search with batch analysis:
1. Generate search queries (batch)
2. Execute queries via Serper API (realtime)
3. Fetch pages via Jina Reader (realtime)
4. Analyze content (batch)
5. Generate follow-up queries (batch)
"""

import json
import os
from pathlib import Path

import click

from .agent import (
    build_analysis_requests,
    build_followup_query_requests,
    build_seed_query_requests,
    build_synthesis_request,
    execute_searches,
    extract_content,
    extract_queries,
    fetch_sources,
)
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

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


def _run_batch(
    client,
    provider: str,
    requests_data: list[dict],
    batch_file_path: Path,
    results_path: Path,
) -> dict[str, dict]:
    """Run a complete batch cycle: create file, upload, wait, download, parse."""
    create_batch_file(requests_data, batch_file_path)

    click.echo("  Uploading batch file...")
    file_id = upload_batch_file(client, batch_file_path)
    click.echo(f"  File ID: {file_id}")

    click.echo("  Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"  Batch ID: {batch_id}")

    click.echo("  Waiting for batch to complete...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed":
        raise click.ClickException(f"Batch ended with status: {batch.status}")

    click.echo("  Downloading results...")
    download_results(batch.output_file_id, results_path, provider)
    return parse_results(results_path)


@click.group()
def cli():
    """Async research agent using the Doubleword Batch API.

    Runs multi-round web research on any topic. Each round searches the web
    via Serper API, fetches pages via Jina Reader, and analyzes content via
    the batch API.
    """
    pass


@cli.command()
@click.option("--topic", required=True, help="Research topic to investigate")
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
@click.option("--rounds", default=3, help="Number of research rounds (default: 3)")
@click.option(
    "--queries-per-round",
    default=5,
    help="Search queries per round (default: 5)",
)
@click.option(
    "--pages-per-round",
    default=8,
    help="Max pages to fetch per round (default: 8)",
)
@click.option("-o", "--output", default="results/", help="Output directory")
@click.option("--dry-run", is_flag=True, help="Create batch files but don't submit")
def run(
    topic: str,
    model: str,
    provider: str,
    rounds: int,
    queries_per_round: int,
    pages_per_round: int,
    output: str,
    dry_run: bool,
):
    """Run a multi-round research agent on a topic.

    Requires SERPER_API_KEY for web search and DOUBLEWORD_API_KEY (or
    OPENAI_API_KEY) for batch inference.
    """
    # Check required env vars
    if not dry_run:
        if not os.environ.get("SERPER_API_KEY"):
            raise click.ClickException(
                "SERPER_API_KEY environment variable not set. "
                "Get a free key at https://serper.dev"
            )

    model_id = MODELS.get(model, model)
    output_dir = Path(output) / topic.lower().replace(" ", "-")[:50]
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Topic: {topic}")
    click.echo(f"Model: {model_id}")
    click.echo(f"Rounds: {rounds}")
    click.echo(f"Queries per round: {queries_per_round}")
    click.echo(f"Pages per round: {pages_per_round}")
    click.echo()

    all_findings = {}
    all_sources_metadata = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    # Step 1: Generate seed search queries (batch)
    click.echo("=" * 60)
    click.echo("Generating seed search queries...")
    click.echo("=" * 60)

    seed_requests = build_seed_query_requests(topic, model_id, count=queries_per_round)
    seed_file = output_dir / "round-0-seed-input.jsonl"
    create_batch_file(seed_requests, seed_file)

    if dry_run:
        click.echo(f"Dry run - batch file created: {seed_file}")
        click.echo("Skipping API calls. Exiting.")
        return

    client, _ = get_client(provider)

    seed_results = _run_batch(
        client,
        provider,
        seed_requests,
        seed_file,
        output_dir / "round-0-seed-output.jsonl",
    )
    tokens = count_tokens(seed_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    seed_content = extract_content(seed_results["seed-queries"])
    queries = extract_queries(seed_content)
    if not queries:
        raise click.ClickException(
            f"Failed to extract search queries.\nRaw output:\n{seed_content}"
        )

    queries = queries[:queries_per_round]
    click.echo(f"Generated {len(queries)} search queries:")
    for q in queries:
        click.echo(f"  - {q}")

    # Research rounds
    for round_num in range(rounds):
        click.echo()
        click.echo("=" * 60)
        click.echo(f"Round {round_num}: Searching & analyzing")
        click.echo("=" * 60)

        # Step 2: Execute web searches (realtime)
        click.echo(f"\nSearching the web ({len(queries)} queries)...")
        search_data = execute_searches(queries, results_per_query=5)
        click.echo(
            f"  Found {len(search_data['all_results'])} results, "
            f"{len(search_data['urls'])} unique URLs"
        )

        # Step 3: Fetch pages (realtime)
        click.echo(f"Fetching top {pages_per_round} pages...")
        sources = fetch_sources(
            search_data["urls"],
            search_data["all_results"],
            max_pages=pages_per_round,
        )
        click.echo(f"  Successfully fetched {len(sources)} pages")

        if not sources:
            click.echo("  No pages fetched, skipping analysis for this round")
            continue

        # Save source metadata
        round_sources = []
        for s in sources:
            meta = {"url": s["url"], "title": s["title"], "round": round_num}
            round_sources.append(meta)
            all_sources_metadata.append(meta)

        with open(output_dir / f"round-{round_num}-sources.json", "w") as f:
            json.dump(round_sources, f, indent=2)

        # Step 4: Analyze fetched content (batch)
        click.echo(f"Analyzing {len(sources)} sources via batch API...")
        analysis_requests = build_analysis_requests(sources, topic, model_id, round_num)
        analysis_results = _run_batch(
            client,
            provider,
            analysis_requests,
            output_dir / f"round-{round_num}-analysis-input.jsonl",
            output_dir / f"round-{round_num}-analysis-output.jsonl",
        )
        tokens = count_tokens(analysis_results)
        total_tokens["input_tokens"] += tokens["input_tokens"]
        total_tokens["output_tokens"] += tokens["output_tokens"]

        round_findings = {
            cid: extract_content(r) for cid, r in analysis_results.items()
        }
        all_findings.update(round_findings)
        click.echo(f"  Analyzed {len(round_findings)} sources")

        # Step 5: Generate follow-up queries for next round (batch)
        if round_num < rounds - 1:
            click.echo("\nGenerating follow-up search queries...")
            findings_text = "\n\n".join(
                f"[{cid}]: {text[:500]}" for cid, text in round_findings.items()
            )
            followup_requests = build_followup_query_requests(
                findings_text,
                topic,
                model_id,
                round_num + 1,
                count=queries_per_round,
            )
            followup_results = _run_batch(
                client,
                provider,
                followup_requests,
                output_dir / f"round-{round_num}-followup-input.jsonl",
                output_dir / f"round-{round_num}-followup-output.jsonl",
            )
            tokens = count_tokens(followup_results)
            total_tokens["input_tokens"] += tokens["input_tokens"]
            total_tokens["output_tokens"] += tokens["output_tokens"]

            followup_content = extract_content(
                followup_results[f"round-{round_num + 1}-queries"]
            )
            queries = extract_queries(followup_content)
            if not queries:
                click.echo("  No follow-up queries extracted, stopping early.")
                break
            queries = queries[:queries_per_round]
            click.echo(f"  Generated {len(queries)} follow-up queries:")
            for q in queries:
                click.echo(f"    - {q}")

    # Synthesis
    click.echo()
    click.echo("=" * 60)
    click.echo("Synthesizing final report")
    click.echo("=" * 60)

    synthesis_requests = build_synthesis_request(all_findings, topic, model_id, rounds)
    synthesis_results = _run_batch(
        client,
        provider,
        synthesis_requests,
        output_dir / "synthesis-input.jsonl",
        output_dir / "synthesis-output.jsonl",
    )
    tokens = count_tokens(synthesis_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    report_text = extract_content(synthesis_results["synthesis"])

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    click.echo(f"Report saved to: {report_path}")

    # Save summary
    summary = {
        "topic": topic,
        "model": model_id,
        "provider": provider,
        "rounds": rounds,
        "queries_per_round": queries_per_round,
        "pages_per_round": pages_per_round,
        "total_sources_fetched": len(all_sources_metadata),
        "total_findings": len(all_findings),
        "tokens": total_tokens,
        "sources": all_sources_metadata,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo()
    click.echo(f"Sources fetched: {len(all_sources_metadata)}")
    click.echo(f"Analyses completed: {len(all_findings)}")
    click.echo(
        f"Tokens used - Input: {total_tokens['input_tokens']:,}, "
        f"Output: {total_tokens['output_tokens']:,}"
    )


@cli.command()
@click.option("--batch-id", required=True, help="Batch ID to check")
@click.option(
    "-p",
    "--provider",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
def status(batch_id: str, provider: str):
    """Check the status of a batch job."""
    client, _ = get_client(provider)
    batch = client.batches.retrieve(batch_id)

    click.echo(f"Batch ID: {batch.id}")
    click.echo(f"Status: {batch.status}")
    click.echo(
        f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
    )


@cli.command()
@click.option("-o", "--output", default="results/", help="Results directory")
def report(output: str):
    """Print the final synthesis report from a completed research run."""
    output_dir = Path(output)
    if not output_dir.exists():
        raise click.ClickException(f"No results directory found: {output_dir}")

    for topic_dir in sorted(output_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        report_path = topic_dir / "report.md"
        summary_path = topic_dir / "summary.json"

        if report_path.exists() and summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            click.echo(f"\n{'=' * 60}")
            click.echo(f"Topic: {summary['topic']}")
            click.echo(f"Model: {summary['model']}")
            click.echo(f"Rounds: {summary['rounds']}")
            click.echo(f"Sources: {summary.get('total_sources_fetched', 'N/A')}")
            click.echo(f"Analyses: {summary['total_findings']}")
            click.echo(
                f"Tokens: {summary['tokens']['input_tokens']:,} in / "
                f"{summary['tokens']['output_tokens']:,} out"
            )
            click.echo(f"{'=' * 60}\n")

            with open(report_path) as f:
                click.echo(f.read())


def main():
    cli()
