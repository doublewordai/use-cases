"""CLI for recursive tool-calling research agents via the batch API.

The entire research process is driven by a single root agent that can
recursively spawn sub-agents. No hardcoded sub-query generation or
synthesis — the model decides everything via tool calls.

Architecture:
1. Create root agent with topic and tools (web_search, fetch_page, spawn_agents, write_report)
2. Orchestrator loop: submit batch → poll → process → execute tools → resolve parents → repeat
3. Root agent calls write_report when done → output the report
"""

import json
from pathlib import Path

import click

from .agent import (
    AgentRegistry,
    execute_pending_tools,
    process_responses,
    resolve_waiting_parents,
)
from .batch import (
    count_tokens,
    create_batch,
    create_batch_file,
    download_results,
    extract_content,
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
MAX_ITERATIONS = 25


def _run_batch(
    client,
    provider: str,
    requests_data: list[dict],
    batch_file_path: Path,
    results_path: Path,
) -> dict[str, dict]:
    """Run a complete batch cycle: create file, upload, wait, download, parse."""
    create_batch_file(requests_data, batch_file_path)
    file_id = upload_batch_file(client, batch_file_path)
    batch_id = create_batch(client, file_id)
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed":
        raise click.ClickException(f"Batch ended with status: {batch.status}")

    download_results(batch.output_file_id, results_path, provider)
    return parse_results(results_path)


@click.group()
def cli():
    """Recursive tool-calling research agents via the Doubleword Batch API.

    A root agent breaks a topic into sub-queries, spawning parallel
    sub-agents that each independently search the web and read pages.
    Sub-agents can recursively spawn their own sub-agents. The root
    agent synthesizes all findings into a final report.
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
@click.option(
    "--max-iterations",
    default=MAX_ITERATIONS,
    help=f"Max batch rounds (default: {MAX_ITERATIONS})",
)
@click.option(
    "--max-depth",
    default=5,
    help="Max agent tree depth — how many levels of sub-agents (default: 5)",
)
@click.option(
    "--max-agent-iterations",
    default=8,
    help="Max batch rounds per individual agent (default: 8)",
)
@click.option("-o", "--output", default="results/", help="Output directory")
@click.option(
    "--dry-run", is_flag=True, help="Create initial batch file but don't submit"
)
def run(
    topic: str,
    model: str,
    provider: str,
    max_iterations: int,
    max_depth: int,
    max_agent_iterations: int,
    output: str,
    dry_run: bool,
):
    """Run a recursive research agent on a topic.

    A single root agent is created with tools for web search, page
    fetching, spawning sub-agents, and writing the final report. The
    model decides how to break down the topic, what to search, when
    to delegate, and when to synthesize.

    Requires SERPER_API_KEY for web search and DOUBLEWORD_API_KEY (or
    OPENAI_API_KEY) for batch inference.
    """
    model_id = MODELS.get(model, model)
    output_dir = Path(output) / topic.lower().replace(" ", "-")[:50]
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Topic: {topic}")
    click.echo(f"Model: {model_id}")
    click.echo(
        f"Max iterations: {max_iterations} (per agent: {max_agent_iterations}, max depth: {max_depth})"
    )
    click.echo()

    # Create the agent tree with a single root
    registry = AgentRegistry(
        max_depth=max_depth, max_agent_iterations=max_agent_iterations
    )
    root = registry.create_root(topic, model_id)

    # Dry run — show the initial batch file
    if dry_run:
        ready = registry.get_ready_agents()
        requests_data = registry.build_batch_requests(ready)
        batch_file = output_dir / "root-iter-0-input.jsonl"
        create_batch_file(requests_data, batch_file)

        click.echo(f"Dry run — batch file: {batch_file}")
        click.echo(f"Root agent: {root.id}")
        tools = requests_data[0].get("tools", [])
        click.echo(f"Tools ({len(tools)}):")
        for tool in tools:
            click.echo(
                f"  - {tool['function']['name']}: {tool['function']['description'][:60]}..."
            )
        click.echo("\nSkipping API calls. Exiting.")
        # Reset root status so it's clean
        root.status = "pending"
        return

    client, _ = get_client(provider)
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    iteration = 0

    # --- Main orchestrator loop ---
    while not registry.root_done() and iteration < max_iterations:
        ready = registry.get_ready_agents()
        if not ready:
            # Could be waiting for children — check if any are still active
            counts = registry.agent_count()
            if counts.get("waiting_for_children", 0) > 0:
                # This shouldn't normally happen (children should be ready),
                # but guard against it
                click.echo("  All agents waiting — possible deadlock, stopping")
            break

        # Print tree at start of each round
        click.echo()
        registry.print_tree(iteration=iteration)

        # Clear last_tool so it only shows for one round
        for a in registry.agents.values():
            a.last_tool = ""

        # 1. Build and submit batch
        requests_data = registry.build_batch_requests(ready)
        batch_results = _run_batch(
            client,
            provider,
            requests_data,
            output_dir / f"iter-{iteration}-input.jsonl",
            output_dir / f"iter-{iteration}-output.jsonl",
        )
        tokens = count_tokens(batch_results)
        total_tokens["input_tokens"] += tokens["input_tokens"]
        total_tokens["output_tokens"] += tokens["output_tokens"]

        # 2. Process responses
        process_responses(registry, batch_results)

        # 3. Execute tools (immediate + deferred)
        execute_pending_tools(registry)

        # 4. Resolve waiting parents
        resolve_waiting_parents(registry)

        iteration += 1

    # --- Force-complete if root didn't finish ---
    if not registry.root_done():
        registry.force_complete_all()

        # Give the root one final batch round with NO tools — the model
        # writes the report directly as its response content.
        root = registry.get_root()
        if root.status == "in_progress":
            requests_data = registry.build_final_report_request(root)
            click.echo()
            registry.print_tree(iteration=iteration)
            batch_results = _run_batch(
                client,
                provider,
                requests_data,
                output_dir / f"iter-{iteration}-final-input.jsonl",
                output_dir / f"iter-{iteration}-final-output.jsonl",
            )
            tokens = count_tokens(batch_results)
            total_tokens["input_tokens"] += tokens["input_tokens"]
            total_tokens["output_tokens"] += tokens["output_tokens"]

            # Extract the report directly from the response content
            for result in batch_results.values():
                content = extract_content(result)
                if content:
                    root.report = content
                    root.findings = content
                    root.status = "completed"
                    root.iteration += 1

    # --- Output ---
    click.echo()
    registry.print_tree()

    # Save report
    root = registry.get_root()
    if root.report:
        report_path = output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(root.report)
        click.echo(f"\nReport saved to: {report_path}")
    elif root.findings:
        # Root completed with stop rather than write_report
        report_path = output_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(root.findings)
        click.echo(f"\nReport saved to: {report_path}")
    else:
        click.echo("\nNo report produced — root agent did not complete.")

    # Save agent tree log
    agent_logs = []
    for agent in registry.agents.values():
        agent_logs.append(
            {
                "id": agent.id,
                "status": agent.status,
                "depth": agent.depth,
                "parent_id": agent.parent_id,
                "children": agent.children_ids,
                "iterations": agent.iteration,
                "is_root": agent.is_root,
                "message_count": len(agent.messages),
            }
        )
    with open(output_dir / "agent-tree.json", "w") as f:
        json.dump(agent_logs, f, indent=2)

    # Save summary
    summary = {
        "topic": topic,
        "model": model_id,
        "provider": provider,
        "max_iterations": max_iterations,
        "batch_rounds": iteration,
        "total_agents": len(registry.agents),
        "agents_completed": sum(
            1 for a in registry.agents.values() if a.status == "completed"
        ),
        "agents_failed": sum(
            1 for a in registry.agents.values() if a.status == "failed"
        ),
        "max_depth": max((a.depth for a in registry.agents.values()), default=0),
        "tokens": total_tokens,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo()
    click.echo(
        f"Total agents: {summary['total_agents']} (completed: {summary['agents_completed']}, failed: {summary['agents_failed']})"
    )
    click.echo(f"Max depth: {summary['max_depth']}")
    click.echo(f"Batch rounds: {iteration}")
    click.echo(
        f"Tokens — Input: {total_tokens['input_tokens']:,}, "
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
    """Print the final report from a completed research run."""
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
            click.echo(
                f"Agents: {summary['total_agents']} ({summary['agents_completed']} completed)"
            )
            click.echo(f"Max depth: {summary['max_depth']}")
            click.echo(f"Batch rounds: {summary['batch_rounds']}")
            click.echo(
                f"Tokens: {summary['tokens']['input_tokens']:,} in / "
                f"{summary['tokens']['output_tokens']:,} out"
            )
            click.echo(f"{'=' * 60}\n")

            with open(report_path) as f:
                click.echo(f.read())


def main():
    cli()
