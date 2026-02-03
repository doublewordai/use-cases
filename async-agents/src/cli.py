"""CLI for running parallel tool-calling research agents via the batch API.

Architecture:
1. Generate sub-queries for a topic (small batch or inline)
2. Create N agents, one per sub-query, each with tool-calling capability
3. Orchestrator loop: submit batch -> poll -> process -> execute tools -> repeat
4. When all agents complete, run synthesis batch for final report
"""

import json
import re
from pathlib import Path

import click

from .agent import (
    Agent,
    all_done,
    build_batch_requests,
    create_agents,
    execute_pending_tools,
    get_ready_agents,
    process_responses,
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
from .prompts import (
    SUB_QUERY_PROMPT,
    SUB_QUERY_SYSTEM,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM,
)

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"
MAX_ITERATIONS = 10


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


def _extract_queries(text: str) -> list[str]:
    """Parse numbered queries from model output."""
    queries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        match = re.match(r"^\d+[.):\-]\s*(.+)$", line)
        if match:
            query = match.group(1).strip().strip('"').strip("'")
            if query:
                queries.append(query)
    return queries


@click.group()
def cli():
    """Parallel tool-calling research agents via the Doubleword Batch API.

    Spawns multiple research agents that independently search the web and
    read pages using tool calling. All agents run in parallel within each
    batch round. The model decides what tools to call and when to stop.
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
    "--agents",
    "num_agents",
    default=5,
    help="Number of parallel research agents (default: 5)",
)
@click.option(
    "--max-iterations",
    default=MAX_ITERATIONS,
    help=f"Max tool-calling iterations per agent (default: {MAX_ITERATIONS})",
)
@click.option("-o", "--output", default="results/", help="Output directory")
@click.option(
    "--dry-run", is_flag=True, help="Create initial batch file but don't submit"
)
def run(
    topic: str,
    model: str,
    provider: str,
    num_agents: int,
    max_iterations: int,
    output: str,
    dry_run: bool,
):
    """Run parallel research agents on a topic.

    First generates sub-queries to cover different angles of the topic,
    then launches one agent per sub-query. Each agent independently
    searches the web and reads pages using tool calling. The model
    decides what to search, what to read, and when it has enough
    information.

    Requires SERPER_API_KEY for web search and DOUBLEWORD_API_KEY (or
    OPENAI_API_KEY) for batch inference.
    """
    model_id = MODELS.get(model, model)
    output_dir = Path(output) / topic.lower().replace(" ", "-")[:50]
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Topic: {topic}")
    click.echo(f"Model: {model_id}")
    click.echo(f"Agents: {num_agents}")
    click.echo(f"Max iterations: {max_iterations}")
    click.echo()

    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    # --- Step 1: Generate sub-queries ---
    click.echo("=" * 60)
    click.echo("Step 1: Generating research sub-queries")
    click.echo("=" * 60)

    sub_query_requests = [
        {
            "custom_id": "sub-queries",
            "model": model_id,
            "messages": [
                {"role": "system", "content": SUB_QUERY_SYSTEM},
                {
                    "role": "user",
                    "content": SUB_QUERY_PROMPT.format(topic=topic, count=num_agents),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    ]

    sub_query_file = output_dir / "sub-queries-input.jsonl"
    create_batch_file(sub_query_requests, sub_query_file)

    if dry_run:
        click.echo(f"\nDry run — sub-query batch file: {sub_query_file}")
        # Also create the initial agent batch to show tool definitions
        dummy_queries = [f"Sub-query {i + 1} for: {topic}" for i in range(num_agents)]
        agents = create_agents(dummy_queries, model_id)
        agent_requests = build_batch_requests(agents)
        agent_file = output_dir / "agents-iter-0-input.jsonl"
        create_batch_file(agent_requests, agent_file)
        click.echo(f"Agent batch file (with tools): {agent_file}")
        click.echo(
            f"\nTool definitions included: {len(agent_requests[0].get('tools', []))} tools"
        )
        for tool in agent_requests[0].get("tools", []):
            click.echo(
                f"  - {tool['function']['name']}: {tool['function']['description'][:60]}..."
            )
        click.echo("\nSkipping API calls. Exiting.")
        return

    client, _ = get_client(provider)

    sub_query_results = _run_batch(
        client,
        provider,
        sub_query_requests,
        sub_query_file,
        output_dir / "sub-queries-output.jsonl",
    )
    tokens = count_tokens(sub_query_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    sub_query_text = extract_content(sub_query_results["sub-queries"])
    sub_queries = _extract_queries(sub_query_text)
    if not sub_queries:
        raise click.ClickException(
            f"Failed to extract sub-queries.\nRaw output:\n{sub_query_text}"
        )
    sub_queries = sub_queries[:num_agents]

    click.echo(f"\nGenerated {len(sub_queries)} sub-queries:")
    for i, q in enumerate(sub_queries):
        click.echo(f"  {i + 1}. {q}")

    # --- Step 2: Create agents and run orchestrator loop ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 2: Running parallel research agents")
    click.echo("=" * 60)

    agents = create_agents(sub_queries, model_id)
    iteration = 0

    while not all_done(agents) and iteration < max_iterations:
        ready = get_ready_agents(agents)
        if not ready:
            # All in-progress agents have tool calls pending
            # This shouldn't happen if execute_pending_tools ran, but guard against it
            break

        click.echo(f"\n--- Iteration {iteration} ---")
        active_ids = [a.id for a in ready]
        completed = sum(1 for a in agents if a.status == "completed")
        click.echo(
            f"  Active agents: {len(ready)}/{len(agents)} (completed: {completed})"
        )

        # Build and submit batch for all ready agents
        batch_requests = build_batch_requests(ready)
        if not batch_requests:
            break

        batch_results = _run_batch(
            client,
            provider,
            batch_requests,
            output_dir / f"agents-iter-{iteration}-input.jsonl",
            output_dir / f"agents-iter-{iteration}-output.jsonl",
        )
        tokens = count_tokens(batch_results)
        total_tokens["input_tokens"] += tokens["input_tokens"]
        total_tokens["output_tokens"] += tokens["output_tokens"]

        # Process responses — updates agent states
        process_responses(agents, batch_results)

        completed_this_round = sum(
            1
            for a in agents
            if a.status == "completed" and a.iteration == (iteration + 1)
            # iteration was incremented in process_responses
        )
        if completed_this_round:
            click.echo(f"  {completed_this_round} agent(s) completed this round")

        # Execute tool calls for agents that need them
        execute_pending_tools(agents)

        iteration += 1

    # Report agent statuses
    click.echo()
    for agent in agents:
        status_icon = "done" if agent.status == "completed" else agent.status
        click.echo(
            f"  {agent.id}: {status_icon} after {agent.iteration} iterations — {agent.sub_query[:60]}"
        )

    # --- Step 3: Synthesis ---
    click.echo()
    click.echo("=" * 60)
    click.echo("Step 3: Synthesizing final report")
    click.echo("=" * 60)

    completed_agents = [a for a in agents if a.status == "completed" and a.findings]
    if not completed_agents:
        raise click.ClickException("No agents completed successfully.")

    findings_text = "\n\n".join(
        f"--- Agent {a.id}: {a.sub_query} ---\n{a.findings}" for a in completed_agents
    )

    synthesis_requests = [
        {
            "custom_id": "synthesis",
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        topic=topic,
                        num_agents=len(completed_agents),
                        findings=findings_text,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
    ]

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
    click.echo(f"\nReport saved to: {report_path}")

    # Save agent conversation logs
    agent_logs = []
    for agent in agents:
        agent_logs.append(
            {
                "id": agent.id,
                "sub_query": agent.sub_query,
                "status": agent.status,
                "iterations": agent.iteration,
                "message_count": len(agent.messages),
            }
        )
    with open(output_dir / "agent-logs.json", "w") as f:
        json.dump(agent_logs, f, indent=2)

    # Save summary
    summary = {
        "topic": topic,
        "model": model_id,
        "provider": provider,
        "num_agents": len(agents),
        "max_iterations": max_iterations,
        "agents_completed": sum(1 for a in agents if a.status == "completed"),
        "agents_failed": sum(1 for a in agents if a.status == "failed"),
        "total_batch_rounds": iteration,
        "tokens": total_tokens,
        "sub_queries": [a.sub_query for a in agents],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo()
    click.echo(f"Agents completed: {summary['agents_completed']}/{len(agents)}")
    click.echo(f"Batch rounds: {iteration}")
    click.echo(
        f"Tokens used — Input: {total_tokens['input_tokens']:,}, "
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
            click.echo(
                f"Agents: {summary.get('num_agents', 'N/A')} ({summary.get('agents_completed', 'N/A')} completed)"
            )
            click.echo(f"Batch rounds: {summary.get('total_batch_rounds', 'N/A')}")
            click.echo(
                f"Tokens: {summary['tokens']['input_tokens']:,} in / "
                f"{summary['tokens']['output_tokens']:,} out"
            )
            click.echo(f"{'=' * 60}\n")

            with open(report_path) as f:
                click.echo(f.read())


def main():
    cli()
