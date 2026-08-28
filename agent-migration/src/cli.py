"""CLI for the agent-migration workbook.

Data prep, scoring, dataset generation, and the shadow demo live here. Batch
submission and cost analytics are done with the `dw` CLI, wired together in
dw.toml. Run `dw project info` to see the steps.
"""
import click

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@click.group()
def cli():
    """Evaluate open-weight candidates against an incumbent on a triage agent."""


@cli.command()
@click.option("--model", default="google/gemma-4-31B-it", help="Model used to write the synthetic tickets")
def generate(model):
    """Regenerate the synthetic golden dataset (ships frozen in data/)."""
    from .generate import generate as run_generate

    n, path = run_generate(model)
    click.echo(f"Wrote {n} labelled tickets to {path}")


@cli.command()
@click.option("--output", "-o", default="batches/batch.jsonl", help="Batch JSONL output path")
@click.option("--max-tokens", default=4096, type=int, help="Completion budget per request")
def prepare(output, max_tokens):
    """Build the model-agnostic batch file from the golden dataset."""
    from .prepare import build

    n, path = build(output, max_tokens)
    click.echo(f"Wrote {path} ({n} requests, no model set)")
    click.echo("Stamp a model with `dw files prepare <file> --model <name>`, then run `dw project info`.")


@cli.command()
@click.option("--model", default="gpt-5.6-sol", help="Closed model on OpenAI (realtime)")
@click.option("--concurrency", default=8, type=int, help="Concurrent requests")
@click.option("--price-in", default=5.0, type=float, help="OpenAI input price per 1M tokens")
@click.option("--price-out", default=30.0, type=float, help="OpenAI output price per 1M tokens")
@click.option("--limit", default=None, type=int, help="Only run the first N tickets")
@click.option("--reasoning-effort", default="high", help="OpenAI reasoning effort (e.g. low, medium, high, xhigh)")
def baseline(model, concurrency, price_in, price_out, limit, reasoning_effort):
    """Baseline against a closed model on OpenAI, realtime. Needs OPENAI_API_KEY."""
    from .baseline import run

    m = run(model, concurrency=concurrency, price_in=price_in, price_out=price_out, limit=limit, reasoning_effort=reasoning_effort)
    click.echo(f"Ran {m['n']} tickets on {model} -> {m['out']}")
    click.echo(f"Tokens: {m['input_tokens']:,} in / {m['output_tokens']:,} out")
    click.echo(f"Avg latency: {m['avg_latency_ms']} ms")
    click.echo(f"Cost: ${m['cost']} (at ${price_in} / ${price_out} per 1M)")


@cli.command()
@click.option("--results", "-r", required=True, help="Results JSONL from `dw batches results`")
@click.option("--label", "-l", default=None, help="Display label for this model")
def score(results, label):
    """Score a results file against the golden labels."""
    from .scoring import score as run_score

    m = run_score(results, label)
    click.echo("=" * 52)
    click.echo(f"{m['label']}")
    click.echo("=" * 52)
    click.echo(f"Category accuracy : {m['category_accuracy']:.1%}  ({int(m['category_accuracy'] * m['n'])}/{m['n']})")
    click.echo(f"Priority accuracy : {m['priority_accuracy']:.1%}")
    click.echo(f"Escalation acc.   : {m['needs_human_accuracy']:.1%}")
    click.echo(f"Schema valid      : {m['schema_valid']}/{m['n']}   enum valid: {m['enum_valid']}/{m['n']}")
    click.echo(f"Tokens            : {m['input_tokens']:,} in / {m['output_tokens']:,} out")


@cli.command()
@click.option("--incumbent", default="gpt-5.6-sol", help="Closed incumbent on OpenAI (or set FLAGSHIP_*)")
@click.option("--candidate", default="moonshotai/kimi-k3", help="Open-weight candidate to shadow")
@click.option("--n", default=5, type=int, help="Number of tickets to shadow")
def shadow(incumbent, candidate, n):
    """Shadow the candidate against the incumbent on a few live tickets."""
    from .shadow import run

    count, out = run(incumbent, candidate, n)
    click.echo(f"Shadowed {count} tickets, logged pairs to {out}")


def main():
    cli()
