"""CLI for the Doubleword batch eval pipeline.

Three steps, designed to slot into the `dw` workflow (see dw.toml):

  prepare        download TruthfulQA, emit the generation batch, upload the Phoenix Dataset
  prepare-judge  turn generated answers into the judge batch
  analyze        score the judge results, record a Phoenix Experiment, report token usage

Batch submission / polling / retrieval is the `dw` CLI's job, not ours.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
from wasabi import msg

from . import data
from .config import settings
from .judge import Score

DATASET_DISPLAY_NAME = "truthfulqa-eval"


@click.group()
def cli() -> None:
    """Doubleword batch LLM-as-judge evals, visualised in Arize Phoenix."""


@cli.command()
@click.option("--limit", "-n", default=None, type=int, help="How many questions (default: all 817).")
@click.option("--out-dir", "-o", default="batches", help="Output directory for batch files.")
@click.option("--max-tokens", default=512, type=int, help="max_tokens for generation.")
def prepare(limit: int | None, out_dir: str, max_tokens: int) -> None:
    """Download TruthfulQA, write the generation batch, upload the Phoenix Dataset."""
    rows = data.load_truthfulqa(limit=limit)
    msg.info(f"Loaded {len(rows)} TruthfulQA questions.")

    gen_path = Path(out_dir) / "generate.jsonl"
    n = data.write_jsonl(
        gen_path, (data.build_generation_request(r, max_tokens=max_tokens) for r in rows)
    )
    gt_path = Path(out_dir) / "ground_truth.json"
    data.save_ground_truth(gt_path, rows)
    msg.good(f"Wrote {n} generation requests → {gen_path}")
    msg.good(f"Saved ground truth → {gt_path}")

    # Create the Phoenix Dataset up front so it's visible before the batch runs.
    try:
        from . import phoenix_io

        phoenix_io.create_eval_dataset(rows, name=DATASET_DISPLAY_NAME)
        msg.good(
            f"Uploaded Phoenix Dataset '{DATASET_DISPLAY_NAME}' → "
            f"{settings.phoenix_collector_endpoint}"
        )
    except Exception as exc:  # Phoenix not up yet shouldn't block batch prep
        msg.warn(f"Could not upload Phoenix Dataset (is Phoenix running?): {exc}")

    msg.text("\nNext: dw files prepare → dw batches run → dw batches results")


@cli.command(name="prepare-judge")
@click.option("--answers", "-a", required=True, help="Generation results JSONL (from dw).")
@click.option("--ground-truth", "-g", default="batches/ground_truth.json")
@click.option("--out-dir", "-o", default="batches")
@click.option("--max-tokens", default=512, type=int, help="max_tokens for the judge.")
def prepare_judge(answers: str, ground_truth: str, out_dir: str, max_tokens: int) -> None:
    """Build the judge batch from generated answers + ground-truth references."""
    gt = data.load_ground_truth(ground_truth)
    results = data.index_by_custom_id(data.read_jsonl(answers))

    lines, skipped = [], 0
    for cid, row in gt.items():
        result = results.get(cid)
        content = data.parse_content(result) if result else None
        if not content:
            skipped += 1
            continue
        lines.append(
            data.build_judge_request(
                cid, row.question, content, reference=row.reference(), max_tokens=max_tokens
            )
        )

    judge_path = Path(out_dir) / "judge.jsonl"
    n = data.write_jsonl(judge_path, lines)
    msg.good(f"Wrote {n} judge requests → {judge_path}")
    if skipped:
        msg.warn(f"Skipped {skipped} questions with no generated answer.")
    msg.text("\nNext: dw files prepare → dw batches run → dw batches results")


@cli.command()
@click.option("--answers", "-a", required=True, help="Generation results JSONL (from dw).")
@click.option("--scores", "-s", required=True, help="Judge results JSONL (from dw).")
@click.option("--ground-truth", "-g", default="batches/ground_truth.json")
@click.option("--out-dir", "-o", default="results")
@click.option("--dataset-name", default=DATASET_DISPLAY_NAME)
@click.option("--no-phoenix", is_flag=True, help="Skip logging the Phoenix Experiment.")
def analyze(
    answers: str,
    scores: str,
    ground_truth: str,
    out_dir: str,
    dataset_name: str,
    no_phoenix: bool,
) -> None:
    """Score judge results, log a Phoenix Experiment, and report token usage."""
    answer_results = data.index_by_custom_id(data.read_jsonl(answers))
    score_results = data.index_by_custom_id(data.read_jsonl(scores))

    answers_by_id: dict[str, str] = {}
    scores_by_id: dict[str, Score] = {}
    parse_errors = 0
    for cid, result in score_results.items():
        content = data.parse_content(result)
        if not content:
            parse_errors += 1
            continue
        try:
            scores_by_id[cid] = Score.model_validate_json(content)
        except Exception:
            parse_errors += 1
            continue
        ans = data.parse_content(answer_results.get(cid, {}))
        answers_by_id[cid] = ans or ""

    if not scores_by_id:
        raise click.ClickException("No parsable judge scores found.")

    metrics = _aggregate(scores_by_id)
    gen_in, gen_out = _sum_tokens(answer_results)
    judge_in, judge_out = _sum_tokens(score_results)

    summary = {
        "n_scored": len(scores_by_id),
        "parse_errors": parse_errors,
        "metrics": metrics,
        "tokens": {
            "generation": {"input": gen_in, "output": gen_out},
            "judge": {"input": judge_in, "output": judge_out},
        },
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_file = out_path / f"scored_{stamp}.json"
    with open(scored_file, "w") as f:
        json.dump(summary, f, indent=2)

    _print_report(summary)
    msg.good(f"\nSaved summary → {scored_file}")

    if not no_phoenix:
        try:
            from . import phoenix_io

            exp = phoenix_io.log_judge_experiment(
                dataset_name=dataset_name,
                answers_by_id=answers_by_id,
                scores_by_id=scores_by_id,
                experiment_name=f"judge-{stamp}",
            )
            msg.good(f"Logged Phoenix Experiment → {settings.phoenix_collector_endpoint}")
            url = getattr(exp, "url", None)
            if url:
                msg.text(url)
        except Exception as exc:
            msg.warn(f"Could not log Phoenix Experiment (is Phoenix running?): {exc}")


def _aggregate(scores_by_id: dict[str, Score]) -> dict[str, float]:
    n = len(scores_by_id)
    s = list(scores_by_id.values())
    return {
        "relevance": sum(x.relevance for x in s) / n,
        "truthfulness": sum(1.0 - x.hallucination_risk for x in s) / n,
        "tone": sum(x.tone for x in s) / n,
        "overall": sum(x.overall for x in s) / n,
    }


def _sum_tokens(results: dict[str, dict]) -> tuple[int, int]:
    tin = tout = 0
    for r in results.values():
        i, o = data.parse_usage(r)
        tin += i
        tout += o
    return tin, tout


def _print_report(summary: dict) -> None:
    m = summary["metrics"]
    click.echo("\n" + "=" * 56)
    click.echo("LLM-AS-JUDGE RESULTS (TruthfulQA)")
    click.echo("=" * 56)
    click.echo(f"Scored:        {summary['n_scored']} answers")
    click.echo(f"Relevance:     {m['relevance']:.3f}")
    click.echo(f"Truthfulness:  {m['truthfulness']:.3f}")
    click.echo(f"Tone:          {m['tone']:.3f}")
    click.echo(f"Overall:       {m['overall']:.3f}")

    t = summary["tokens"]
    click.echo("\nToken usage (this run):")
    click.echo(f"  generation  {t['generation']['input']:,} in / {t['generation']['output']:,} out")
    click.echo(f"  judge       {t['judge']['input']:,} in / {t['judge']['output']:,} out")
    click.echo(
        "\nCost: run `dw batches analytics --from-file <id>` for the authoritative"
        " per-batch cost."
    )
    click.echo("Batch runs on Doubleword's high-throughput backend — see doubleword.ai/pricing.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
