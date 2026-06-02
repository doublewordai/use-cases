"""Heavy LLM-as-judge eval on Doubleword's batch tier, visualised in Arize Phoenix.

Prereqs: a local Phoenix (`docker compose up -d`) and `DOUBLEWORD_API_KEY` set.
Then:

    uv run python eval.py            # full TruthfulQA (817 questions)
    uv run python eval.py -n 50      # quick run

What it does, end to end, in one process:
  1. load TruthfulQA,
  2. *generate* answers as one Doubleword batch,
  3. *judge* every answer as a second Doubleword batch (the expensive part),
  4. push the eval set + scores to Phoenix as a Dataset + Experiment,
  5. report token usage (authoritative batch cost: `dw batches analytics`).

autobatcher.BatchOpenAI collects every `chat.completions.create` call inside the
`async with` block and submits them as a single batch job — no manual file
upload or polling. The `dw` CLI workflow (see dw.toml / README) does the same
thing file-first; this script is the all-in-one version.
"""

from __future__ import annotations

import argparse
import asyncio

from autobatcher import BatchOpenAI
from wasabi import msg

from src import data
from src.cli import DATASET_DISPLAY_NAME
from src.config import settings
from src.judge import Score, build_judge_messages


def _client() -> BatchOpenAI:
    return BatchOpenAI(
        api_key=settings.doubleword_api_key,
        base_url=settings.doubleword_base_url,
        completion_window=settings.batch_completion_window,
    )


async def _generate(rows: list[data.Row]) -> tuple[dict[str, str], int, int]:
    answers: dict[str, str] = {}
    tin = tout = 0
    async with _client() as client:
        tasks = {
            r.id: client.chat.completions.create(
                model=settings.model_chat,
                messages=[
                    {"role": "system", "content": data.GENERATION_SYSTEM},
                    {"role": "user", "content": r.question},
                ],
                temperature=0,
                max_tokens=512,
            )
            for r in rows
        }
        done = await asyncio.gather(*tasks.values())
    for cid, resp in zip(tasks.keys(), done):
        answers[cid] = resp.choices[0].message.content or ""
        if resp.usage:
            tin += resp.usage.prompt_tokens
            tout += resp.usage.completion_tokens
    return answers, tin, tout


async def _judge(rows: list[data.Row], answers: dict[str, str]) -> tuple[dict[str, Score], int, int]:
    scores: dict[str, Score] = {}
    tin = tout = 0
    async with _client() as client:
        tasks = {
            r.id: client.chat.completions.create(
                model=settings.model_judge,
                messages=build_judge_messages(r.question, answers.get(r.id, ""), r.reference()),
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            for r in rows
            if answers.get(r.id)
        }
        done = await asyncio.gather(*tasks.values())
    for cid, resp in zip(tasks.keys(), done):
        try:
            scores[cid] = Score.model_validate_json(resp.choices[0].message.content or "{}")
        except Exception:
            continue
        if resp.usage:
            tin += resp.usage.prompt_tokens
            tout += resp.usage.completion_tokens
    return scores, tin, tout


async def main(limit: int | None) -> None:
    rows = data.load_truthfulqa(limit=limit)
    msg.info(f"Loaded {len(rows)} TruthfulQA questions.")

    msg.info("Batch 1/2 — generating answers on Doubleword batch...")
    answers, gen_in, gen_out = await _generate(rows)
    msg.good(f"Generated {len(answers)} answers.")

    msg.info("Batch 2/2 — judging answers on Doubleword batch (the expensive workload)...")
    scores, judge_in, judge_out = await _judge(rows, answers)
    msg.good(f"Judged {len(scores)} answers.")

    # Push to Phoenix as a Dataset + Experiment.
    from src import phoenix_io

    phoenix_io.create_eval_dataset(rows, name=DATASET_DISPLAY_NAME)
    phoenix_io.log_judge_experiment(
        dataset_name=DATASET_DISPLAY_NAME,
        answers_by_id=answers,
        scores_by_id=scores,
        experiment_name="judge-eval-py",
    )
    msg.good(f"Phoenix Dataset + Experiment → {settings.phoenix_collector_endpoint}")

    # Token usage. Authoritative batch cost comes from `dw batches analytics`
    # (file-first workflow). Batch runs on Doubleword's high-throughput backend.
    overall = sum(s.overall for s in scores.values()) / len(scores) if scores else 0.0
    msg.text(f"\nMean quality (overall): {overall:.3f} over {len(scores)} answers")
    msg.text(
        f"Tokens — generation: {gen_in:,} in / {gen_out:,} out · "
        f"judge: {judge_in:,} in / {judge_out:,} out"
    )
    msg.text("Cost: batch runs on Doubleword's high-throughput backend — see doubleword.ai/pricing.")
    print(
        "\nThat's it. You've just built a scalable, cost-managed, "
        "rate-limit-proof eval pipeline."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Doubleword batch eval → Arize Phoenix")
    p.add_argument("-n", "--limit", type=int, default=None, help="How many questions (default: all).")
    asyncio.run(main(p.parse_args().limit))
