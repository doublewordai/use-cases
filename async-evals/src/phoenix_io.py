"""The Doubleword-batch ↔ Arize-Phoenix bridge.

The expensive work (generation + judging) happens out-of-process on Doubleword's
batch tier. By the time we get here, every answer and every score is already
computed. We surface them in Phoenix as a first-class **Dataset + Experiment**:

- the eval set becomes a Phoenix Dataset (question + reference answers),
- the run becomes an Experiment whose `task` returns the (already generated)
  answer and whose `evaluators` return the (already computed) judge scores.

Because task/evaluators are pure lookups, `run_experiment` makes no LLM calls —
it just records the batch's results against the dataset so you get per-example
outputs, scores, and run-to-run comparison in the Phoenix UI.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .config import settings
from .judge import Score


def get_client():
    from phoenix.client import Client

    # api_key is None for local Phoenix and set for Arize Phoenix Cloud.
    return Client(
        base_url=settings.phoenix_collector_endpoint,
        api_key=settings.phoenix_api_key,
    )


def _example_input(example: Any) -> dict[str, Any]:
    """DatasetExample is dict-like in the Phoenix client, but be defensive."""
    if isinstance(example, dict):
        return example.get("input", {}) or {}
    return getattr(example, "input", {}) or {}


def create_eval_dataset(rows: list, name: str) -> Any:
    """Create (or version) a Phoenix Dataset from TruthfulQA rows.

    `rows` are src.data.Row. custom_id is carried in the *input* so the
    experiment task/evaluators can look results up deterministically.
    """
    df = pd.DataFrame(
        [
            {
                "custom_id": r.id,
                "question": r.question,
                "best_answer": r.best_answer,
                "reference": r.reference(),
            }
            for r in rows
        ]
    )
    client = get_client()
    return client.datasets.create_dataset(
        name=name,
        dataframe=df,
        input_keys=["question", "custom_id"],
        output_keys=["best_answer"],
        metadata_keys=["reference"],
        dataset_description="TruthfulQA generation split — judged on Doubleword batch.",
    )


def _make_evaluator(name: str, scores_by_id: dict[str, Score], pick: Callable[[Score], float]):
    def _evaluator(input: dict[str, Any]) -> float:  # noqa: A002 - Phoenix binds by name
        cid = (input or {}).get("custom_id")
        score = scores_by_id.get(cid)
        return float(pick(score)) if score is not None else 0.0

    # Phoenix keys evaluators by name AND qualname — give each a distinct identity
    # so all four scores are recorded (not deduped to one).
    _evaluator.__name__ = name
    _evaluator.__qualname__ = f"evaluator_{name}"
    return _evaluator


def log_judge_experiment(
    dataset_name: str,
    answers_by_id: dict[str, str],
    scores_by_id: dict[str, Score],
    experiment_name: str,
) -> Any:
    """Record batch results as a Phoenix Experiment over the eval dataset."""
    client = get_client()
    dataset = client.datasets.get_dataset(dataset=dataset_name)

    def task(example: Any) -> str:
        cid = _example_input(example).get("custom_id")
        return answers_by_id.get(cid, "")

    evaluators = [
        _make_evaluator("relevance", scores_by_id, lambda s: s.relevance),
        _make_evaluator("truthfulness", scores_by_id, lambda s: 1.0 - s.hallucination_risk),
        _make_evaluator("tone", scores_by_id, lambda s: s.tone),
        _make_evaluator("overall", scores_by_id, lambda s: s.overall),
    ]

    return client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description="LLM-as-judge scores computed on Doubleword's batch tier.",
        print_summary=True,
    )
