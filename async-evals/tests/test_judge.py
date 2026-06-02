"""Offline tests for the judge prompt + Score model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.judge import JUDGE_SYSTEM, Score, build_judge_messages


def test_score_overall():
    s = Score(relevance=1.0, hallucination_risk=0.0, tone=1.0)
    assert s.overall == pytest.approx(1.0)
    s2 = Score(relevance=0.0, hallucination_risk=1.0, tone=0.0)
    assert s2.overall == pytest.approx(0.0)


def test_score_bounds_enforced():
    with pytest.raises(ValidationError):
        Score(relevance=1.5, hallucination_risk=0.0, tone=0.0)


def test_score_parses_judge_json():
    s = Score.model_validate_json(
        '{"relevance": 0.8, "hallucination_risk": 0.2, "tone": 0.9, "rationale": "ok"}'
    )
    assert s.relevance == 0.8
    assert s.overall == pytest.approx((0.8 + 0.8 + 0.9) / 3)


def test_judge_messages_have_system_and_reference():
    msgs = build_judge_messages("Q?", "A.", reference=["correct one"])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == JUDGE_SYSTEM
    assert "correct one" in msgs[1]["content"]


def test_judge_messages_without_reference():
    msgs = build_judge_messages("Q?", "A.")
    assert "Reference correct answers" not in msgs[1]["content"]
