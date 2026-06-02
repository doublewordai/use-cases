"""Offline tests for JSONL emit + result parsing. No network."""

from __future__ import annotations

import json

from src import data


def test_generation_request_shape(sample_rows):
    req = data.build_generation_request(sample_rows[0])
    assert req["custom_id"] == "tqa-0000"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"
    # Model is intentionally NOT set — `dw files prepare` sets it.
    assert "model" not in req["body"]
    assert req["body"]["messages"][-1]["content"] == sample_rows[0].question


def test_judge_request_includes_reference_and_json_mode(sample_rows):
    row = sample_rows[0]
    req = data.build_judge_request(row.id, row.question, "Nothing happens.", row.reference())
    assert req["custom_id"] == row.id
    assert req["body"]["response_format"] == {"type": "json_object"}
    blob = json.dumps(req["body"]["messages"])
    assert "Nothing happens." in blob  # the answer
    assert "broken mirror" in blob  # a reference answer


def test_reference_puts_best_answer_first(sample_rows):
    refs = sample_rows[0].reference()
    assert refs[0] == "Nothing in particular happens."
    assert "Nothing happens." in refs


def test_write_then_read_jsonl(tmp_path, sample_rows):
    path = tmp_path / "batch.jsonl"
    n = data.write_jsonl(path, (data.build_generation_request(r) for r in sample_rows))
    assert n == 2
    rows = data.read_jsonl(path)
    assert [r["custom_id"] for r in rows] == ["tqa-0000", "tqa-0001"]


def test_ground_truth_roundtrip(tmp_path, sample_rows):
    path = tmp_path / "gt.json"
    data.save_ground_truth(path, sample_rows)
    loaded = data.load_ground_truth(path)
    assert set(loaded) == {"tqa-0000", "tqa-0001"}
    assert loaded["tqa-0001"].question == "What is the capital of France?"


def test_parse_content_dw_stream_shape(batch_result_factory):
    r = batch_result_factory("tqa-0000", "an answer")
    assert data.parse_content(r) == "an answer"


def test_parse_content_openai_batch_shape():
    r = {"custom_id": "x", "response": {"body": {"choices": [{"message": {"content": "hi"}}]}}}
    assert data.parse_content(r) == "hi"


def test_parse_content_missing_returns_none():
    assert data.parse_content({"custom_id": "x"}) is None


def test_parse_usage(batch_result_factory):
    r = batch_result_factory("tqa-0000", "a", prompt_tokens=12, completion_tokens=34)
    assert data.parse_usage(r) == (12, 34)


def test_index_by_custom_id(batch_result_factory):
    rs = [batch_result_factory("a", "x"), batch_result_factory("b", "y")]
    idx = data.index_by_custom_id(rs)
    assert set(idx) == {"a", "b"}
