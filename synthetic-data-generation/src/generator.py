"""Generation functions for the three-stage synthetic data pipeline."""

import csv
import json
import math

from .prompts import (
    CONVERSATION_SCHEMA,
    CONVERSATION_SYSTEM_PROMPT,
    DIFFICULTY_DISTRIBUTION,
    QUALITY_SCHEMA,
    QUALITY_SYSTEM_PROMPT,
    SCENARIO_SCHEMA,
    SCENARIO_SYSTEM_PROMPT,
    SUPPORT_TOPICS,
)


def load_seed_topics(seed_file: str) -> list[str]:
    """Load topic seeds from a CSV or JSONL file.

    CSV: expects a 'topic' column
    JSONL: expects objects with a 'topic' key
    """
    topics = []
    if seed_file.endswith(".jsonl"):
        with open(seed_file) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if "topic" in obj:
                        topics.append(obj["topic"])
    else:
        with open(seed_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "topic" in row:
                    topics.append(row["topic"])

    if not topics:
        raise ValueError(
            f"No topics found in {seed_file}. "
            "File must have a 'topic' column (CSV) or 'topic' key (JSONL)."
        )
    return topics


def build_scenario_requests(
    count: int,
    model: str,
    topics: list[str] = SUPPORT_TOPICS,
    difficulty_dist: dict[str, float] = DIFFICULTY_DISTRIBUTION,
    domain: str = "customer support",
    product: str = "SaaS platform",
) -> list[dict]:
    """Build batch requests for scenario generation.

    Distributes scenarios evenly across topics and difficulty levels.
    """
    system_prompt = SCENARIO_SYSTEM_PROMPT.format(domain=domain, product=product)

    requests_data = []
    idx = 0
    per_topic = math.ceil(count / len(topics))

    for topic in topics:
        for difficulty, proportion in difficulty_dist.items():
            n = max(1, round(per_topic * proportion))
            for _ in range(n):
                if idx >= count:
                    break
                requests_data.append(
                    {
                        "custom_id": f"scenario-{idx:06d}",
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"Generate a {domain} scenario.\n"
                                    f"Topic: {topic}\n"
                                    f"Difficulty: {difficulty}\n"
                                    f"Make it unique and realistic."
                                ),
                            },
                        ],
                        "response_format": SCENARIO_SCHEMA,
                        "temperature": 0.8,
                        "max_tokens": 512,
                    }
                )
                idx += 1
            if idx >= count:
                break
        if idx >= count:
            break

    return requests_data


def build_conversation_requests(
    scenarios: list[dict],
    model: str,
    domain: str = "customer support",
) -> list[dict]:
    """Build batch requests to generate conversations from parsed scenarios."""
    system_prompt = CONVERSATION_SYSTEM_PROMPT.format(domain=domain)

    requests_data = []
    for idx, scenario in enumerate(scenarios):
        scenario_text = json.dumps(scenario, indent=2)
        requests_data.append(
            {
                "custom_id": f"conv-{idx:06d}",
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a conversation for this scenario:\n\n"
                            f"{scenario_text}"
                        ),
                    },
                ],
                "response_format": CONVERSATION_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        )
    return requests_data


def build_quality_requests(
    conversations: list[dict],
    model: str,
) -> list[dict]:
    """Build batch requests to score conversation quality."""
    requests_data = []
    for idx, conversation in enumerate(conversations):
        conv_text = json.dumps(conversation, indent=2)
        requests_data.append(
            {
                "custom_id": f"quality-{idx:06d}",
                "model": model,
                "messages": [
                    {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Evaluate this customer support conversation:\n\n"
                            f"{conv_text}"
                        ),
                    },
                ],
                "response_format": QUALITY_SCHEMA,
                "temperature": 0,
                "max_tokens": 256,
            }
        )
    return requests_data


def format_for_training(
    conversations: list[dict],
    scores: list[dict],
    min_score: float = 3.5,
) -> list[dict]:
    """Filter by quality score and convert to OpenAI fine-tuning format.

    Each output entry has the structure:
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                   {"role": "assistant", ...}, ...]}
    """
    training_data = []

    for conv, score in zip(conversations, scores):
        overall = score.get("overall", 0)
        if overall < min_score:
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support agent. Be empathetic, "
                    "ask clarifying questions when needed, provide clear "
                    "step-by-step solutions, and confirm the issue is resolved."
                ),
            },
        ]

        conv_messages = conv.get("messages", [])
        for msg in conv_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "customer":
                messages.append({"role": "user", "content": content})
            elif role == "agent":
                messages.append({"role": "assistant", "content": content})

        if len(messages) > 1:
            training_data.append({"messages": messages})

    return training_data
