"""Batch API utilities for async research agents with tool calling."""

import json
import os
import time
from pathlib import Path

import click
import requests
from openai import OpenAI

PROVIDERS = {
    "doubleword": {
        "base_url": "https://api.doubleword.ai/v1",
        "env_var": "DOUBLEWORD_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
    },
}


def get_client(provider: str = "doubleword") -> tuple[OpenAI, str]:
    """Get OpenAI-compatible client for specified provider."""
    if provider not in PROVIDERS:
        raise click.ClickException(
            f"Unknown provider: {provider}. Valid options: {list(PROVIDERS.keys())}"
        )
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_var"])
    if not api_key:
        raise click.ClickException(f"{config['env_var']} environment variable not set")
    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    return client, provider


def create_batch_file(requests_data: list[dict], output_path: Path) -> Path:
    """Write requests to JSONL file for batch processing.

    Supports the tools field for function calling. Each request dict can contain:
    - custom_id, model, messages (required)
    - tools, tool_choice (optional, for function calling)
    - temperature, max_tokens (optional)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests_data:
            body = {
                "model": req["model"],
                "messages": req["messages"],
                "temperature": req.get("temperature", 0),
                "max_tokens": req.get("max_tokens", 4096),
            }
            if "tools" in req:
                body["tools"] = req["tools"]
            if "tool_choice" in req:
                body["tool_choice"] = req["tool_choice"]
            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line) + "\n")
    return output_path


def upload_batch_file(client: OpenAI, file_path: Path) -> str:
    """Upload batch file and return file ID."""
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    return file.id


def create_batch(client: OpenAI, file_id: str, completion_window: str = "24h") -> str:
    """Create batch and return batch ID."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch.id


def wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 30,
) -> dict:
    """Poll until batch completes, returning final status."""
    while True:
        batch = client.batches.retrieve(batch_id)

        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch

        time.sleep(poll_interval)


def download_results(
    output_file_id: str, output_path: Path, provider: str = "doubleword"
) -> Path:
    """Download batch results to file."""
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_var"])
    response = requests.get(
        f"{config['base_url']}/files/{output_file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def parse_results(results_path: Path) -> dict[str, dict]:
    """Parse results JSONL, keyed by custom_id."""
    results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                results[obj["custom_id"]] = obj
    return results


def extract_message(result: dict) -> dict:
    """Extract the assistant message from a batch result.

    Returns the full message dict which may contain 'content' and/or 'tool_calls'.
    """
    try:
        return result["response"]["body"]["choices"][0]["message"]
    except (KeyError, IndexError):
        return {"role": "assistant", "content": ""}


def extract_content(result: dict) -> str:
    """Extract the text content from a batch result."""
    msg = extract_message(result)
    return msg.get("content") or ""


def get_finish_reason(result: dict) -> str:
    """Extract the finish reason from a batch result."""
    try:
        return result["response"]["body"]["choices"][0]["finish_reason"]
    except (KeyError, IndexError):
        return "unknown"


def count_tokens(results: dict) -> dict:
    """Sum tokens across all results."""
    input_tokens = 0
    output_tokens = 0
    for r in results.values():
        usage = r.get("response", {}).get("body", {}).get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
