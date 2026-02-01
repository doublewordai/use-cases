"""
Batch API utilities for bug detection.

Supports both Doubleword (Qwen models) and OpenAI (GPT-5 models).
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import click
import requests
from openai import OpenAI


# Model aliases - consistent across all use cases
MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


def resolve_model(alias: str) -> str:
    """Resolve model alias to full name."""
    return MODELS.get(alias, alias)


def is_openai_model(model: str) -> bool:
    """Check if model is an OpenAI model (vs Doubleword)."""
    resolved = resolve_model(model)
    return resolved.startswith("gpt-")


def get_client(model: str = None) -> OpenAI:
    """Get OpenAI client configured for the appropriate provider."""
    if model and is_openai_model(model):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise click.ClickException("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("DOUBLEWORD_API_KEY")
        if not api_key:
            raise click.ClickException("DOUBLEWORD_API_KEY environment variable not set")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.doubleword.ai/v1"
        )


def get_doubleword_client() -> OpenAI:
    """Get OpenAI client configured for Doubleword."""
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise click.ClickException("DOUBLEWORD_API_KEY environment variable not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.doubleword.ai/v1"
    )


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for OpenAI."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise click.ClickException("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def create_batch_file(requests_list: list[dict], output_path: Path) -> Path:
    """Write requests to JSONL file."""
    with open(output_path, "w") as f:
        for i, req in enumerate(requests_list):
            line = {
                "custom_id": req.get("custom_id", f"req-{i:06d}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req["model"],
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0.0),
                    "response_format": {"type": "json_object"},
                }
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
        completion_window=completion_window
    )
    return batch.id


def wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 30,
    timeout: Optional[int] = None
) -> dict:
    """Poll until batch completes."""
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch

        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        click.echo(f"Status: {batch.status} ({completed}/{total})")

        if timeout and (time.time() - start) > timeout:
            raise click.ClickException(f"Batch timed out after {timeout}s")

        time.sleep(poll_interval)


def download_results(client: OpenAI, output_file_id: str, output_path: Path, provider: str = "doubleword") -> bool:
    """Download results file. Returns True if complete."""
    if provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = "https://api.openai.com/v1"
    else:
        api_key = os.environ["DOUBLEWORD_API_KEY"]
        base_url = "https://api.doubleword.ai/v1"

    response = requests.get(
        f"{base_url}/files/{output_file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    response.raise_for_status()

    is_incomplete = response.headers.get("X-Incomplete") == "true"

    with open(output_path, "wb") as f:
        f.write(response.content)

    return not is_incomplete


def parse_results(results_path: Path) -> dict[str, dict]:
    """Parse results JSONL, keyed by custom_id."""
    results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                results[obj["custom_id"]] = obj
    return results


def count_tokens(results: dict) -> dict:
    """Sum tokens across all results."""
    input_tokens = 0
    output_tokens = 0
    for r in results.values():
        usage = r.get("response", {}).get("body", {}).get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
