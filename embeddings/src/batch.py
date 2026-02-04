"""Batch API utilities for embedding generation."""

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


def create_batch_file(texts: list[str], model: str, output_path: Path) -> Path:
    """Write embedding requests to JSONL file for batch processing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, text in enumerate(texts):
            line = {
                "custom_id": f"emb-{i:06d}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": text,
                },
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
        endpoint="/v1/embeddings",
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
        counts = batch.request_counts

        click.echo(
            f"  Status: {batch.status} ({counts.completed}/{counts.total} completed)"
        )

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


def extract_embedding(result: dict) -> list[float]:
    """Extract embedding vector from a batch result."""
    return result["response"]["body"]["data"][0]["embedding"]


def count_tokens(results: dict) -> dict:
    """Sum tokens across all results (embeddings only have input tokens)."""
    input_tokens = 0
    for r in results.values():
        usage = r.get("response", {}).get("body", {}).get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0) + usage.get("total_tokens", 0)
    return {"input_tokens": input_tokens, "output_tokens": 0}
