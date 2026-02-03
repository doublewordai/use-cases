"""Batch API utilities."""

import json
import os
import time
from pathlib import Path

import click
import requests
from openai import OpenAI


def get_client() -> OpenAI:
    """Create OpenAI client configured for Doubleword API."""
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise click.ClickException("DOUBLEWORD_API_KEY environment variable not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.doubleword.ai/v1",
    )


def create_batch_file(requests_data: list[dict], output_path: Path) -> Path:
    """Write requests to JSONL file for batch processing."""
    with open(output_path, "w") as f:
        for req in requests_data:
            f.write(json.dumps(req) + "\n")
    return output_path


def upload_batch_file(client: OpenAI, file_path: Path) -> str:
    """Upload a batch input file and return the file ID."""
    with open(file_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    return file.id


def create_batch(client: OpenAI, file_id: str, completion_window: str = "24h") -> str:
    """Create a batch job and return the batch ID."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch.id


def wait_for_batch(
    client: OpenAI, batch_id: str, poll_interval: int = 10, progress: bool = True
) -> dict:
    """Poll until batch completes or fails."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts

        if progress:
            click.echo(
                f"  Status: {status} | "
                f"Completed: {counts.completed}/{counts.total} | "
                f"Failed: {counts.failed}"
            )

        if status in ("completed", "failed", "cancelled", "expired"):
            return batch

        time.sleep(poll_interval)


def download_results(client: OpenAI, output_file_id: str, output_path: Path) -> bool:
    """Download batch results. Returns True if complete, False if still in progress."""
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    response = requests.get(
        f"https://api.doubleword.ai/v1/files/{output_file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
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


def count_tokens(results: dict[str, dict]) -> dict[str, int]:
    """Sum tokens across all results."""
    input_tokens = 0
    output_tokens = 0

    for r in results.values():
        usage = r.get("response", {}).get("body", {}).get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
