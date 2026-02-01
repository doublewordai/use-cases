"""Batch API utilities for dataset compilation."""

import json
import os
import time
from pathlib import Path

import requests
from openai import OpenAI


def get_client() -> OpenAI:
    """Get Doubleword API client."""
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise ValueError("DOUBLEWORD_API_KEY environment variable not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.doubleword.ai/v1",
    )


def create_batch_file(requests_data: list[dict], output_path: Path) -> Path:
    """Write requests to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests_data:
            f.write(json.dumps(req) + "\n")
    return output_path


def upload_batch_file(client: OpenAI, file_path: Path) -> str:
    """Upload batch file and return file ID."""
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    return response.id


def create_batch(client: OpenAI, file_id: str, completion_window: str = "24h") -> str:
    """Create batch and return batch ID."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll until batch completes."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch
        time.sleep(poll_interval)


def download_results(client: OpenAI, file_id: str, output_path: Path) -> bool:
    """Download results file."""
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    response = requests.get(
        f"https://api.doubleword.ai/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code == 200:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    return False


def parse_results(results_path: Path) -> dict[str, dict]:
    """Parse results JSONL, keyed by custom_id."""
    results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                results[obj["custom_id"]] = obj
    return results


def count_tokens(results: dict[str, dict]) -> dict:
    """Sum tokens across all results."""
    input_tokens = 0
    output_tokens = 0
    for r in results.values():
        usage = r.get("response", {}).get("body", {}).get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def get_response_content(result: dict) -> str | None:
    """Extract response content from a batch result."""
    try:
        return result["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
