# Use Case Implementation Guide

This folder contains batch API use cases demonstrating "more is different" capabilities. Each subfolder is a self-contained project.

## Your Task

1. Read `INSTRUCTIONS.md` in your use case folder for the brief
2. Read `RUBRIC.md` for evaluation criteria
3. Implement the CLI in `src/`
4. Run against real data
5. Generate `report.md` with results

## Doubleword Batch API

### Authentication

```bash
export DOUBLEWORD_API_KEY="your-api-key"
```

Base URL: `https://api.doubleword.ai/v1`

### Batch Flow

1. **Prepare** a `.jsonl` file (one request per line)
2. **Upload** the file
3. **Create** a batch
4. **Poll** for completion (or wait)
5. **Download** results

### JSONL Format

Each line is a JSON object:

```json
{"custom_id": "req-001", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "messages": [{"role": "user", "content": "Hello"}]}}
```

Required fields:
- `custom_id`: Your tracking ID (use it to match responses)
- `method`: Always `"POST"`
- `url`: Always `"/v1/chat/completions"`
- `body`: The chat completion request

### Python Client

Use the OpenAI client with Doubleword's base URL:

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DOUBLEWORD_API_KEY"],
    base_url="https://api.doubleword.ai/v1"
)
```

### Upload File

```python
with open("batch_input.jsonl", "rb") as f:
    file = client.files.create(file=f, purpose="batch")
print(f"File ID: {file.id}")
```

### Create Batch

```python
batch = client.batches.create(
    input_file_id=file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"  # or "1h" for faster
)
print(f"Batch ID: {batch.id}")
```

### Check Status

```python
status = client.batches.retrieve(batch.id)
print(f"Status: {status.status}")
print(f"Progress: {status.request_counts.completed}/{status.request_counts.total}")
```

### Download Results

```python
import requests

response = requests.get(
    f"https://api.doubleword.ai/v1/files/{batch.output_file_id}/content",
    headers={"Authorization": f"Bearer {os.environ['DOUBLEWORD_API_KEY']}"}
)

# Check if still in progress
is_incomplete = response.headers.get("X-Incomplete") == "true"

with open("results.jsonl", "wb") as f:
    f.write(response.content)
```

Results are available as they complete (don't need to wait for full batch).

### Available Models

- `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` - Mid-size, GPT-4.1-mini tier
- `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` - Flagship, GPT-4/Claude Opus tier

## Code Conventions

### Structure

```
use-case-name/
├── INSTRUCTIONS.md
├── src/
│   ├── __init__.py
│   ├── cli.py          # Click CLI entrypoint
│   ├── batch.py        # Batch API utilities
│   └── ...
├── data/               # Sample input data
├── results/            # Output artifacts
└── report.md           # Final report
```

### CLI Pattern

Use Click. Provide sensible defaults. Example:

```python
import click

@click.command()
@click.option("--input", "-i", required=True, help="Input file or directory")
@click.option("--output", "-o", default="results/", help="Output directory")
@click.option("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
@click.option("--dry-run", is_flag=True, help="Prepare batch but don't submit")
def main(input, output, model, dry_run):
    """One-line description of what this does."""
    pass

if __name__ == "__main__":
    main()
```

### Dependencies

Use `uv` for dependency management:

```bash
cd use-case-name
uv init
uv add openai click requests
```

### Async Batch Utilities

Common pattern for batch operations:

```python
import json
import time
from pathlib import Path

def create_batch_file(requests: list[dict], output_path: Path) -> Path:
    """Write requests to JSONL file."""
    with open(output_path, "w") as f:
        for i, req in enumerate(requests):
            line = {
                "custom_id": req.get("custom_id", f"req-{i:06d}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req["model"],
                    "messages": req["messages"],
                }
            }
            f.write(json.dumps(line) + "\n")
    return output_path

def wait_for_batch(client, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll until batch completes."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "cancelled"):
            return batch
        click.echo(f"Status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total})")
        time.sleep(poll_interval)

def parse_results(results_path: Path) -> dict[str, dict]:
    """Parse results JSONL, keyed by custom_id."""
    results = {}
    with open(results_path) as f:
        for line in f:
            obj = json.loads(line)
            results[obj["custom_id"]] = obj
    return results
```

## Cost Calculation

Include a cost comparison in every report.

### Pricing Reference (as of Jan 2025)

| Provider | Model | Input (per 1M) | Output (per 1M) | Batch Discount |
|----------|-------|----------------|-----------------|----------------|
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | 50% |
| OpenAI | GPT-4o | $2.50 | $10.00 | 50% |
| Doubleword | Qwen3-30B | TBD | TBD | - |
| Doubleword | Qwen3-235B | TBD | TBD | - |

### Tracking Usage

Log token counts from responses:

```python
def count_tokens(results: dict) -> dict:
    """Sum tokens across all results."""
    input_tokens = 0
    output_tokens = 0
    for r in results.values():
        if r.get("response", {}).get("body", {}).get("usage"):
            usage = r["response"]["body"]["usage"]
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
```

### Report Format

```markdown
## Cost Comparison

| Metric | Value |
|--------|-------|
| Total requests | 10,000 |
| Input tokens | 2,450,000 |
| Output tokens | 890,000 |

| Provider | Estimated Cost |
|----------|----------------|
| OpenAI GPT-4o-mini (real-time) | $X.XX |
| OpenAI GPT-4o-mini (batch) | $X.XX |
| Doubleword Qwen3-30B | $X.XX |
```

## Report Requirements

The `report.md` must include:

1. **What we did** - One paragraph summary
2. **Data** - What input, how much, source
3. **Baseline** - What we compared against
4. **Results** - Key findings with numbers, charts if helpful
5. **Cost comparison** - Table showing provider costs
6. **Conclusion** - One paragraph takeaway

Keep it concise. The report should be readable in 2 minutes.
