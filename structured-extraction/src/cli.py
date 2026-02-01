"""CLI for structured extraction with ensemble voting."""

import base64
import json
import os
from pathlib import Path

import click
from tqdm import tqdm

from .batch import (
    count_tokens,
    create_batch,
    create_batch_file,
    download_results,
    get_client,
    parse_results,
    upload_batch_file,
    wait_for_batch,
)
from .schema import EVAL_FIELDS, get_image_extraction_prompt
from .voting import calculate_field_accuracy, ensemble_vote, extract_json

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


def encode_image(image_path: str) -> str:
    """Encode image as base64 data URL."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{media_type};base64,{data}"


def build_message(record: dict) -> list[dict]:
    """Build message content for a record (image or text)."""
    prompt = get_image_extraction_prompt()

    if "image_path" in record:
        # Vision model with image
        image_url = encode_image(record["image_path"])
        return [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    elif "text" in record:
        # Text-only fallback
        from .schema import get_extraction_prompt
        return [
            {
                "type": "text",
                "text": get_extraction_prompt(record["text"]),
            },
        ]
    else:
        raise ValueError("Record must have 'image_path' or 'text' field")


@click.group()
def main():
    """Structured extraction with ensemble voting."""
    pass


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input JSONL file with receipts (image_path + ground_truth)")
@click.option("--output", "-o", "output_dir", default="results", type=click.Path(),
              help="Output directory for batch files and results")
@click.option("--models", "-m", default=DEFAULT_MODEL,
              help="Comma-separated model aliases (30b, 235b) or full names")
@click.option("--ensemble-sizes", "-n", default="1,5", help="Comma-separated ensemble sizes")
@click.option("--dry-run", is_flag=True, help="Prepare batch files but don't submit")
@click.option("--limit", "-l", type=int, help="Limit number of receipts to process")
def run(input_path: str, output_dir: str, models: str, ensemble_sizes: str, dry_run: bool, limit: int):
    """Run extraction experiment with multiple ensemble sizes and models."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(s.strip()) for s in ensemble_sizes.split(",")]
    model_list = []
    for m in models.split(","):
        m = m.strip()
        model_list.append(MODELS.get(m, m))  # Use alias or full name

    click.echo(f"Models: {model_list}")
    click.echo(f"Ensemble sizes: {sizes}")

    # Load input data
    receipts = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                receipts.append(json.loads(line))

    if limit:
        receipts = receipts[:limit]

    click.echo(f"Loaded {len(receipts)} receipts")

    # Check if we have images
    has_images = any("image_path" in r for r in receipts)
    if has_images:
        click.echo("Detected image-based input (vision mode)")
    else:
        click.echo("Detected text-based input")

    # Pre-build message content for all receipts (images are expensive to encode)
    click.echo("Encoding images...")
    receipt_contents = {}
    for receipt in tqdm(receipts, desc="Building messages"):
        receipt_id = receipt.get("id", str(len(receipt_contents)))
        try:
            receipt_contents[receipt_id] = build_message(receipt)
        except Exception as e:
            click.echo(f"Warning: Skipping {receipt_id}: {e}")

    # Generate batch requests for each model and ensemble size
    for model in model_list:
        model_short = next((k for k, v in MODELS.items() if v == model), model.split("/")[-1])

        for n in sizes:
            click.echo(f"\n=== Preparing {model_short} N={n} ===")
            requests_data = []

            for receipt_id, content in receipt_contents.items():
                # Create N copies of each request
                for i in range(n):
                    custom_id = f"{receipt_id}__run_{i}"
                    requests_data.append({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": content}],
                            "temperature": 0.7,  # Some variance for ensemble diversity
                        },
                    })

            # Write batch file
            batch_file = output_dir / f"batch_{model_short}_n{n}.jsonl"
            create_batch_file(requests_data, batch_file)
            click.echo(f"Created {batch_file} with {len(requests_data)} requests")

            if dry_run:
                click.echo("Dry run - skipping submission")
                continue

            # Submit batch
            client = get_client()
            click.echo("Uploading batch file...")
            file_id = upload_batch_file(client, batch_file)
            click.echo(f"File ID: {file_id}")

            click.echo("Creating batch...")
            batch_id = create_batch(client, file_id)
            click.echo(f"Batch ID: {batch_id}")

            # Save batch info
            batch_info = {"batch_id": batch_id, "file_id": file_id, "n": n, "model": model, "model_short": model_short}
            with open(output_dir / f"batch_{model_short}_n{n}_info.json", "w") as f:
                json.dump(batch_info, f, indent=2)

    if dry_run:
        click.echo("\nDry run complete. Review batch files and run without --dry-run to submit.")


@main.command()
@click.option("--output", "-o", "output_dir", default="results", type=click.Path(exists=True),
              help="Output directory with batch info files")
@click.option("--wait/--no-wait", default=True, help="Wait for batches to complete")
def status(output_dir: str, wait: bool):
    """Check batch status and download results when complete."""
    output_dir = Path(output_dir)
    client = get_client()

    # Find all batch info files (supports both old and new naming)
    info_files = sorted(output_dir.glob("batch_*_info.json"))

    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)

        batch_id = info["batch_id"]
        n = info["n"]
        model_short = info.get("model_short", "default")
        click.echo(f"\n=== Batch {model_short} N={n} ({batch_id}) ===")

        batch = client.batches.retrieve(batch_id)
        click.echo(f"Status: {batch.status}")
        click.echo(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}")

        if batch.status == "in_progress" and wait:
            click.echo("Waiting for completion...")
            batch = wait_for_batch(client, batch_id)

        if batch.status == "completed" and batch.output_file_id:
            results_file = output_dir / f"results_{model_short}_n{n}.jsonl"
            click.echo(f"Downloading results to {results_file}...")
            download_results(client, batch.output_file_id, results_file)
            click.echo("Done!")


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input JSONL file with receipts")
@click.option("--output", "-o", "output_dir", default="results", type=click.Path(),
              help="Output directory for results")
@click.option("--model", "-m", default="gpt-5-mini", help="Model to use")
@click.option("--concurrency", "-c", default=20, help="Number of concurrent requests")
@click.option("--limit", "-l", type=int, help="Limit number of receipts to process")
def realtime(input_path: str, output_dir: str, model: str, concurrency: int, limit: int):
    """Run extraction via real-time API with async concurrency."""
    import asyncio
    from .batch import run_realtime_extraction

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    receipts = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                receipts.append(json.loads(line))

    if limit:
        receipts = receipts[:limit]

    click.echo(f"Loaded {len(receipts)} receipts")
    click.echo(f"Model: {model}")
    click.echo(f"Concurrency: {concurrency}")

    # Build requests
    click.echo("Encoding images...")
    requests_data = []
    for receipt in tqdm(receipts, desc="Building messages"):
        receipt_id = receipt.get("id", str(len(requests_data)))
        try:
            content = build_message(receipt)
            requests_data.append({
                "custom_id": f"{receipt_id}__run_0",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                },
            })
        except Exception as e:
            click.echo(f"Warning: Skipping {receipt_id}: {e}")

    click.echo(f"\nRunning {len(requests_data)} requests...")

    # Progress bar
    pbar = tqdm(total=len(requests_data), desc="Extracting")

    def progress_callback(completed, total):
        pbar.n = completed
        pbar.refresh()

    # Run async extraction
    results = asyncio.run(
        run_realtime_extraction(requests_data, model, concurrency, progress_callback)
    )
    pbar.close()

    # Save results in same format as batch results
    model_short = model.replace("-", "").replace(".", "")
    results_file = output_dir / f"results_{model_short}_n1.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Count errors
    errors = sum(1 for r in results if "error" in r)
    click.echo(f"\nCompleted: {len(results) - errors}/{len(results)}")
    if errors:
        click.echo(f"Errors: {errors}")
    click.echo(f"Results saved to {results_file}")


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Original input JSONL with ground truth")
@click.option("--results-dir", "-r", "results_dir", default="results", type=click.Path(exists=True),
              help="Directory containing result files")
@click.option("--output", "-o", "output_file", default="results/analysis.json",
              help="Output file for analysis results")
def analyze(input_path: str, results_dir: str, output_file: str):
    """Analyze results and compare ensemble sizes across models."""
    input_path = Path(input_path)
    results_dir = Path(results_dir)
    output_file = Path(output_file)

    # Load ground truth
    ground_truth = {}
    with open(input_path) as f:
        for line in f:
            if line.strip():
                receipt = json.loads(line)
                receipt_id = receipt.get("id", receipt.get("file_id"))
                ground_truth[receipt_id] = receipt.get("ground_truth", receipt.get("labels", {}))

    click.echo(f"Loaded {len(ground_truth)} ground truth records")

    # Find result files (supports both old results_n*.jsonl and new results_*_n*.jsonl)
    result_files = sorted(results_dir.glob("results_*.jsonl"))
    if not result_files:
        raise click.ClickException("No result files found. Run 'status' first to download results.")

    # Group by model
    analysis = {"by_model": {}, "token_usage": {}}

    for result_file in result_files:
        # Extract model and N from filename (results_30b_n5.jsonl or results_n5.jsonl)
        stem = result_file.stem
        if "_n" in stem:
            parts = stem.replace("results_", "").rsplit("_n", 1)
            if len(parts) == 2:
                model_short, n_str = parts
                n = int(n_str)
            else:
                model_short = "default"
                n = int(parts[0].replace("n", ""))
        else:
            model_short = "default"
            n = 1

        click.echo(f"\n=== Analyzing {model_short} N={n} ===")

        results = parse_results(result_file)
        click.echo(f"Loaded {len(results)} results")

        # Group results by receipt
        by_receipt = {}
        for custom_id, result in results.items():
            receipt_id = custom_id.rsplit("__run_", 1)[0]
            if receipt_id not in by_receipt:
                by_receipt[receipt_id] = []

            # Extract the model's response
            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                extraction = extract_json(content)
                if extraction:
                    by_receipt[receipt_id].append(extraction)
            except (KeyError, IndexError, TypeError) as e:
                click.echo(f"Warning: Failed to parse result for {custom_id}: {e}")

        # Calculate accuracy
        field_correct = {field: 0 for field in EVAL_FIELDS}
        field_total = {field: 0 for field in EVAL_FIELDS}
        agreement_rates = {field: [] for field in EVAL_FIELDS}
        extraction_rates = {field: [] for field in EVAL_FIELDS}

        for receipt_id, extractions in tqdm(by_receipt.items(), desc="Evaluating"):
            if receipt_id not in ground_truth:
                continue

            gt = ground_truth[receipt_id]

            # Perform ensemble voting
            if len(extractions) > 1:
                consensus, field_stats = ensemble_vote(extractions)
                for field, stats in field_stats.items():
                    agreement_rates[field].append(stats["agreement_rate"])
                    extraction_rates[field].append(stats["extraction_rate"])
            elif len(extractions) == 1:
                consensus = extractions[0]
                for field in EVAL_FIELDS:
                    agreement_rates[field].append(1.0)
                    extraction_rates[field].append(1.0 if consensus.get(field) is not None else 0.0)
            else:
                continue

            # Compare to ground truth
            accuracy = calculate_field_accuracy(consensus, gt)
            for field, is_correct in accuracy.items():
                field_total[field] += 1
                if is_correct:
                    field_correct[field] += 1

        # Calculate percentages
        field_acc = {}
        for field in EVAL_FIELDS:
            if field_total[field] > 0:
                field_acc[field] = {
                    "accuracy": field_correct[field] / field_total[field],
                    "correct": field_correct[field],
                    "total": field_total[field],
                    "avg_agreement": sum(agreement_rates[field]) / len(agreement_rates[field]) if agreement_rates[field] else 0,
                    "avg_extraction_rate": sum(extraction_rates[field]) / len(extraction_rates[field]) if extraction_rates[field] else 0,
                }

        # Overall accuracy
        total_correct = sum(field_correct.values())
        total_fields = sum(field_total.values())
        overall_acc = total_correct / total_fields if total_fields > 0 else 0

        # Store by model
        if model_short not in analysis["by_model"]:
            analysis["by_model"][model_short] = {"ensemble_results": {}, "field_accuracy": {}}

        analysis["by_model"][model_short]["field_accuracy"][n] = field_acc
        analysis["by_model"][model_short]["ensemble_results"][n] = {
            "overall_accuracy": overall_acc,
            "receipts_evaluated": len(by_receipt),
            "total_extractions": len(results),
        }

        # Token usage
        key = f"{model_short}_n{n}"
        analysis["token_usage"][key] = count_tokens(results)

        click.echo(f"Overall accuracy: {overall_acc:.1%}")
        for field, acc in field_acc.items():
            click.echo(f"  {field}: {acc['accuracy']:.1%} (agreement: {acc['avg_agreement']:.1%}, extracted: {acc['avg_extraction_rate']:.1%})")

    # Save analysis
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    click.echo(f"\nAnalysis saved to {output_file}")

    # Print summary comparison
    click.echo("\n=== Summary ===")
    for model_short, model_data in analysis["by_model"].items():
        click.echo(f"\n{model_short.upper()}:")
        click.echo(f"{'N':>4} | {'Overall':>8} | " + " | ".join(f"{f[:8]:>8}" for f in EVAL_FIELDS))
        click.echo("-" * 60)
        for n in sorted(model_data["ensemble_results"].keys()):
            overall = model_data["ensemble_results"][n]["overall_accuracy"]
            fields = [model_data["field_accuracy"][n].get(f, {}).get("accuracy", 0) for f in EVAL_FIELDS]
            click.echo(f"{n:>4} | {overall:>7.1%} | " + " | ".join(f"{a:>7.1%}" for a in fields))


if __name__ == "__main__":
    main()
