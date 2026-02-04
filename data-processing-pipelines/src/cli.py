"""CLI for running data processing pipelines via batch API."""

import json
import math
from collections import Counter
from pathlib import Path

import click

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
from .pipeline import (
    build_dedup_requests,
    build_enrich_requests,
    build_normalize_requests,
    download_sec_data,
    generate_dedup_candidates,
    load_csv_data,
    parse_json_response,
)

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


def _extract_content(result: dict) -> str:
    """Extract message content from a batch result."""
    try:
        return result["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def _run_stage(
    client,
    provider: str,
    requests_data: list[dict],
    stage_name: str,
    output_dir: Path,
    model_slug: str,
) -> dict[str, dict]:
    """Run a complete batch stage: create file, upload, wait, download, parse."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Stage: {stage_name}")
    click.echo(f"{'=' * 60}")
    click.echo(f"Requests: {len(requests_data)}")

    batch_file = output_dir / f"{stage_name}_{model_slug}_input.jsonl"
    create_batch_file(requests_data, batch_file)

    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)
    click.echo(f"  File ID: {file_id}")

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"  Batch ID: {batch_id}")

    click.echo("Waiting for batch to complete...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed":
        raise click.ClickException(f"Batch failed with status: {batch.status}")

    results_file = output_dir / f"{stage_name}_{model_slug}_output.jsonl"
    download_results(batch.output_file_id, results_file, provider)
    click.echo(f"Results saved: {results_file}")

    return parse_results(results_file)


BATCH_CHUNK_SIZE = 10_000


def _run_stage_chunked(
    client,
    provider: str,
    requests_data: list[dict],
    stage_name: str,
    output_dir: Path,
    model_slug: str,
) -> dict[str, dict]:
    """Run a stage in chunks of BATCH_CHUNK_SIZE, merging all results."""
    total = len(requests_data)
    num_chunks = math.ceil(total / BATCH_CHUNK_SIZE)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Stage: {stage_name} ({total} requests in {num_chunks} batch(es))")
    click.echo(f"{'=' * 60}")

    all_results = {}
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BATCH_CHUNK_SIZE
        end = min(start + BATCH_CHUNK_SIZE, total)
        chunk = requests_data[start:end]

        chunk_label = f"{stage_name}_chunk{chunk_idx}"
        click.echo(f"\n  Batch {chunk_idx + 1}/{num_chunks} ({len(chunk)} requests)")

        batch_file = output_dir / f"{chunk_label}_{model_slug}_input.jsonl"
        create_batch_file(chunk, batch_file)

        click.echo("  Uploading batch file...")
        file_id = upload_batch_file(client, batch_file)
        click.echo(f"    File ID: {file_id}")

        click.echo("  Creating batch...")
        batch_id = create_batch(client, file_id)
        click.echo(f"    Batch ID: {batch_id}")

        click.echo("  Waiting for batch to complete...")
        batch = wait_for_batch(client, batch_id)

        if batch.status != "completed":
            raise click.ClickException(
                f"Batch {chunk_idx + 1}/{num_chunks} failed with status: {batch.status}"
            )

        results_file = output_dir / f"{chunk_label}_{model_slug}_output.jsonl"
        download_results(batch.output_file_id, results_file, provider)
        click.echo(f"  Results saved: {results_file}")

        all_results.update(parse_results(results_file))

    return all_results


@click.group()
def cli():
    """Data processing pipeline using the Doubleword Batch API.

    Clean, enrich, and deduplicate company records at scale. Uses real SEC
    EDGAR data by default, or bring your own CSV.
    """
    pass


@cli.command()
@click.option(
    "--limit", default=500, help="Number of records to download (default: 500)"
)
@click.option(
    "--input",
    "input_csv",
    default=None,
    help="Path to your own CSV file (must have a 'name' column)",
)
@click.option("-o", "--output", default="data/records.json", help="Output file path")
def prepare(limit: int, input_csv: str, output: str):
    """Download SEC EDGAR company data, or load your own CSV.

    By default, downloads real company records from the SEC EDGAR public
    dataset (no API key required). Use --input to provide your own CSV.
    """
    if input_csv:
        click.echo(f"Loading records from {input_csv}...")
        records = load_csv_data(input_csv, limit=limit)
        click.echo(f"Loaded {len(records)} records from CSV")
    else:
        click.echo(f"Downloading up to {limit} records from SEC EDGAR...")
        records = download_sec_data(limit=limit)
        click.echo(f"Downloaded {len(records)} company records")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    click.echo(f"Saved to {output_path}")

    # Show sample
    click.echo("\nSample records:")
    for r in records[:5]:
        name = r.get("name", "N/A")
        ticker = r.get("ticker", "")
        exchange = r.get("exchange", "")
        extra = f" ({ticker}, {exchange})" if ticker else ""
        click.echo(f"  {r['id']}: {name}{extra}")


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    default="data/records.json",
    help="Input records file",
)
@click.option("-o", "--output", default="results/", help="Output directory")
@click.option(
    "-m",
    "--model",
    default=DEFAULT_MODEL,
    help="Model alias (30b, 235b, gpt5-nano, gpt5-mini, gpt5.2) or full name",
)
@click.option(
    "-p",
    "--provider",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
@click.option("--dry-run", is_flag=True, help="Create batch files but don't submit")
def run(input_path: str, output: str, model: str, provider: str, dry_run: bool):
    """Run the full data processing pipeline (normalize -> enrich -> deduplicate)."""
    model_id = MODELS.get(model, model)
    model_slug = model_id.replace("/", "_")

    input_file = Path(input_path)
    if not input_file.exists():
        raise click.ClickException(
            f"Input file not found: {input_file}\nRun 'prepare' command first."
        )

    with open(input_file) as f:
        records = json.load(f)

    click.echo(f"Loaded {len(records)} records from {input_file}")
    click.echo(f"Model: {model_id}")
    click.echo(f"Provider: {provider}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    # Stage 1: Normalize
    norm_requests = build_normalize_requests(records, model_id)
    norm_file = output_dir / f"normalize_{model_slug}_input.jsonl"
    create_batch_file(norm_requests, norm_file)

    if dry_run:
        click.echo(f"\nDry run - created {len(norm_requests)} normalize requests")
        click.echo("(Dedup and enrich batch files are generated at runtime.)")
        click.echo("Batch files created but not submitted.")
        return

    client, _ = get_client(provider)

    norm_results = _run_stage(
        client,
        provider,
        norm_requests,
        "normalize",
        output_dir,
        model_slug,
    )
    tokens = count_tokens(norm_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    # Apply normalization results
    for record in records:
        key = f"normalize-{record['id']}"
        if key in norm_results:
            content = _extract_content(norm_results[key])
            if content:
                norm_data = parse_json_response(content)
                if norm_data:
                    record["normalized_name"] = norm_data.get(
                        "normalized_name", record["name"]
                    )
                    record["street"] = norm_data.get("street", "")
                    record["city"] = norm_data.get("city", "")
                    record["state"] = norm_data.get("state", "")
                    record["zip_code"] = norm_data.get("zip_code", "")
                    record["country"] = norm_data.get("country", "")

    norm_output = output_dir / "stage1_normalized.json"
    with open(norm_output, "w") as f:
        json.dump(records, f, indent=2)
    click.echo(f"Normalized records saved to {norm_output}")

    # Stage 2: Deduplicate (before enrichment to avoid enriching duplicates)
    click.echo(f"\n{'=' * 60}")
    click.echo("Stage: Deduplicate")
    click.echo(f"{'=' * 60}")

    click.echo(
        "Finding candidate duplicate pairs (token blocking + fuzzy/Jaccard scoring)..."
    )
    auto_dupes, candidates = generate_dedup_candidates(records)
    click.echo(
        f"Found {len(auto_dupes)} auto-confirmed duplicates (score >= 95), "
        f"{len(candidates)} candidates for LLM verification"
    )

    # Auto-confirmed duplicates (high fuzzy score, no LLM needed)
    duplicates = []
    duplicate_ids = set()
    for rec_a, rec_b, score in auto_dupes:
        duplicates.append(
            {
                "record_a": rec_a["id"],
                "record_b": rec_b["id"],
                "name_a": rec_a.get("normalized_name", rec_a["name"]),
                "name_b": rec_b.get("normalized_name", rec_b["name"]),
                "fuzzy_score": score,
                "confidence": "auto",
                "relationship": "exact_or_near_match",
            }
        )
        duplicate_ids.add(rec_b["id"])

    # LLM-verified duplicates (ambiguous fuzzy score, submitted in chunks)
    if candidates:
        dedup_requests = build_dedup_requests(candidates, model_id)
        dedup_results = _run_stage_chunked(
            client,
            provider,
            dedup_requests,
            "dedup",
            output_dir,
            model_slug,
        )
        tokens = count_tokens(dedup_results)
        total_tokens["input_tokens"] += tokens["input_tokens"]
        total_tokens["output_tokens"] += tokens["output_tokens"]

        for idx, (rec_a, rec_b, score) in enumerate(candidates):
            key = f"dedup-{idx:06d}"
            if key in dedup_results:
                content = _extract_content(dedup_results[key])
                if content:
                    dedup_data = parse_json_response(content)
                    if dedup_data and dedup_data.get("is_duplicate"):
                        duplicates.append(
                            {
                                "record_a": rec_a["id"],
                                "record_b": rec_b["id"],
                                "name_a": rec_a.get("normalized_name", rec_a["name"]),
                                "name_b": rec_b.get("normalized_name", rec_b["name"]),
                                "fuzzy_score": score,
                                "confidence": dedup_data.get("confidence", ""),
                                "relationship": dedup_data.get("relationship", ""),
                            }
                        )
                        duplicate_ids.add(rec_b["id"])

    dedup_output = output_dir / "stage2_duplicates.json"
    with open(dedup_output, "w") as f:
        json.dump(duplicates, f, indent=2)
    click.echo(f"Found {len(duplicates)} confirmed duplicates -> {dedup_output}")

    # Remove duplicates before enrichment
    unique_records = [r for r in records if r["id"] not in duplicate_ids]
    click.echo(f"Deduplicated {len(records)} -> {len(unique_records)} unique records")

    # Stage 3: Enrich (only unique records)
    enrich_requests = build_enrich_requests(unique_records, model_id)
    enrich_results = _run_stage(
        client,
        provider,
        enrich_requests,
        "enrich",
        output_dir,
        model_slug,
    )
    tokens = count_tokens(enrich_results)
    total_tokens["input_tokens"] += tokens["input_tokens"]
    total_tokens["output_tokens"] += tokens["output_tokens"]

    for record in unique_records:
        key = f"enrich-{record['id']}"
        if key in enrich_results:
            content = _extract_content(enrich_results[key])
            if content:
                enrich_data = parse_json_response(content)
                if enrich_data:
                    record["industry"] = enrich_data.get("industry", "")
                    record["sub_industry"] = enrich_data.get("sub_industry", "")
                    record["size"] = enrich_data.get("size", "")
                    record["industry_confidence"] = enrich_data.get("confidence", "")

    enrich_output = output_dir / "stage3_enriched.json"
    with open(enrich_output, "w") as f:
        json.dump(unique_records, f, indent=2)
    click.echo(f"Enriched records saved to {enrich_output}")

    # Summary
    summary = {
        "model": model_id,
        "provider": provider,
        "num_records": len(records),
        "unique_records": len(unique_records),
        "tokens": total_tokens,
        "duplicates_found": len(duplicates),
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    click.echo(f"\nPipeline complete. Tokens: {total_tokens}")


@cli.command()
@click.option("--batch-id", required=True, help="Batch ID to check")
@click.option(
    "-p",
    "--provider",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
def status(batch_id: str, provider: str):
    """Check the status of a batch job."""
    client, _ = get_client(provider)
    batch = client.batches.retrieve(batch_id)

    click.echo(f"Batch ID: {batch.id}")
    click.echo(f"Status: {batch.status}")
    click.echo(
        f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
    )


@cli.command()
@click.option("-o", "--output", default="results/", help="Results directory to analyze")
def analyze(output: str):
    """Analyze pipeline results and print summary statistics."""
    output_dir = Path(output)

    enriched_path = output_dir / "stage3_enriched.json"
    if enriched_path.exists():
        with open(enriched_path) as f:
            records = json.load(f)

        click.echo(f"{'=' * 60}")
        click.echo("Normalization Summary")
        click.echo(f"{'=' * 60}")
        normalized_count = sum(
            1
            for r in records
            if r.get("normalized_name") and r["normalized_name"] != r["name"]
        )
        click.echo(f"Total records: {len(records)}")
        click.echo(f"Names corrected: {normalized_count}")
        address_count = sum(1 for r in records if r.get("street"))
        click.echo(f"Addresses parsed: {address_count}")

        click.echo(f"\n{'=' * 60}")
        click.echo("Enrichment Summary")
        click.echo(f"{'=' * 60}")
        industry_counts = Counter(r.get("industry", "Unknown") for r in records)
        click.echo("Industry distribution:")
        for industry, count in industry_counts.most_common():
            pct = count / len(records) * 100
            click.echo(f"  {industry}: {count} ({pct:.1f}%)")

        size_counts = Counter(r.get("size", "Unknown") for r in records)
        click.echo("\nSize distribution:")
        for size, count in size_counts.most_common():
            pct = count / len(records) * 100
            click.echo(f"  {size}: {count} ({pct:.1f}%)")
    else:
        click.echo(f"No enriched records found at {enriched_path}")

    dedup_path = output_dir / "stage2_duplicates.json"
    if dedup_path.exists():
        with open(dedup_path) as f:
            duplicates = json.load(f)

        click.echo(f"\n{'=' * 60}")
        click.echo("Deduplication Summary")
        click.echo(f"{'=' * 60}")
        click.echo(f"Duplicate pairs found: {len(duplicates)}")

        if duplicates:
            click.echo("\nSample duplicates:")
            for dup in duplicates[:5]:
                click.echo(
                    f"  {dup['name_a']} <-> {dup['name_b']} "
                    f"(fuzzy: {dup['fuzzy_score']}, confidence: {dup['confidence']})"
                )

    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        click.echo(f"\n{'=' * 60}")
        click.echo("Token Usage")
        click.echo(f"{'=' * 60}")
        click.echo(f"Input tokens: {summary['tokens']['input_tokens']:,}")
        click.echo(f"Output tokens: {summary['tokens']['output_tokens']:,}")


def main():
    cli()
