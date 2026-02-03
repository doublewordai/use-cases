"""CLI for image summarisation via the Doubleword Batch API."""

import asyncio
import base64
import io
import json
import os
from pathlib import Path

import click
import pandas as pd
from PIL import Image
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

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

PROMPT = (
    "Summarize this image for a social media-style post.\n\n"
    "Caption: {description}\n\n"
    "Photographer: {photographer}\n\n"
    "Write a concise summary ignoring any irrelevant metadata."
)


def load_unsplash_data(csv_path: str, num_images: int) -> pd.DataFrame:
    """Load and filter the Unsplash Lite dataset."""
    df = pd.read_csv(csv_path, sep="\t", dtype=str, on_bad_lines="skip")
    df = df[df["photo_image_url"].notna()]
    return df.head(num_images)


async def fetch_images(urls: list[str], cache_dir: Path) -> list[str | None]:
    """Fetch images in parallel, resize to target dimensions, return base64 strings."""
    import hashlib
    import ssl

    import aiohttp
    import certifi

    cache_dir.mkdir(exist_ok=True)

    def url_to_cache_path(url: str) -> Path:
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        ext = url.split(".")[-1].split("?")[0].lower()
        if ext not in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}:
            ext = "jpg"
        return cache_dir / f"{url_hash}.{ext}"

    def resize_and_encode(img_bytes: bytes) -> str | None:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return None

        orig_w, orig_h = img.size
        target_aspect = TARGET_WIDTH / TARGET_HEIGHT
        orig_aspect = orig_w / orig_h

        if orig_aspect > target_aspect:
            new_w = int(orig_h * target_aspect)
            left = (orig_w - new_w) // 2
            box = (left, 0, left + new_w, orig_h)
        else:
            new_h = int(orig_w / target_aspect)
            top = (orig_h - new_h) // 2
            box = (0, top, orig_w, top + new_h)

        img = img.crop(box).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async def fetch_one(session: aiohttp.ClientSession, url: str) -> str | None:
        cache_path = url_to_cache_path(url)
        if cache_path.exists():
            img_bytes = cache_path.read_bytes()
        else:
            try:
                timeout = aiohttp.ClientTimeout(total=200)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    img_bytes = await resp.read()
                cache_path.write_bytes(img_bytes)
            except Exception:
                return None

        return resize_and_encode(img_bytes)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)


@click.group()
def cli():
    """Image summarisation at scale with the Doubleword Batch API."""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to Unsplash Lite photos.csv000 file",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    default="results",
    type=click.Path(),
    help="Output directory for batch files and results",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    help="Model alias (30b, 235b, gpt5-nano, gpt5-mini, gpt5.2) or full name",
)
@click.option(
    "--num-images", "-n", default=1000, type=int, help="Number of images to process"
)
@click.option("--dry-run", is_flag=True, help="Prepare batch file but don't submit")
def run(input_path: str, output_dir: str, model: str, num_images: int, dry_run: bool):
    """Fetch images, build prompts, and submit a batch for summarisation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_full = MODELS.get(model, model)
    model_short = next(
        (k for k, v in MODELS.items() if v == model_full), model.split("/")[-1]
    )

    click.echo(f"Model: {model_full}")
    click.echo(f"Loading dataset...")
    df = load_unsplash_data(input_path, num_images)
    click.echo(f"Loaded {len(df)} images")

    # Fetch and encode images
    click.echo("Fetching and resizing images...")
    urls = df["photo_image_url"].tolist()
    b64_images = asyncio.run(fetch_images(urls, Path("image_cache")))

    # Build batch requests
    requests_data = []
    skipped = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        b64 = b64_images[idx]
        if b64 is None:
            skipped += 1
            continue

        description = row.get("photo_description", "") or ""
        photographer = row.get("photographer_username", "") or ""
        prompt = PROMPT.format(description=description, photographer=photographer)

        requests_data.append(
            {
                "custom_id": f"img-{idx:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_full,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                },
            }
        )

    click.echo(f"Built {len(requests_data)} requests ({skipped} images skipped)")

    # Write batch file
    batch_file = output_dir / f"batch_{model_short}.jsonl"
    create_batch_file(requests_data, batch_file)
    click.echo(f"Created {batch_file}")

    if dry_run:
        click.echo("Dry run - skipping submission")
        return

    # Submit batch
    client = get_client()
    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)
    click.echo(f"File ID: {file_id}")

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch ID: {batch_id}")

    # Save batch info
    batch_info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "model": model_full,
        "model_short": model_short,
    }
    info_path = output_dir / f"batch_{model_short}_info.json"
    with open(info_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    click.echo(f"Batch submitted. Run 'image-summarisation status' to check progress.")


@cli.command()
@click.option(
    "--output",
    "-o",
    "output_dir",
    default="results",
    type=click.Path(exists=True),
    help="Output directory with batch info files",
)
@click.option("--wait/--no-wait", default=True, help="Wait for batches to complete")
def status(output_dir: str, wait: bool):
    """Check batch status and download results when complete."""
    output_dir = Path(output_dir)
    client = get_client()

    info_files = sorted(output_dir.glob("batch_*_info.json"))
    if not info_files:
        raise click.ClickException("No batch info files found. Run 'run' first.")

    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)

        batch_id = info["batch_id"]
        model_short = info.get("model_short", "default")
        click.echo(f"\n=== Batch {model_short} ({batch_id}) ===")

        batch = client.batches.retrieve(batch_id)
        click.echo(f"Status: {batch.status}")
        click.echo(
            f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
        )

        if batch.status == "in_progress" and wait:
            click.echo("Waiting for completion...")
            batch = wait_for_batch(client, batch_id)

        if batch.status == "completed" and batch.output_file_id:
            results_file = output_dir / f"results_{model_short}.jsonl"
            click.echo(f"Downloading results to {results_file}...")
            download_results(client, batch.output_file_id, results_file)
            click.echo("Done!")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to Unsplash Lite photos.csv000 file",
)
@click.option(
    "--results-dir",
    "-r",
    "results_dir",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result files",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="results/summaries.csv",
    help="Output CSV file with summaries",
)
@click.option(
    "--num-images",
    "-n",
    default=1000,
    type=int,
    help="Number of images (must match the run)",
)
def analyze(input_path: str, results_dir: str, output_file: str, num_images: int):
    """Combine results into a CSV and print token usage summary."""
    results_dir = Path(results_dir)
    output_file = Path(output_file)

    df = load_unsplash_data(input_path, num_images)
    click.echo(f"Loaded {len(df)} images from dataset")

    result_files = sorted(results_dir.glob("results_*.jsonl"))
    if not result_files:
        raise click.ClickException(
            "No result files found. Run 'status' first to download results."
        )

    for result_file in result_files:
        model_short = result_file.stem.replace("results_", "")
        click.echo(f"\n=== {model_short} ===")

        results = parse_results(result_file)
        click.echo(f"Results: {len(results)}")

        # Token usage
        tokens = count_tokens(results)
        click.echo(f"Input tokens: {tokens['input_tokens']:,}")
        click.echo(f"Output tokens: {tokens['output_tokens']:,}")

        # Extract summaries and attach to dataframe
        summaries = {}
        errors = 0
        for custom_id, result in results.items():
            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                summaries[custom_id] = content
            except (KeyError, IndexError, TypeError):
                errors += 1
                summaries[custom_id] = "[ERROR]"

        if errors:
            click.echo(f"Errors: {errors}")

        # Map summaries back to dataframe rows by index
        summary_col = []
        for idx in range(len(df)):
            cid = f"img-{idx:05d}"
            summary_col.append(summaries.get(cid, ""))

        out_df = df[
            ["photo_image_url", "photo_description", "photographer_username"]
        ].copy()
        out_df["summary"] = summary_col
        out_df["model"] = model_short

        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_file, index=False)
        click.echo(f"Saved {len(out_df)} summaries to {output_file}")


if __name__ == "__main__":
    cli()
