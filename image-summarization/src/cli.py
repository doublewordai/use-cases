"""CLI for image summarization via the Doubleword CLI.

This module handles image fetching, encoding, and prompt building.
Batch submission and result retrieval are done via the `dw` CLI —
see the README for the full workflow.
"""

import asyncio
import base64
import io
import json
import math
from pathlib import Path

import click
import pandas as pd
from PIL import Image

# Unsplash originals can exceed Pillow's default pixel limit
Image.MAX_IMAGE_PIXELS = None

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
IMAGES_PER_REQUEST = 20
NUM_BATCH_FILES = 10

PROMPT = (
    "You are given {num_images} images. For each image, write a concise social "
    "media-style summary. Number your summaries 1 through {num_images} to match "
    "the order the images appear.\n\n"
    "Image metadata (in order):\n{metadata}\n\n"
    "Write a concise summary for each image, ignoring any irrelevant metadata."
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
        img.save(buf, format="JPEG", quality=95)
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
    """Image summarization at scale with the Doubleword CLI."""
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
    default="batches",
    type=click.Path(),
    help="Output directory for batch JSONL files",
)
@click.option(
    "--num-images", "-n", default=1000, type=int, help="Number of images to process"
)
def prepare(input_path: str, output_dir: str, num_images: int):
    """Fetch images, encode to base64, and generate batch JSONL files.

    The output files have no model set — use `dw files prepare --model <name>`
    to set the model before submitting with `dw stream` or `dw batches run`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading dataset...")
    df = load_unsplash_data(input_path, num_images)
    click.echo(f"Loaded {len(df)} images")

    # Fetch and encode images
    click.echo("Fetching and resizing images (cached after first download)...")
    urls = df["photo_image_url"].tolist()
    b64_images = asyncio.run(fetch_images(urls, Path("image_cache")))

    # Collect valid images with their metadata
    valid_images = []
    skipped = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        b64 = b64_images[idx]
        if b64 is None:
            skipped += 1
            continue
        description = row.get("photo_description", "") or ""
        photographer = row.get("photographer_username", "") or ""
        valid_images.append((idx, b64, description, photographer))

    click.echo(f"Valid images: {len(valid_images)} ({skipped} skipped)")

    # Group into multi-image requests
    requests_data = []
    for group_start in range(0, len(valid_images), IMAGES_PER_REQUEST):
        group = valid_images[group_start : group_start + IMAGES_PER_REQUEST]
        group_idx = group_start // IMAGES_PER_REQUEST

        metadata_lines = []
        for i, (orig_idx, _, desc, photog) in enumerate(group, 1):
            metadata_lines.append(f"{i}. Caption: {desc} | Photographer: {photog}")
        metadata = "\n".join(metadata_lines)

        prompt = PROMPT.format(num_images=len(group), metadata=metadata)

        content = [{"type": "text", "text": prompt}]
        for _, b64, _, _ in group:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        requests_data.append(
            {
                "custom_id": f"group-{group_idx:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": [{"role": "user", "content": content}],
                },
            }
        )

    click.echo(
        f"Built {len(requests_data)} requests ({IMAGES_PER_REQUEST} images each)"
    )

    # Split across multiple batch files (vision requests can be large)
    requests_per_file = math.ceil(len(requests_data) / NUM_BATCH_FILES)
    batch_files = []
    for file_idx in range(NUM_BATCH_FILES):
        chunk = requests_data[
            file_idx * requests_per_file : (file_idx + 1) * requests_per_file
        ]
        if not chunk:
            break
        batch_file = output_dir / f"batch_{file_idx:02d}.jsonl"
        with open(batch_file, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")
        size_mb = batch_file.stat().st_size / (1024 * 1024)
        batch_files.append(batch_file)
        click.echo(f"  {batch_file} ({size_mb:.1f} MB, {len(chunk)} requests)")

    click.echo(f"\nCreated {len(batch_files)} batch files in {output_dir}/")
    click.echo(
        f"\nNext steps:\n"
        f"  dw files prepare {output_dir}/ --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8\n"
        f"  dw stream {output_dir}/ > results/summaries.jsonl\n"
        f"  # or: dw batches run {output_dir}/ --watch"
    )


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
    "--results",
    "-r",
    required=True,
    help="Results JSONL file (from `dw stream` or `dw batches results`)",
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
    help="Number of images (must match the prepare run)",
)
def analyze(input_path: str, results: str, output_file: str, num_images: int):
    """Parse results and create a CSV of image summaries."""
    results_path = Path(results)
    output_path = Path(output_file)

    if not results_path.exists():
        raise click.ClickException(f"Results file not found: {results_path}")

    df = load_unsplash_data(input_path, num_images)
    click.echo(f"Loaded {len(df)} images from dataset")

    # Parse results
    batch_results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                batch_results[obj["custom_id"]] = obj

    click.echo(f"Loaded {len(batch_results)} results")

    # Token usage
    input_tokens = 0
    output_tokens = 0
    for r in batch_results.values():
        rb = r.get("response_body") or r.get("response", {}).get("body", {})
        usage = rb.get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)

    click.echo(f"Input tokens: {input_tokens:,}")
    click.echo(f"Output tokens: {output_tokens:,}")

    # Extract summaries
    summaries = {}
    errors = 0
    for custom_id, result in batch_results.items():
        try:
            rb = result.get("response_body") or result.get("response", {}).get("body", {})
            content = rb["choices"][0]["message"]["content"]
            summaries[custom_id] = content
        except (KeyError, IndexError, TypeError):
            errors += 1
            summaries[custom_id] = "[ERROR]"

    if errors:
        click.echo(f"Errors: {errors}")

    click.echo(f"Summaries extracted: {len(summaries)}")

    # Build output CSV
    out_df = df[
        ["photo_image_url", "photo_description", "photographer_username"]
    ].copy()

    # Map group summaries back to individual images
    summary_col = []
    for idx in range(len(df)):
        group_idx = idx // IMAGES_PER_REQUEST
        cid = f"group-{group_idx:05d}"
        summary_col.append(summaries.get(cid, ""))

    out_df["summary"] = summary_col

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    click.echo(f"\nSaved {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    cli()
