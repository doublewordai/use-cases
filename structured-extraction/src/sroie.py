"""Download and prepare SROIE dataset.

SROIE (Scanned Receipts OCR and Information Extraction) is a benchmark from ICDAR 2019.
Official site: https://rrc.cvc.uab.es/?ch=13

We use a GitHub mirror that includes both images and JSON labels.
"""

import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import click

# SROIE dataset mirror with JSON labels
# Original: https://rrc.cvc.uab.es/?ch=13
SROIE_URL = "https://github.com/zzzDavid/ICDAR-2019-SROIE/archive/refs/heads/master.zip"


def parse_sroie_json(label_path: Path) -> dict:
    """Parse SROIE JSON label file."""
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)

    labels = {}

    # Map SROIE keys to our schema
    if "company" in data:
        labels["vendor_name"] = data["company"]
    if "address" in data:
        labels["vendor_address"] = data["address"]
    if "date" in data:
        labels["date"] = data["date"]
    if "total" in data:
        try:
            labels["total"] = float(data["total"].replace(",", ""))
        except (ValueError, AttributeError):
            labels["total"] = data["total"]

    return labels


def prepare_dataset(data_dir: Path, output_path: Path, limit: int = None):
    """Prepare SROIE data as JSONL with image paths and labels."""
    # Find image and label directories
    img_dir = data_dir / "data" / "img"
    key_dir = data_dir / "data" / "key"

    if not img_dir.exists():
        raise click.ClickException(f"Could not find image directory at {img_dir}")
    if not key_dir.exists():
        raise click.ClickException(f"Could not find key directory at {key_dir}")

    records = []
    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    for img_path in image_files:
        # Find corresponding label file (JSON)
        label_path = key_dir / f"{img_path.stem}.json"
        if not label_path.exists():
            continue

        try:
            labels = parse_sroie_json(label_path)
        except Exception as e:
            click.echo(f"Warning: Failed to parse {label_path}: {e}")
            continue

        if not labels:
            continue

        records.append({
            "id": img_path.stem,
            "image_path": str(img_path.absolute()),
            "ground_truth": labels,
        })

        if limit and len(records) >= limit:
            break

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return len(records)


@click.command()
@click.option("--output-dir", "-o", default="data/sroie", help="Output directory for dataset")
@click.option("--limit", "-l", type=int, help="Limit number of samples")
def download_sroie(output_dir: str, limit: int):
    """Download and prepare SROIE dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "sroie.zip"
    extract_dir = output_dir / "raw"

    # Download if needed
    if not extract_dir.exists():
        click.echo("Downloading SROIE dataset...")
        urlretrieve(SROIE_URL, zip_path)
        click.echo(f"Downloaded to {zip_path}")

        click.echo("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        click.echo(f"Extracted to {extract_dir}")

    # Find the actual data directory
    data_dir = extract_dir / "ICDAR-2019-SROIE-master"
    if not data_dir.exists():
        dirs = list(extract_dir.glob("*"))
        if dirs:
            data_dir = dirs[0]

    click.echo(f"Found data at {data_dir}")

    # Prepare dataset
    output_path = output_dir / "receipts.jsonl"
    count = prepare_dataset(data_dir, output_path, limit)
    click.echo(f"Prepared {count} receipts to {output_path}")


if __name__ == "__main__":
    download_sroie()
