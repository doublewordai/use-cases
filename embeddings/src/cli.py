"""CLI for batch embedding and semantic search."""

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from .batch import (
    count_tokens,
    create_batch,
    create_batch_file,
    download_results,
    extract_embedding,
    get_client,
    parse_results,
    upload_batch_file,
    wait_for_batch,
)
from .data import load_documents, load_wikipedia_abstracts, save_documents
from .index import build_index, load_index, save_index, search

# Embedding models have different aliases
MODELS = {
    "qwen3-emb": "Qwen/Qwen3-Embedding-0.6B",
    "text-emb-large": "text-embedding-3-large",
    "text-emb-small": "text-embedding-3-small",
}
DEFAULT_MODEL = "qwen3-emb"


@click.group()
def cli():
    """Batch embedding and semantic search via Doubleword Batch API.

    Embed large document corpora affordably and build semantic search indices.
    """
    pass


@cli.command()
@click.option(
    "--limit", "-n", default=10000, help="Number of documents to load (default: 10000)"
)
@click.option("--output", "-o", default="data/documents.json", help="Output file path")
def prepare(limit: int, output: str):
    """Download and prepare Wikipedia abstracts."""
    click.echo(f"Loading Wikipedia abstracts (limit: {limit})...")
    docs = load_wikipedia_abstracts(limit=limit)
    output_path = Path(output)
    save_documents(docs, output_path)
    click.echo(f"Saved {len(docs)} documents to {output_path}")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="data/documents.json",
    help="Input documents file",
)
@click.option("--output", "-o", default="results/", help="Output directory")
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    help="Model alias (qwen3-emb, text-emb-large, text-emb-small) or full name",
)
@click.option(
    "--provider",
    "-p",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
@click.option("--limit", "-n", default=0, help="Limit number of documents (0 = all)")
@click.option("--dry-run", is_flag=True, help="Create batch file but don't submit")
@click.option("--wait/--no-wait", default=True, help="Wait for batch completion")
def run(
    input_path: str,
    output: str,
    model: str,
    provider: str,
    limit: int,
    dry_run: bool,
    wait: bool,
):
    """Submit embedding batch for documents."""
    model = MODELS.get(model, model)

    input_file = Path(input_path)
    if not input_file.exists():
        raise click.ClickException(
            f"Documents not found: {input_file}\nRun 'prepare' command first."
        )

    docs = load_documents(input_file)
    if limit > 0:
        docs = docs[:limit]

    click.echo(f"Loaded {len(docs)} documents from {input_file}")

    texts = [doc["text"] for doc in docs]

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")

    batch_file = output_dir / f"batch_{model_slug}_{timestamp}.jsonl"
    create_batch_file(texts, model, batch_file)
    click.echo(f"Created batch file: {batch_file} ({len(texts)} requests)")

    if dry_run:
        click.echo("Dry run - batch not submitted")
        return

    client, provider_name = get_client(provider)
    click.echo(f"Using provider: {provider_name}")

    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)
    click.echo(f"  File ID: {file_id}")

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"  Batch ID: {batch_id}")

    metadata = {
        "batch_id": batch_id,
        "file_id": file_id,
        "provider": provider_name,
        "model": model,
        "num_documents": len(docs),
        "timestamp": timestamp,
        "input_file": str(input_file),
        "batch_file": str(batch_file),
    }
    metadata_file = output_dir / f"batch_{model_slug}_{timestamp}_meta.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    click.echo(f"Saved metadata: {metadata_file}")

    if not wait:
        click.echo(
            f"\nBatch submitted. Check status with:\n"
            f"  embeddings status --batch-id {batch_id}"
        )
        return

    click.echo("\nWaiting for batch completion...")
    batch = wait_for_batch(client, batch_id, poll_interval=15)

    if batch.status != "completed":
        raise click.ClickException(f"Batch failed with status: {batch.status}")

    click.echo("Batch completed!")

    results_file = output_dir / f"results_{model_slug}_{timestamp}.jsonl"
    download_results(batch.output_file_id, results_file, provider=provider_name)
    click.echo(f"Downloaded results: {results_file}")

    # Extract embeddings and build index
    click.echo("Extracting embeddings...")
    results = parse_results(results_file)
    embeddings = []
    for i in range(len(docs)):
        key = f"emb-{i:06d}"
        if key in results:
            embeddings.append(extract_embedding(results[key]))
        else:
            click.echo(f"  Warning: missing result for {key}")
            embeddings.append([0.0] * 1024)

    dim = len(embeddings[0]) if embeddings else 1024
    click.echo(f"Building HNSW index (dim={dim}, {len(embeddings)} vectors)...")
    index = build_index(embeddings, dim=dim)

    index_path = output_dir / f"index_{model_slug}_{timestamp}.bin"
    save_index(index, index_path)
    click.echo(f"Saved index: {index_path}")

    # Save embeddings for later use
    emb_path = output_dir / f"embeddings_{model_slug}_{timestamp}.npy"
    np.save(str(emb_path), np.array(embeddings, dtype=np.float32))
    click.echo(f"Saved embeddings: {emb_path}")

    # Update metadata
    metadata["results_file"] = str(results_file)
    metadata["index_file"] = str(index_path)
    metadata["embeddings_file"] = str(emb_path)
    metadata["output_file_id"] = batch.output_file_id
    metadata["embedding_dim"] = dim
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    tokens = count_tokens(results)
    click.echo(f"\nTokens used: {tokens['input_tokens']:,} input")


@cli.command()
@click.option("--batch-id", required=True, help="Batch ID to check")
@click.option(
    "--provider",
    "-p",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
def status(batch_id: str, provider: str):
    """Check status of a running batch."""
    client, _ = get_client(provider)
    batch = client.batches.retrieve(batch_id)

    click.echo(f"Batch ID: {batch.id}")
    click.echo(f"Status: {batch.status}")
    click.echo(
        f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}"
    )
    if batch.output_file_id:
        click.echo(f"Output file: {batch.output_file_id}")


@cli.command("search")
@click.option("--query", "-q", required=True, help="Search query text")
@click.option(
    "--output", "-o", default="results/", help="Results directory containing index"
)
@click.option(
    "--model", "-m", default=DEFAULT_MODEL, help="Model to use for query embedding"
)
@click.option(
    "--provider",
    "-p",
    default="doubleword",
    type=click.Choice(["doubleword", "openai"]),
)
@click.option("--top-k", "-k", default=10, help="Number of results to return")
def search_cmd(query: str, output: str, model: str, provider: str, top_k: int):
    """Search the embedded document corpus."""
    model = MODELS.get(model, model)
    output_dir = Path(output)

    meta_files = sorted(output_dir.glob("batch_*_meta.json"))
    if not meta_files:
        raise click.ClickException(
            f"No index found in {output_dir}. Run 'embeddings run' first."
        )

    with open(meta_files[-1]) as f:
        metadata = json.load(f)

    index_file = metadata.get("index_file")
    if not index_file or not Path(index_file).exists():
        raise click.ClickException("Index file not found. Run 'embeddings run' first.")

    dim = metadata.get("embedding_dim", 1024)
    index = load_index(Path(index_file), dim=dim)

    docs = load_documents(Path(metadata["input_file"]))

    # Get query embedding via realtime API
    client, _ = get_client(provider)
    click.echo(f"Embedding query: {query!r}")
    response = client.embeddings.create(model=model, input=query)
    query_emb = response.data[0].embedding

    results = search(index, query_emb, k=top_k)

    click.echo(f"\nTop {top_k} results:\n")
    for rank, (doc_idx, distance) in enumerate(results, 1):
        if doc_idx < len(docs):
            doc = docs[doc_idx]
            similarity = 1 - distance
            click.echo(f"{rank}. [{similarity:.3f}] {doc['title']}")
            click.echo(f"   {doc['text'][:150]}...")
            click.echo()


@cli.command()
@click.option("--output", "-o", default="results/", help="Results directory")
def analyze(output: str):
    """Analyze embedding results and token usage."""
    output_dir = Path(output)

    meta_files = sorted(output_dir.glob("batch_*_meta.json"))
    if not meta_files:
        raise click.ClickException(f"No results found in {output_dir}")

    for meta_file in meta_files:
        with open(meta_file) as f:
            metadata = json.load(f)

        click.echo(f"\n{'=' * 50}")
        click.echo(f"Model: {metadata['model']}")
        click.echo(f"Documents: {metadata['num_documents']}")
        click.echo(f"Timestamp: {metadata['timestamp']}")

        results_file = metadata.get("results_file")
        if results_file and Path(results_file).exists():
            results = parse_results(Path(results_file))
            tokens = count_tokens(results)
            click.echo(f"Input tokens: {tokens['input_tokens']:,}")

            sample = next(iter(results.values()))
            emb = extract_embedding(sample)
            click.echo(f"Embedding dimension: {len(emb)}")
            click.echo(f"Results count: {len(results)}")

        if metadata.get("index_file") and Path(metadata["index_file"]).exists():
            click.echo(f"Index: {metadata['index_file']}")
        click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
