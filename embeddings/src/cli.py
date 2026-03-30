"""CLI for batch embedding and semantic search.

This module handles data preparation, index building, and search.
Batch submission and result retrieval are done via the `dw` CLI —
see the README for the full workflow.
"""

import json
import sys
from pathlib import Path

import click
import numpy as np

from .data import load_documents, load_wikipedia_abstracts, save_documents
from .index import build_index, load_index, save_index, search


@click.group()
def cli():
    """Batch embedding and semantic search via the Doubleword CLI.

    Prepare batch files, then use `dw stream` to submit and retrieve results.
    """
    pass


@cli.command()
@click.option(
    "--limit", "-n", default=10000, help="Number of documents to load (default: 10000)"
)
@click.option("--output", "-o", default="batches/batch.jsonl", help="Output JSONL batch file")
def prepare(limit: int, output: str):
    """Download Wikipedia abstracts and generate an embedding batch JSONL.

    The output file has no model set — use `dw files prepare --model <name>`
    to set the model before submitting.
    """
    click.echo(f"Loading Wikipedia abstracts (limit: {limit})...")
    docs = load_wikipedia_abstracts(limit=limit)
    click.echo(f"Loaded {len(docs)} documents")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write batch-ready JSONL for embeddings (model intentionally omitted)
    with open(output_path, "w") as f:
        for i, doc in enumerate(docs):
            line = {
                "custom_id": f"emb-{i:06d}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "input": doc["text"],
                },
            }
            f.write(json.dumps(line) + "\n")

    # Save documents for later index building and search
    docs_path = output_path.parent / "documents.json"
    save_documents(docs, docs_path)

    click.echo(f"Created {output_path} ({len(docs)} requests)")
    click.echo(f"Documents saved to {docs_path}")



@cli.command("build-index")
@click.option(
    "--results",
    "-r",
    required=True,
    help="Results JSONL file (from `dw stream` or `dw batches results`)",
)
@click.option(
    "--documents",
    "-d",
    default="batches/documents.json",
    help="Documents file from prepare step",
)
@click.option("--output", "-o", default="results/", help="Output directory for index")
def build_index_cmd(results: str, documents: str, output: str):
    """Build a search index from embedding results."""
    results_path = Path(results)
    docs_path = Path(documents)

    if not results_path.exists():
        raise click.ClickException(f"Results file not found: {results_path}")
    if not docs_path.exists():
        raise click.ClickException(f"Documents file not found: {docs_path}")

    docs = load_documents(docs_path)

    # Parse results
    batch_results = {}
    with open(results_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                batch_results[obj["custom_id"]] = obj

    click.echo(f"Loaded {len(batch_results)} embedding results")

    # Determine embedding dimension from the first available result
    dim = None
    model_used = None
    for obj in batch_results.values():
        rb = obj.get("response_body") or obj.get("response", {}).get("body", {})
        data = rb.get("data", [])
        if data:
            dim = len(data[0]["embedding"])
            model_used = rb.get("model")
            break

    if dim is None:
        raise click.ClickException("No valid embedding results found — cannot determine dimension.")

    # Extract embeddings in order, filling missing/errored with zeros
    embeddings = []
    missing = 0
    errors = 0
    for i in range(len(docs)):
        key = f"emb-{i:06d}"
        if key not in batch_results:
            missing += 1
            embeddings.append([0.0] * dim)
            continue

        obj = batch_results[key]
        if obj.get("error"):
            errors += 1
            embeddings.append([0.0] * dim)
            continue

        rb = obj.get("response_body") or obj.get("response", {}).get("body", {})
        data = rb.get("data", [])
        if not data or "embedding" not in data[0]:
            errors += 1
            embeddings.append([0.0] * dim)
            continue

        emb = data[0]["embedding"]
        if len(emb) != dim:
            raise click.ClickException(
                f"Dimension mismatch at {key}: expected {dim}, got {len(emb)}"
            )
        embeddings.append(emb)

    if missing > 0 or errors > 0:
        click.echo(f"Warning: {missing} missing, {errors} errored embeddings (filled with zeros)")

    click.echo(f"Building HNSW index (dim={dim}, {len(embeddings)} vectors)...")
    index = build_index(embeddings, dim=dim)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.bin"
    save_index(index, index_path)
    click.echo(f"Saved index: {index_path}")

    emb_path = output_dir / "embeddings.npy"
    np.save(str(emb_path), np.array(embeddings, dtype=np.float32))
    click.echo(f"Saved embeddings: {emb_path}")

    # Save metadata for search (including model so query uses the same one)
    meta = {
        "documents_file": str(docs_path),
        "index_file": str(index_path),
        "embeddings_file": str(emb_path),
        "embedding_dim": dim,
        "num_documents": len(docs),
    }
    if model_used:
        meta["model"] = model_used
    meta_path = output_dir / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    click.echo(f"Saved metadata: {meta_path}")
    if model_used:
        click.echo(f"Corpus model: {model_used}")


@cli.command("search")
@click.option("--query", "-q", required=True, help="Search query text")
@click.option("--results-dir", "-o", default="results/", help="Directory containing index")
@click.option("--top-k", "-k", default=10, help="Number of results to return")
def search_cmd(query: str, results_dir: str, top_k: int):
    """Search the embedded document corpus.

    Embeds the query via the Doubleword inference API using the same model
    that was used for the corpus. Requires DOUBLEWORD_API_KEY in environment
    (automatically set by `dw project run`).
    """
    import os

    from openai import OpenAI

    output_dir = Path(results_dir)
    meta_path = output_dir / "index_meta.json"

    if not meta_path.exists():
        raise click.ClickException(
            f"No index found in {output_dir}. Run `build-index` first."
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    index_file = metadata["index_file"]
    dim = metadata.get("embedding_dim", 1024)
    index = load_index(Path(index_file), dim=dim)

    docs = load_documents(Path(metadata["documents_file"]))

    # Get query embedding via realtime API
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        raise click.ClickException(
            "DOUBLEWORD_API_KEY not set. Run `export DOUBLEWORD_API_KEY=$(dw keys create --name search --output plain)`"
        )

    model = metadata.get("model")
    if not model:
        raise click.ClickException(
            "No model recorded in index metadata. Rebuild the index from results "
            "that include model info (the model is extracted from the batch response)."
        )

    client = OpenAI(base_url="https://api.doubleword.ai/v1", api_key=api_key)
    click.echo(f"Embedding query with {model}: {query!r}")
    response = client.embeddings.create(
        model=model,
        input=query,
    )
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


def main():
    try:
        cli(standalone_mode=False)
    except SystemExit as e:
        # Click raises SystemExit on --help, errors, etc.
        code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if "datasets" in sys.modules:
            import os
            os._exit(code)
        sys.exit(code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "datasets" in sys.modules:
            import os
            os._exit(1)
        sys.exit(1)

    # Normal completion — force exit if HF datasets was imported
    if "datasets" in sys.modules:
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
