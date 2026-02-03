"""HNSW vector index utilities for semantic search."""

from pathlib import Path

import hnswlib
import numpy as np


def build_index(
    embeddings: list[list[float]],
    dim: int = 1024,
    ef_construction: int = 200,
    m: int = 16,
) -> hnswlib.Index:
    """Build an HNSW index from embedding vectors."""
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=m)
    data = np.array(embeddings, dtype=np.float32)
    index.add_items(data)
    index.set_ef(50)
    return index


def save_index(index: hnswlib.Index, output_path: Path) -> Path:
    """Save HNSW index to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(output_path))
    return output_path


def load_index(index_path: Path, dim: int = 1024) -> hnswlib.Index:
    """Load HNSW index from file."""
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(50)
    return index


def search(
    index: hnswlib.Index,
    query_embedding: list[float],
    k: int = 10,
) -> list[tuple[int, float]]:
    """Search the index for nearest neighbors.

    Returns list of (id, distance) tuples.
    """
    query = np.array([query_embedding], dtype=np.float32)
    labels, distances = index.knn_query(query, k=k)
    return list(zip(labels[0].tolist(), distances[0].tolist()))
