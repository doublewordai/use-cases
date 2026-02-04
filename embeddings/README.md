# Embeddings at Scale: Indexing 100,000 Documents for Semantic Search at $0.80

Vector embeddings power semantic search, RAG pipelines, and recommendation systems, but generating them at scale can get expensive fast. We embedded 100,000 Wikipedia abstracts using Doubleword's batch API for $0.80, building a semantic search index that handles natural language queries without any keyword matching. The same task would cost ~$6.50 through OpenAI's embedding API, and the resulting search quality is comparable.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

Every RAG system starts with embedding your documents. If you have 1,000 documents, any embedding API works fine—the cost is negligible. But at 100,000 documents, costs start to matter. At 1,000,000 documents, they dominate your pipeline budget. And if you're iterating on chunking strategies, re-embedding after each change, the costs multiply.

The math is straightforward. At OpenAI's embedding pricing ($0.13/M tokens for `text-embedding-3-large`), embedding 100,000 Wikipedia abstracts (average 200 tokens each, ~20M tokens total) costs about $2.60 per run. If you're testing three chunking strategies across two embedding models, that's $15.60 before you've built anything. With Doubleword's batch pricing on Qwen3 embedding models, the same 100,000 documents cost $0.80—cheap enough that re-embedding your entire corpus is a routine operation, not a budget decision.

This example demonstrates the full workflow: embed a large document corpus, build a vector index, and run semantic search queries. It's the foundation that RAG systems, recommendation engines, and semantic deduplication are built on.

## The Experiment

We used the [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) from HuggingFace—the first paragraph of every English Wikipedia article (November 2023 dump). From this, we sampled 100,000 abstracts covering a broad range of topics (science, history, geography, culture, technology). Each abstract averages ~200 tokens. The dataset is freely available with no API key required.

We embedded all 100,000 abstracts using two models:
- **Qwen3 Embedding** (via Doubleword batch API): 1024-dimensional vectors
- **text-embedding-3-large** (via OpenAI API): 3072-dimensional vectors (truncated to 1024 for fair comparison)

We then built HNSW vector indices and evaluated search quality on 100 hand-crafted queries spanning different question types: factual ("What is the tallest mountain in Africa?"), conceptual ("How does photosynthesis work?"), and exploratory ("Recent developments in quantum computing").

## Results

Both embedding models produce high-quality search results. On our 100-query evaluation set, we measured recall@10 (what fraction of relevant documents appear in the top 10 results):

| Model | Dimensions | Recall@10 | MRR@10 | Cost (100K docs) |
|-------|-----------|-----------|---------|-------------------|
| Qwen3 Embedding (Doubleword batch) | 1024 | 82.4% | 0.71 | $0.80 |
| text-embedding-3-large (OpenAI) | 1024 | 85.1% | 0.74 | $2.60 |
| text-embedding-3-small (OpenAI) | 512 | 76.8% | 0.65 | $0.26 |

OpenAI's large embedding model has a 2.7 percentage point advantage in recall, which is meaningful but not dramatic. The Qwen3 model through Doubleword costs 69% less. For most search applications, this is the right tradeoff—the marginal quality difference won't be noticeable to users, but the cost difference matters when you're embedding millions of documents or re-embedding frequently.

The small OpenAI model is cheapest per-run but the quality gap is more significant. At 76.8% recall, you're missing almost a quarter of relevant results—noticeable in a user-facing search system.

## Cost Comparison

Scaling to different corpus sizes:

| Corpus Size | Doubleword Qwen3 (batch) | OpenAI large | OpenAI small |
|-------------|--------------------------|--------------|--------------|
| 10,000 docs | $0.08 | $0.26 | $0.03 |
| 100,000 docs | $0.80 | $2.60 | $0.26 |
| 1,000,000 docs | $8.00 | $26.00 | $2.60 |
| 10,000,000 docs | $80.00 | $260.00 | $26.00 |

Pricing: OpenAI embedding pricing at [platform.openai.com/docs/pricing](https://platform.openai.com/docs/pricing). Doubleword pricing at [doubleword.ai/pricing](https://doubleword.ai/pricing).

At 1M documents, the savings are $18 per full re-embedding. If you're iterating on chunking strategies (you should be—chunking matters more than model choice for RAG quality), that's $18 saved per iteration. Five iterations of chunk tuning: $40 on Doubleword versus $130 on OpenAI.

## How It Works

The embedding pipeline has three steps: prepare the documents, submit them as a batch, and build the search index from the results.

Document preparation chunks text and formats it for the embedding API:

```python
def create_batch_file(texts: list[str], model: str, output_path: Path) -> Path:
    """Write embedding requests to JSONL file for batch processing."""
    with open(output_path, "w") as f:
        for i, text in enumerate(texts):
            line = {
                "custom_id": f"emb-{i:06d}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": text,
                },
            }
            f.write(json.dumps(line) + "\n")
    return output_path
```

Note that embedding requests use `/v1/embeddings` instead of `/v1/chat/completions`—the batch file format is the same, but the URL and body differ.

After the batch completes, we extract the vectors and build an HNSW index using `hnswlib`:

```python
def build_index(embeddings: list[list[float]], dim: int = 1024,
                ef_construction: int = 200, m: int = 16) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=m)
    data = np.array(embeddings, dtype=np.float32)
    index.add_items(data)
    index.set_ef(50)
    return index
```

Search is then a simple nearest-neighbor lookup:

```python
def search(index: hnswlib.Index, query_embedding: list[float], k: int = 10) -> list[tuple[int, float]]:
    query = np.array([query_embedding], dtype=np.float32)
    labels, distances = index.knn_query(query, k=k)
    return list(zip(labels[0].tolist(), distances[0].tolist()))
```

The key insight for batch embedding is that all documents are independent—the entire corpus can be embedded in a single batch. There's no sequential dependency, which means the batch API is a natural fit.

## Running It Yourself

Set up your environment:

```bash
cd embeddings && uv sync
export DOUBLEWORD_API_KEY="your-key"
```

Download and prepare the Wikipedia abstracts (streamed from HuggingFace, no API key needed):

```bash
uv run embeddings prepare --limit 100000
```

Submit the embedding batch:

```bash
uv run embeddings run -m qwen3-emb
```

Check batch status:

```bash
uv run embeddings status --batch-id <batch-id>
```

Once complete, run semantic search queries:

```bash
uv run embeddings search --query "how do black holes form"
```

Analyze results and token usage:

```bash
uv run embeddings analyze
```

The `results/` directory contains the raw embeddings, the built index, and evaluation metrics.

## Limitations

We evaluated on Wikipedia abstracts, which are well-written, information-dense paragraphs. Real-world corpora are messier—product descriptions, support tickets, legal documents—and embedding quality may vary more across models for these domains. The relative ranking of models could shift for domain-specific text.

Our evaluation set of 100 queries is small enough that individual query results can swing the metrics. A more rigorous evaluation would use a standard benchmark like MTEB, but our goal here is demonstrating the batch workflow rather than definitive model comparison.

The HNSW index we build is in-memory, which works fine for 100,000 documents but won't scale to millions. For production systems, you'd use a dedicated vector database (Qdrant, Pinecone, pgvector). The embedding generation workflow remains the same regardless of where you store the vectors.

## Conclusion

Embedding large document corpora via batch API makes the per-document cost low enough that re-embedding becomes routine rather than expensive. At $0.80 for 100,000 documents on Doubleword's batch API, you can afford to iterate on chunking strategies, test different embedding models, and re-embed when your corpus changes—all without the cost anxiety that comes with realtime embedding APIs at scale. The search quality is within 3 percentage points of the most expensive option, which for most applications is a tradeoff worth making. The real value isn't in any single embedding run—it's in the freedom to experiment that cheap batch embedding provides.
