# Doubleword Batch API Examples

Production-ready examples demonstrating what becomes possible when LLM inference is cheap enough to use at scale. Each example includes working code, real-world data, and measured costs.

To get started, install the [dw CLI](https://github.com/doublewordai/dw) and log in:

```bash
curl -fsSL https://raw.githubusercontent.com/doublewordai/dw/main/install.sh | sh
dw login
```

Or sign up at [app.doubleword.ai](https://app.doubleword.ai) if you don't have an account yet.

## Examples

| Example | What It Does | Cost | Key Insight |
|---------|--------------|------|-------------|
| [async-agents](./async-agents/) | Deep research with recursive agent trees | \$0.34 for 47 agents | 1-hour SLA enables multi-round agentic workflows |
| [synthetic-data-generation](./synthetic-data-generation/) | Generate training data with quality filtering | \$3.21 for 10K samples | 3-stage pipeline in 3 hours, not 3 days |
| [data-processing-pipelines](./data-processing-pipelines/) | Clean and enrich messy records | \$0.80 for 50K records | LLM-powered normalization at pipeline scale |
| [embeddings](./embeddings/) | Semantic search over document corpus | \$0.03 for 1.6M tokens | 70% cheaper than OpenAI for same quality |
| [model-evals](./model-evals/) | Benchmark models on GSM8K | \$0.21 for 1,319 questions | Comprehensive evaluation becomes routine |
| [bug-detection-ensemble](./bug-detection-ensemble/) | Classify security vulnerabilities | \$0.40 for 4,642 samples | Run twice for calibration, still under \$1 |
| [dataset-compilation](./dataset-compilation/) | Build company datasets via search + LLM | \$1.05 for 188 companies | 100% recall vs Gartner Magic Quadrant |
| [structured-extraction](./structured-extraction/) | Extract fields from scanned receipts | \$0.12 for 626 receipts | Qwen3-30B beats GPT-5.2 at 1/12th cost |
| [image-summarization](./image-summarization/) | Caption images for social media | \$0.10 for 1,000 images | Vision batch makes captioning automatic |

## Why These Examples Matter

Each example demonstrates a "more is different" capability: something that becomes qualitatively new when inference costs drop by 10-50x.

**Multi-stage pipelines benefit from the 1-hour SLA.** Synthetic data generation and data processing both run three sequential batches. With a 24-hour SLA, that's three days minimum. With Doubleword's 1-hour SLA, the same pipelines complete in 3 hours. This changes iteration speed: you can refine prompts and re-run the full pipeline multiple times in a single day.

**Agentic workflows compound the SLA advantage.** The async-agents example spawns recursive agent trees that require 20+ batch rounds. At 24 hours per round, that's potentially a month of wall-clock time. At 1 hour per round, it completes in a day.

**Single-batch workloads are about cost.** Embeddings, model evals, and image summarization process everything in one batch. The 24-hour SLA is fine; what matters is the 70-95% cost reduction versus real-time APIs. At these prices, you can embed your entire corpus, evaluate every model on every benchmark, and caption your whole image library.

## Cost Comparison

Across all examples, Doubleword's batch pricing delivers 10-50x cost savings versus real-time APIs from OpenAI and Anthropic:

| Task | Doubleword | OpenAI GPT-4o | Anthropic Sonnet 4.5 | Savings |
|------|------------|---------------|----------------------|---------|
| Deep research (2M tokens) | \$0.34 | \$5.81 | \$7.25 | 17-21x |
| Synthetic data (20M tokens) | \$3.21 | \$108.83 | \$154.62 | 34-48x |
| Data cleaning (6M tokens) | \$0.80 | \$27.40 | \$38.15 | 34-48x |
| Model evaluation (400K tokens) | \$0.21 | \$1.06 | - | 5x |
| Document embeddings (1.6M tokens) | \$0.03 | \$0.10 | - | 3x |

Pricing sources: [Doubleword](https://doubleword.ai/pricing), [OpenAI](https://openai.com/api/pricing/), [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

## Running the Examples

Each example is a **dw project** — a self-contained workflow defined in a `dw.toml` manifest. Clone any example and run it with the [dw CLI](https://github.com/doublewordai/dw):

```bash
dw examples clone model-evals
cd model-evals
dw project setup
dw project info
```

Run everything end-to-end:

```bash
dw project run-all
```

Or run steps individually for more control — see `dw project info` for the full workflow, and each example's README for step-by-step instructions.

### How dw Projects Work

Each example has a `dw.toml` that defines:
- **Setup** — dependency installation (e.g., `uv sync`)
- **Custom project steps** — Python commands for data preparation and analysis
- **Workflow** — the full end-to-end sequence mixing `dw` CLI commands with project steps

The `dw` CLI handles batch file management, submission, progress watching, and result retrieval. Your project steps handle domain-specific logic: loading datasets, building prompts, and analyzing results.

```bash
dw project run prepare          # Your code: generate batch JSONL
dw files prepare batch.jsonl --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
dw batches run batch.jsonl --watch --output-id .batch-id  # CLI: upload, batch, watch
dw batches results --from-file .batch-id -o results/results.jsonl  # CLI: download results
dw project run analyze -- -r results/results.jsonl  # Your code: score/analyze
dw batches analytics --from-file .batch-id  # CLI: cost and token breakdown
```

## Models

Examples use models available on the Doubleword platform:

| Model | Best For | Batch Pricing (per 1M tokens) |
|-------|----------|-------------------------------|
| Qwen3-VL-30B-A3B-Instruct-FP8 | Best value for most tasks | Input \$0.05, Output \$0.20 |
| Qwen3-VL-235B-A22B-Instruct-FP8 | Maximum accuracy | Input \$0.10, Output \$0.40 |
| Qwen3-Embedding-8B | Embeddings | Input \$0.01 |

Set the model when preparing batch files:

```bash
dw files prepare batch.jsonl --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```

See the full model list and current pricing at [doubleword.ai/pricing](https://doubleword.ai/pricing) or run `dw models list`.

## Project Structure

Each example follows this layout:

```
example-name/
├── dw.toml            # Project manifest (steps, workflow)
├── README.md          # Results, methodology, replication instructions
├── pyproject.toml     # Python dependencies and CLI entry point
├── src/
│   ├── cli.py         # Click CLI (prepare, analyze, run, etc.)
│   └── ...            # Task-specific modules
└── results/           # Output artifacts (gitignored)
```

## Requirements

- [dw CLI](https://github.com/doublewordai/dw) — install with `curl -fsSL https://raw.githubusercontent.com/doublewordai/dw/main/install.sh | sh`
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management (installed automatically by `dw project setup` if not present)
- Doubleword account — sign up at [app.doubleword.ai](https://app.doubleword.ai)
- Some examples require additional API keys (Serper for web search)
