# Doubleword Batch API Examples

Production-ready examples demonstrating what becomes possible when LLM inference is cheap enough to use at scale. Each example includes working code, real-world data, and measured costs.

To get started, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## CLI Examples

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

Each example follows the same structure:

```bash
cd <example-name>
uv sync
export DOUBLEWORD_API_KEY="your-key"
uv run <example-name> run --help
```

Common CLI patterns across all examples:

| Flag | Meaning |
|------|---------|
| `-m`, `--model` | Model alias (`30b`, `235b`) or full name |
| `-n`, `--limit` | Number of items to process |
| `-o`, `--output` | Output directory (default: `results/`) |
| `--dry-run` | Prepare batch file without submitting |
| `status` | Check batch progress |
| `analyze` | Generate accuracy/cost analysis |

## Models

All examples support these models with consistent aliases:

| Alias | Model | Use Case |
|-------|-------|----------|
| `30b` | Qwen3-VL-30B-A3B-Instruct-FP8 | Best value for most tasks |
| `235b` | Qwen3-VL-235B-A22B-Instruct-FP8 | Maximum accuracy |
| `gpt5-nano` | gpt-5-nano | OpenAI budget tier |
| `gpt5-mini` | gpt-5-mini | OpenAI mid-tier |
| `gpt5.2` | gpt-5.2 | OpenAI flagship |

The Qwen models are available through Doubleword's batch API; OpenAI models use their API directly.

## Project Structure

Each example follows this layout:

```
example-name/
├── README.md          # Results, methodology, replication instructions
├── pyproject.toml     # Dependencies and CLI entry point
├── src/
│   ├── cli.py         # Click CLI with run/status/analyze commands
│   ├── batch.py       # Batch API utilities
│   └── ...            # Task-specific modules
├── data/              # Sample data or scripts to fetch it
└── results/           # Output artifacts (gitignored)
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Doubleword API key from [app.doubleword.ai](https://app.doubleword.ai)
- Some examples require additional API keys (Serper for web search, OpenAI for comparison runs)
