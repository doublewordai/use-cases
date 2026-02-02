# Batch Document Extraction with Vision-Language Models

Extract structured data from documents at scale using vision-language models. Compare batch API (Doubleword) vs real-time API (OpenAI).

## The Idea

Document extraction at scale requires balancing accuracy, cost, and speed. This use case compares five models across 200 receipt images, finding that Qwen models outperform all OpenAI variants on this vision task.

## Key Results

| Model | Accuracy | Cost/Receipt (batch) |
|-------|----------|---------------------|
| **Qwen3-235B** | **91.8%** | $0.0006 |
| Qwen3-30B | 90.0% | $0.0001 |
| GPT-5-mini | 87.2% | $0.0001 |
| GPT-5.2 | 85.5% | $0.003 |
| GPT-5-nano | 85.5% | $0.000075 |

**Key finding:** Qwen3-30B beats GPT-5-mini at the same batch cost with +2.8% higher accuracy.

## Dataset

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) — 200 scanned Malaysian receipts with ground truth labels for vendor name, date, and total.

## Quick Start

```bash
cd structured-extraction && uv sync

# Download dataset
uv run python -m src.sroie --limit 200

# Run Doubleword batch extraction
export DOUBLEWORD_API_KEY="your-key"
uv run python -m src.cli run -i data/sroie/receipts.jsonl -m 30b,235b -n 1
uv run python -m src.cli status
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl

# Run GPT real-time extraction
export OPENAI_API_KEY="your-key"
uv run python -m src.cli realtime -i data/sroie/receipts.jsonl -m gpt-5-mini
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl -r results/sroie_gpt5mini
```

## Results

### Model Comparison

| Model | Overall | Vendor Name | Date | Total |
|-------|---------|-------------|------|-------|
| GPT-5-nano | 85.5% | 72.5% | 86.5% | 97.5% |
| GPT-5.2 | 85.5% | 72.0% | 87.5% | 97.0% |
| GPT-5-mini | 87.2% | 76.0% | 88.5% | 97.0% |
| Qwen3-VL-30B | 90.0% | 81.5% | 92.0% | 96.5% |
| **Qwen3-VL-235B** | **91.8%** | **85.0%** | **92.5%** | **98.0%** |

### Key Findings

1. **Qwen models outperform all OpenAI models** on this task
2. **GPT-5.2 (flagship) underperforms GPT-5-mini** — surprising result
3. **Vendor names are hardest** — 13 percentage point gap between best (85%) and worst (72%)
4. **Totals are easiest** — all models achieve 96-98%

### Batch Pricing Comparison

| Model | Input/1M | Output/1M | Cost (200 receipts) |
|-------|----------|-----------|---------------------|
| **Qwen3-30B** | $0.05 | $0.15 | **$0.02** |
| GPT-5-nano | $0.05 | $0.20 | $0.015 |
| GPT-5-mini | $0.075 | $0.30 | $0.02 |
| Qwen3-235B | $0.25 | $0.75 | $0.12 |
| GPT-5.2 | $1.25 | $5.00 | $0.60 |

### Accuracy vs Cost

| Model | Accuracy | Cost/Receipt | Value (Accuracy per $0.0001) |
|-------|----------|--------------|------------------------------|
| GPT-5-nano | 85.5% | $0.000075 | 1140% |
| **Qwen3-30B** | **90.0%** | $0.0001 | **900%** |
| GPT-5-mini | 87.2% | $0.0001 | 872% |
| Qwen3-235B | 91.8% | $0.0006 | 153% |
| GPT-5.2 | 85.5% | $0.003 | 29% |

## Processing Performance

| Model | Mode | Concurrency | Time (200 docs) | Throughput |
|-------|------|-------------|-----------------|------------|
| GPT-5.2 | Real-time | 20 | 39 sec | 5.1 docs/sec |
| GPT-5-mini | Real-time | 50 | 43 sec | 4.6 docs/sec |
| GPT-5-nano | Real-time | 50 | 74 sec | 2.7 docs/sec |
| Qwen3-30B | Batch | — | ~2 min | 1.7 docs/sec |
| Qwen3-235B | Batch | — | ~5 min | 0.7 docs/sec |

Real-time API is faster for small batches. Batch API is more economical for bulk processing.

## Error Analysis

| Error Type | GPT-5-nano | GPT-5-mini | GPT-5.2 | Qwen3-30B | Qwen3-235B |
|------------|------------|------------|---------|-----------|------------|
| Vendor name | 55 | 48 | 56 | 37 | 30 |
| Date | 27 | 23 | 25 | 16 | 15 |
| Total | 5 | 6 | 6 | 7 | 4 |

Vendor name errors dominate across all models. Most stem from receipts showing multiple business names (franchise + operator, building + tenant).

## When to Use Each Model

| Use Case | Recommended Model |
|----------|-------------------|
| Best accuracy | Qwen3-235B (91.8%) |
| Best value | Qwen3-30B (90% @ $0.0001) |
| Fastest processing | GPT-5.2 (5.1 docs/sec) |
| Lowest cost | Qwen3-30B ($0.02/200 docs) |

## Conclusion

For document extraction at scale (batch pricing):

1. **Qwen3-30B** matches GPT-5-mini's batch cost ($0.0001/receipt) with +2.8% higher accuracy
2. **Qwen3-235B** achieves the highest accuracy (91.8%) for quality-critical applications
3. **GPT-5.2 underperforms** — costs 30x more than Qwen3-30B with 4.5% lower accuracy
4. **GPT-5-nano** is cheapest but least accurate (85.5%)

At equivalent batch pricing, the Qwen models outperform OpenAI on this vision extraction task.

---

*Data: SROIE (ICDAR 2019), 200 scanned receipts. All costs shown use batch API pricing. Sources: [OpenAI](https://platform.openai.com/docs/pricing), [Doubleword](https://docs.doubleword.ai/batches/model-pricing)*
