# Batch Document Extraction

Extract structured data from documents using vision-language models. Compare batch API (Doubleword) vs real-time API (OpenAI).

## The Idea

Document extraction at scale requires balancing accuracy, cost, and speed. This use case compares:

- **Qwen3-30B** — cheap, fast, good accuracy
- **Qwen3-235B** — highest accuracy, moderate cost
- **GPT-5-mini** — real-time baseline

## Dataset

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019) — 626 scanned receipt images with ground truth labels for vendor name, date, and total.

## Workflow

```bash
# Install
uv sync

# Download dataset
uv run python -m src.sroie --limit 200

# Run Doubleword batch extraction
export DOUBLEWORD_API_KEY="your-key"
uv run python -m src.cli run -i data/sroie/receipts.jsonl -m 30b,235b -n 1
uv run python -m src.cli status
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl

# Run GPT-5-mini real-time extraction
export OPENAI_API_KEY="your-key"
uv run python -m src.cli realtime -i data/sroie/receipts.jsonl -m gpt-5-mini
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl -r results/sroie_gpt5mini
```

## Key Results

| Model | Accuracy | Cost/Receipt |
|-------|----------|--------------|
| Qwen3-30B | 90.0% | $0.0001 |
| Qwen3-235B | 91.8% | $0.0006 |
| GPT-5-mini | 87.2% | $0.0002 |

Qwen3-30B offers the best value: higher accuracy than GPT-5-mini at half the cost.

## See Also

- [report.md](report.md) — Full results and analysis
- [RUBRIC.md](../RUBRIC.md) — Evaluation criteria
