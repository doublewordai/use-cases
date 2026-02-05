# Extract Receipt Data for \$0.00019 per Document

**Qwen3-30B beats GPT-5.2 at 1/12th the cost**

Structured extraction from documents is a common production workload: pull vendor names, dates, and totals from receipts; extract fields from invoices; parse information from scanned forms. Vision-language models handle this well, but the choice of model matters more than you'd expect. We ran five models against 626 scanned receipts and found that Qwen3-VL-30B, a relatively small open-weights model, outperforms OpenAI's flagship GPT-5.2 while costing 12x less per document. For applications that need maximum accuracy, Qwen3-VL-235B reaches 93% at under a tenth of a cent per receipt.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Results

We extracted three fields from each receipt (vendor name, date, total amount) and compared against ground truth labels from the [SROIE dataset](https://rrc.cvc.uab.es/?ch=13), an academic benchmark of 626 scanned Malaysian receipts from ICDAR 2019.

| Model | Overall | Vendor Name | Date | Total | Cost (626 receipts) |
|-------|---------|-------------|------|-------|---------------------|
| **Qwen3-VL-235B** | **93.0%** | **87.7%** | **92.5%** | **98.9%** | \$0.58 |
| Qwen3-VL-30B | 90.6% | 86.1% | 90.7% | 94.9% | \$0.12 |
| GPT-5-mini | 87.7% | 77.8% | 88.0% | 97.4% | \$0.15 |
| GPT-5.2 | 86.9% | 77.8% | 86.2% | 96.8% | \$1.51 |
| GPT-5-nano | 84.3% | 73.1% | 84.3% | 95.5% | \$0.23 |

The Qwen models pull ahead on vendor name extraction, which is the hardest field. Qwen3-30B gets 86.1% of vendor names correct versus GPT-5-mini's 77.8%, an 8.3 percentage point gap. This matters because vendor names are where real-world extraction typically fails: receipts display multiple business names (franchise plus operator, building plus tenant), and the model needs to pick the right one.

GPT-5.2 underperforming GPT-5-mini was unexpected. OpenAI's flagship vision model costs 10x more but scores worse on this task (86.9% vs 87.7%). Generic benchmarks won't tell you this; you have to test on your actual workload.

### Cost breakdown

All prices use batch API rates.

The Qwen models show 2x more input tokens than GPT for the same images. This is because Doubleword processes images at higher resolution. Despite the higher token count, Qwen's lower per-token pricing makes it cheaper overall.

The output token differences are also notable: GPT-5-nano generates 833K tokens versus Qwen's 56-59K for the same extraction task. Combined with GPT's higher per-token rates (GPT-5.2 charges 25x more per input token than Qwen3-30B), the cost difference adds up fast.

| Model | Input Tokens | Output Tokens | Batch Cost | Per Receipt |
|-------|--------------|---------------|------------|-------------|
| **Qwen3-VL-30B** | 2.14M | 59K | **\$0.12** | **\$0.00019** |
| GPT-5-mini | 1.00M | 257K | \$0.15 | \$0.00024 |
| GPT-5-nano | 1.23M | 833K | \$0.23 | \$0.00037 |
| Qwen3-VL-235B | 2.14M | 56K | \$0.58 | \$0.00093 |
| GPT-5.2 | 1.00M | 51K | \$1.51 | \$0.00242 |

We ran the GPT models via OpenAI's real-time API (their batch API doesn't support 1-hour SLAs or partial result downloads). For a fair cost comparison, we quote OpenAI's batch pricing throughout.

Prices: [OpenAI pricing](https://platform.openai.com/docs/pricing), [Doubleword model pricing](https://docs.doubleword.ai/batches/model-pricing).

### Which model to use

| Need | Model | Accuracy | Cost/Receipt |
|------|-------|----------|--------------|
| Best value | Qwen3-VL-30B | 90.6% | \$0.00019 |
| Maximum accuracy | Qwen3-VL-235B | 93.0% | \$0.00093 |

The GPT models don't make a compelling case for this task. GPT-5-mini costs more than Qwen3-30B with lower accuracy. GPT-5.2 costs 12x more than Qwen3-30B with even lower accuracy. GPT-5-nano is the cheapest GPT option but has the worst accuracy by a significant margin.

### Error analysis

| Model | Vendor Errors | Date Errors | Total Errors | Total Errors |
|-------|---------------|-------------|--------------|--------------|
| Qwen3-VL-235B | 77 | 47 | 7 | 131 |
| Qwen3-VL-30B | 87 | 58 | 32 | 177 |
| GPT-5-mini | 139 | 75 | 16 | 230 |
| GPT-5.2 | 139 | 86 | 20 | 245 |
| GPT-5-nano | 168 | 98 | 28 | 294 |

Vendor name errors dominate across all models. The Qwen models make roughly half as many vendor name errors as the GPT models (77-87 vs 139-168).

## Replication

### Setup

```bash
cd structured-extraction && uv sync
```

Download the SROIE dataset:

```bash
uv run python -m src.sroie
```

This creates `data/sroie/receipts.jsonl` with 626 receipt images and ground truth labels.

### Running extraction with Doubleword Batch API

Set your API key and submit a batch:

```bash
export DOUBLEWORD_API_KEY="your-key"
uv run python -m src.cli run -i data/sroie/receipts.jsonl -m 30b -n 1
```

The `-m 30b` flag selects Qwen3-VL-30B. Use `-m 235b` for maximum accuracy, or `-m 30b,235b` to run both.

Check status and download results:

```bash
uv run python -m src.cli status
```

Analyze accuracy against ground truth:

```bash
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl
```

### Comparing with OpenAI

For comparison, you can run GPT models via their real-time API:

```bash
export OPENAI_API_KEY="your-key"
uv run python -m src.cli realtime -i data/sroie/receipts.jsonl -m gpt-5-mini
uv run python -m src.cli analyze -i data/sroie/receipts.jsonl -r results/
```

Note: OpenAI's real-time API costs 2x their batch API rates. The costs in our comparison use batch pricing for all models.

## Limitations

**Dataset specificity.** SROIE contains Malaysian receipts with thermal printing, mixed Malay/English text, and specific date formats. Results may differ on US receipts, handwritten documents, or higher-resolution scans.

**Field simplicity.** We extracted three well-defined fields. More complex extraction (line items, addresses, tables) would stress the models differently.

**Ground truth ambiguity.** Some vendor names in the dataset are debatable. Our accuracy numbers reflect agreement with the provided labels, which aren't always unambiguous.

## Conclusion

For document extraction at scale, the Qwen models via Doubleword's Batch API offer the best combination of accuracy and cost:

- **Qwen3-VL-30B** delivers 90.6% accuracy at \$0.00019 per document, beating GPT-5.2 (86.9%) at 1/12th the price
- **Qwen3-VL-235B** reaches 93.0% accuracy at \$0.00093 per document for applications where accuracy is critical

The batch API makes this economical at any scale. Processing 626 receipts cost \$0.12 with the 30B model. Scale linearly from there: 10,000 documents for ~\$2, 100,000 for ~\$20.
