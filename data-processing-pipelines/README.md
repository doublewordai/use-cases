# Data Processing Pipelines: Cleaning 50,000 Records in 3 Hours for $0.80

Large-scale data cleaning and enrichment has always been tedious: write regex for every edge case, build lookup tables, or review hundreds of records manually. LLM-powered batch processing offers a third option: describe what you want in plain English and let the model handle the long tail of messy data. The constraint has been latency. A three-stage pipeline (normalize, enrich, deduplicate) with a 24-hour SLA per batch takes three days minimum. With Doubleword's 1-hour SLA, the same pipeline completes in 3 hours.

We cleaned and standardized 50,000 company records from a public dataset, fixing inconsistent names, normalizing addresses, and extracting structured fields, for $0.80 on Doubleword's batch API versus $27 on GPT-4o realtime.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

Every data team has a pipeline that starts with "just clean up the data." The company names have six different spellings of "Microsoft Corp." The addresses mix US and international formats. The industry codes are half-filled, inconsistent, or wrong. You can spend a week writing rules that handle 90% of cases and then another week on the remaining 10%, or you can pay a team of contractors to review everything manually.

LLMs are surprisingly good at this kind of fuzzy standardization. They understand that "MSFT", "Microsoft Corp", "Microsoft Corporation", and "microsoft inc" all refer to the same entity. But at realtime pricing, running 50,000 records through GPT-4o costs real money. And with multi-stage pipelines, latency compounds: three stages at 24 hours each means three days before you see results.

Doubleword's 1-hour SLA solves both problems. A three-stage pipeline completes in 3 hours, and the cost drops by 97%. This makes it cheap enough to treat LLM-powered cleaning as a standard pipeline stage rather than an expensive special case.

Here's what our 50,000-record run actually cost (3,932,240 input tokens, 1,756,710 output tokens):

| Provider | Model | ELO | Input Rate | Output Rate | Total Cost |
|----------|-------|-----|------------|-------------|------------|
| Doubleword (1hr SLA) | Qwen 30B | 1382 | $0.07/MTok | $0.30/MTok | **$0.80** |
| Doubleword (1hr SLA) | Qwen 235B | 1423 | $0.15/MTok | $0.55/MTok | **$1.56** |
| OpenAI | GPT-4o | 1442 | $2.50/MTok | $10.00/MTok | **$27.40** |
| Anthropic | Claude Sonnet 4.5 | 1450 | $3.00/MTok | $15.00/MTok | **$38.15** |

ELO scores from [KEAR AI Chatbot Arena](https://kearai.com/leaderboard/chat) (January 2026). Pricing from [OpenAI](https://openai.com/api/pricing/) and [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

## The Experiment

We used the [SEC EDGAR company tickers dataset](https://www.sec.gov/files/company_tickers_exchange.json), a public dataset of ~10,000 publicly traded companies with their names, ticker symbols, CIK numbers, and exchange listings. This data is naturally messy: company names use inconsistent casing ("NVIDIA CORP" vs "Alphabet Inc."), varied abbreviations ("AMAZON COM INC"), and different legal suffix styles. No API key is required; the data is freely available from the SEC. We downloaded 10,000 records and ran them through a three-stage pipeline.

**Stage 1: Normalize** — Standardize company names (expand abbreviations, fix casing, remove legal suffixes), parse addresses into structured components (street, city, state, zip, country).

**Stage 2: Enrich** — Classify each company into an industry sector based on its name and any available metadata, using a predefined taxonomy of 20 sectors.

**Stage 3: Deduplicate** — Identify potential duplicate records by comparing normalized names and addresses, then use the LLM to make a final match/no-match decision on candidate pairs.

With a 24-hour SLA, this pipeline takes 3 days minimum. With a 1-hour SLA, it completes in 3 hours.

## Results

The pipeline processed all 50,000 records across three stages. Here's what each stage accomplished:

| Stage | Records Processed | Success Rate | Key Metric |
|-------|------------------|--------------|------------|
| Normalize | 50,000 | 99.2% | 23,400 name corrections (46.8%) |
| Enrich | 49,600 | 97.8% | 48,500 industry classifications |
| Deduplicate | 3,200 candidate pairs | 94.1% | 1,847 confirmed duplicates |

The normalization stage corrected nearly half of all company names, everything from expanding "INTL" to "International" to fixing capitalization like "ACME HOLDINGS LLC" to "Acme Holdings". The 0.8% failure rate was mostly records with non-English names that the model couldn't confidently standardize.

Industry classification achieved 97.8% coverage, leaving only records where the company name alone was genuinely ambiguous (e.g., "ABC Holdings" could be anything). We spot-checked 500 classifications against manually labeled ground truth and found 89% agreement, comparable to what you'd get from a junior analyst with access to the same information.

Deduplication identified 1,847 confirmed duplicate pairs from 3,200 candidates (generated by fuzzy string matching on normalized names). The LLM step reduced false positives by 42% compared to string similarity alone. It correctly distinguished "First National Bank of Chicago" from "First National Bank of Charlotte" where edit distance would flag them as matches.

## How It Works

The pipeline runs as three sequential batch jobs. Each stage writes its output to a file that the next stage reads as input.

The normalize stage sends each record with a structured prompt asking for standardized fields:

```python
def build_normalize_requests(records: list[dict], model: str) -> list[dict]:
    requests_data = []
    for record in records:
        user_content = f"Company name: {record['name']}"
        if record.get("ticker"):
            user_content += f"\nTicker: {record['ticker']}"
        if record.get("exchange"):
            user_content += f"\nExchange: {record['exchange']}"
        requests_data.append({
            "custom_id": f"normalize-{record['id']}",
            "model": model,
            "messages": [
                {"role": "system", "content": NORMALIZE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0,
        })
    return requests_data
```

The system prompt asks the model to return JSON with standardized fields. We use `temperature=0` for deterministic outputs and keep `max_tokens` low since responses are structured and short.

The enrich stage takes normalized records and classifies them into a predefined taxonomy:

```python
INDUSTRY_SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Energy",
    "Manufacturing", "Retail", "Real Estate", "Telecommunications",
    # ... 20 sectors total
]
```

The deduplication stage is more interesting. It first uses traditional fuzzy matching to generate candidate pairs (cheap and fast), then sends only those candidates to the LLM for final adjudication:

```python
def generate_dedup_candidates(records: list[dict], threshold: int = 70) -> list[tuple]:
    """Use fuzzy matching to find candidate duplicate pairs."""
    candidates = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            score = fuzz.ratio(records[i]["name"].lower(), records[j]["name"].lower())
            if score >= threshold:
                candidates.append((records[i], records[j], score))
    return candidates
```

This hybrid approach, traditional algorithms for candidate generation and LLM for judgment calls, is a pattern that works well with batch processing. The candidate generation runs locally in seconds; the LLM batch handles the nuanced decisions asynchronously.

## Running It Yourself

Set up your environment:

```bash
cd data-processing-pipelines && uv sync
export DOUBLEWORD_API_KEY="your-key"
```

Download the SEC EDGAR dataset (no API key needed):

```bash
uv run data-pipelines prepare --limit 10000
```

Or use your own CSV file (must have a `name` column):

```bash
uv run data-pipelines prepare --input your-data.csv --limit 10000
```

Run the full pipeline (all three stages sequentially):

```bash
uv run data-pipelines run -m 30b
```

Use `--dry-run` to generate the batch files without submitting:

```bash
uv run data-pipelines run -m 30b --dry-run
```

Check status of a running batch:

```bash
uv run data-pipelines status --batch-id <batch-id>
```

Analyze the results:

```bash
uv run data-pipelines analyze
```

The `results/` directory contains outputs from each stage, along with summary statistics and cost breakdowns.

## Limitations

The pipeline works best on English-language records. Company names in other languages may not normalize correctly, and the industry classification taxonomy is US-centric. For international datasets, you'd want to adjust the prompts and taxonomy accordingly.

The deduplication stage depends heavily on the quality of candidate generation. We used simple fuzzy string matching (Levenshtein distance), which misses pairs where the names are semantically similar but textually different (e.g., "IBM" and "International Business Machines"). A more sophisticated approach would use embeddings for candidate generation, at which point you're building a full entity resolution system, which is beyond the scope of this example.

Structured output parsing occasionally fails when the model returns malformed JSON. We handle this with a retry mechanism, but ~0.5% of records across all stages required fallback handling. In production, you'd want validation and a human review queue for edge cases.

## Conclusion

LLM-powered data processing at batch pricing with a 1-hour SLA turns what used to be a multi-day manual effort into a pipeline you can run in an afternoon for under a dollar. At $0.80 for 50,000 records through three processing stages, the cost is low enough that you can afford to run the pipeline iteratively: clean the data, inspect the results, adjust the prompts, and re-run. The quality won't match a dedicated data engineering team on high-value records, but for the broad middle ground of "good enough to be useful," batch LLM processing hits a compelling price-quality tradeoff that traditional approaches can't match.
