# Dataset Compilation

Compile exhaustive datasets of real-world entities using LLM + search. "Find all X that match Y" becomes a batch job.

## The Thesis

Manual dataset compilation takes days and misses the long tail. With LLM + search:
- **Systematically explore** a search space with diverse queries
- **Extract structured data** from unstructured search results
- **Find entities** that manual research and commercial databases miss
- **Cost: $0.35** vs $500+ for analyst time or database subscriptions

## Quick Start

```bash
cd dataset-compilation && uv sync

export DOUBLEWORD_API_KEY="your-key"
export SERPER_API_KEY="your-key"

# 1. Generate diverse search queries
uv run dataset-compilation generate-queries \
  --topic "datacenter networking hardware companies" -n 15

# 2. Run searches via Serper
uv run dataset-compilation search -n 10

# 3. Extract company data (batch LLM)
uv run dataset-compilation extract -m 30b

# 4. Wait for batch completion
uv run dataset-compilation status

# 5. Deduplicate and merge
uv run dataset-compilation dedupe

# 6. Filter to actual hardware vendors (web search + LLM)
uv run dataset-compilation filter -m 30b

# 7. Get filtered results
uv run dataset-compilation filter-status

# 8. Validate against known companies
uv run dataset-compilation validate --known data/known_companies.txt
```

## Commands

### `generate-queries` - Create diverse search queries

```bash
uv run dataset-compilation generate-queries \
  --topic "AI chip startups" \
  --count 20 \
  --model 30b
```

Uses LLM to generate queries covering:
- Major players and emerging companies
- Different geographies
- Product subcategories
- Acquisitions and subsidiaries

### `search` - Run queries through Tavily

```bash
uv run dataset-compilation search --max-results 10
```

### `extract` - Batch extract structured data

```bash
uv run dataset-compilation extract --model 30b
```

Extracts from each search result:
- Company name
- Description
- Products
- Headquarters
- Website
- Confidence level

### `dedupe` - Merge and deduplicate

```bash
uv run dataset-compilation dedupe
```

### `filter` - Validate companies via web search + LLM

```bash
uv run dataset-compilation filter --model 30b
```

For each company:
1. Runs a web search for "{company} networking hardware products"
2. Uses batch LLM to classify based on search results
3. Outputs filtered list of actual networking hardware vendors

Excludes:
- Cloud providers (AWS, CoreWeave)
- Chip companies (unless selling complete hardware)
- Data center operators (Equinix)
- Software-only SDN vendors
- Resellers and distributors

### `filter-status` - Get filter results

```bash
uv run dataset-compilation filter-status
```

Downloads batch results and outputs:
- `results/companies_filtered.json` - Validated hardware vendors
- `results/filter_excluded.json` - Companies excluded with reasoning
- `results/filter_uncertain.json` - Low-confidence classifications

### `validate` - Check against known companies

```bash
echo "Cisco
Juniper
Arista" > known.txt

uv run dataset-compilation validate --known known.txt
```

## Example Topics

| Topic | Expected Companies |
|-------|-------------------|
| Datacenter networking hardware | Cisco, Arista, Juniper + white-box vendors |
| AI chip startups | Groq, Cerebras, SambaNova + emerging players |
| European fintech companies | Revolut, Klarna, N26 + long tail |
| Open source database companies | MongoDB, Cockroach, PlanetScale + niche |

## Available Models

| Alias | Model | Best For |
|-------|-------|----------|
| `30b` | Qwen3-VL-30B | Default, good balance |
| `235b` | Qwen3-VL-235B | Higher accuracy extraction |
| `gpt5-mini` | GPT-5-mini | Alternative |

## Output

`results/companies.json`:
```json
{
  "companies": [
    {
      "name": "Arista Networks",
      "description": "High-performance data center switches",
      "products": ["7000 series", "CloudVision"],
      "headquarters": "Santa Clara, CA",
      "website": "arista.com",
      "mention_count": 13,
      "source_count": 8
    }
  ],
  "total": 341
}
```

## See Also

- [report.md](./report.md) - Full results and analysis
- [RUBRIC.md](../RUBRIC.md) - Evaluation criteria
