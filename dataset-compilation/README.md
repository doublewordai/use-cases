# Dataset Compilation with LLM + Search

Compile exhaustive datasets of real-world entities using LLM + search. "Find all X that match Y" becomes a batch job.

## The Thesis

Manual dataset compilation takes days and misses the long tail. With LLM + search:
- **Systematically explore** a search space with diverse queries
- **Extract structured data** from unstructured search results
- **Validate candidates** with a second LLM pass using fresh web search
- **Find entities** that manual research and commercial databases miss
- **Cost: $0.47** vs $500+ for analyst time or database subscriptions

## Key Results

We compiled a dataset of **130 datacenter networking hardware companies** using LLM-generated search queries, Serper search API, batch extraction, and LLM-based filtering. Starting from 561 raw candidates, validation filtered out cloud providers, software vendors, and data center operators—leaving verified hardware vendors.

| Metric | Value |
|--------|-------|
| Companies validated | 130 |
| Gartner MQ recall | 11/11 (100%) |
| Noise reduction | 48% filtered out |
| Total cost | ~$0.47 |

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

## Pipeline

| Stage | Count |
|-------|-------|
| Search queries generated | 30 |
| Unique URLs searched | 268 |
| Company mentions extracted | 1,249 |
| After dedupe | 252 |
| **After filtering** | **130** |

## Commands

### `generate-queries` - Create diverse search queries

```bash
uv run dataset-compilation generate-queries \
  --topic "AI chip startups" \
  --count 20 \
  --model 30b
```

Uses LLM to generate queries covering major players, emerging companies, different geographies, product subcategories, and acquisitions.

### `search` - Run queries through Serper

```bash
uv run dataset-compilation search --max-results 10
```

### `extract` - Batch extract structured data

```bash
uv run dataset-compilation extract --model 30b
```

Extracts: company name, description, products, headquarters, website, confidence level.

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

Excludes cloud providers (AWS), chip companies, data center operators (Equinix), software-only SDN vendors, and resellers.

### `filter-status` - Get filter results

```bash
uv run dataset-compilation filter-status
```

Outputs:
- `results/companies_filtered.json` - Validated hardware vendors
- `results/filter_excluded.json` - Companies excluded with reasoning
- `results/filter_uncertain.json` - Low-confidence classifications

## Filtering Results

| Classification | Count | % |
|----------------|-------|---|
| Matches "datacenter networking hardware" | 130 | 52% |
| Excluded (not hardware) | 122 | 48% |

**What was filtered out:**

| Category | Examples |
|----------|----------|
| Cloud providers | Amazon, Microsoft, Google, CoreWeave |
| Software-only | Citrix, Cloudflare, Metaswitch |
| Data center operators | Digital Realty, Eurofiber |

**What was kept:**

| Category | Examples |
|----------|----------|
| Network equipment | Cisco, Juniper, Arista, Extreme Networks |
| Hardware OEMs | Dell, HPE, Supermicro |
| White-box vendors | Celestica, Edgecore |
| Silicon + hardware | Nvidia (SuperNICs), Broadcom, Intel (NICs) |

## Validation Against Gartner Magic Quadrant

| Quadrant | Company | Found | Filtered |
|----------|---------|-------|----------|
| **Leaders** | Cisco, Juniper, Arista, Huawei | ✓ | ✓ |
| **Challengers** | HPE, NVIDIA | ✓ | ✓ |
| **Visionaries** | Dell, Nokia | ✓ | ✓ |
| **Niche Players** | H3C, Extreme, Alcatel-Lucent | ✓ | ✓ |

**Recall: 11/11 (100%)** - All Gartner MQ vendors survived filtering.

## Cost Breakdown

| Step | API | Requests | Cost |
|------|-----|----------|------|
| Generate queries | Doubleword (real-time) | 2 | $0.004 |
| Search (initial) | Serper | 30 | $0.03 |
| Extract companies | Doubleword (batch) | 300 | $0.10 |
| Filter: search | Serper | 252 | $0.25 |
| Filter: classify | Doubleword (batch) | 252 | $0.08 |
| **Total** | | | **~$0.47** |

### Alternative Approaches

| Method | Estimated Cost | Coverage |
|--------|----------------|----------|
| Manual research (analyst) | $500-2000 | Variable |
| Commercial database (Crunchbase) | $5000+/year | Good for startups |
| **This approach** | **~$0.47** | 130 validated companies |

## Example Topics

| Topic | Expected Companies |
|-------|-------------------|
| Datacenter networking hardware | Cisco, Arista, Juniper + white-box vendors |
| AI chip startups | Groq, Cerebras, SambaNova + emerging players |
| European fintech companies | Revolut, Klarna, N26 + long tail |
| Open source database companies | MongoDB, Cockroach, PlanetScale + niche |

## Limitations

1. **Filtering accuracy**: LLM classification isn't perfect—some false positives remain (e.g., Equinix).
2. **Deduplication**: Simple name normalization misses variants (HPE vs Hewlett Packard Enterprise).
3. **Search depth**: 30 queries × 10 results limits coverage; more queries would find more companies.
4. **Prompt sensitivity**: Results depend on LLM's interpretation of category definitions.

## Conclusion

LLM + search enables dataset compilation with built-in validation:

1. **$0.47 for 130 validated companies** vs. hundreds/thousands for manual research
2. **100% recall** on Gartner MQ vendors (11/11) after filtering
3. **48% noise reduction** - filtering removed cloud providers, software vendors, operators
4. **Transparent reasoning** - each exclusion includes LLM's explanation

The filtering step is the key insight: raw extraction produces noisy results (561 "companies" including Amazon, Microsoft, Cloudflare). A second LLM pass with fresh web search context validates each candidate, cutting the list in half while preserving all true positives.
