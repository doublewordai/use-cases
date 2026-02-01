# Dataset Compilation with LLM + Search

## Summary

We compiled a dataset of **130 datacenter networking hardware companies** using LLM-generated search queries, Serper search API, batch extraction, and LLM-based filtering. Starting from 561 raw candidates, a validation step filtered out cloud providers, software vendors, and data center operators—leaving verified hardware vendors. The approach found **100% of Gartner Magic Quadrant vendors** (11/11) for under $1.

## Experiment

- **Target**: Companies in the datacenter networking hardware space
- **Method**:
  1. LLM generates 30 diverse search queries
  2. Serper API returns 10 results per query (268 unique URLs)
  3. Batch LLM extracts structured company data from raw snippets
  4. Deduplicate and merge by company name (561 → 252 unique)
  5. **Filter**: Web search per company + LLM classification to validate relevance
- **Model**: Qwen3-VL-30B via Doubleword Batch API
- **Search**: Serper API (raw Google results)

## Results

### Pipeline Stats

| Stage | Count |
|-------|-------|
| Search queries generated | 30 |
| Unique URLs searched | 268 |
| Company mentions extracted | 1,249 |
| After dedupe | 252 |
| **After filtering** | **130** |

### Filtering Results

The filtering step used web search + LLM to classify each company:

| Classification | Count | % |
|----------------|-------|---|
| Matches "datacenter networking hardware" | 130 | 52% |
| Excluded (not hardware) | 122 | 48% |
| Uncertain | 0 | 0% |

**What was filtered out:**

| Category | Examples |
|----------|----------|
| Cloud providers | Amazon, Microsoft, Google, CoreWeave |
| Software-only | Citrix, Cloudflare, Metaswitch, Infovista |
| Data center operators | Digital Realty, Eurofiber |
| Telecom/connectivity | Colt Technology Services |
| Test equipment | Keysight |

**What was kept:**

| Category | Examples |
|----------|----------|
| Network equipment | Cisco, Juniper, Arista, Extreme Networks |
| Hardware OEMs | Dell, HPE, Supermicro |
| White-box vendors | Celestica, Edgecore |
| Silicon + hardware | Nvidia (SuperNICs), Broadcom, Intel (NICs) |

### Validation Against Gartner Magic Quadrant

We validated against the **2025 Gartner Magic Quadrant for Data Center Switching**:

| Quadrant | Company | Found | Filtered |
|----------|---------|-------|----------|
| **Leaders** | Cisco | ✓ | ✓ |
| | Juniper Networks | ✓ | ✓ |
| | Arista Networks | ✓ | ✓ |
| | Huawei | ✓ | ✓ |
| **Challengers** | HPE | ✓ | ✓ |
| | NVIDIA | ✓ | ✓ |
| **Visionaries** | Dell Technologies | ✓ | ✓ |
| | Nokia | ✓ | ✓ |
| **Niche Players** | H3C | ✓ | ✓ |
| | Extreme Networks | ✓ | ✓ |
| | Alcatel-Lucent Enterprise | ✓ | ✓ |

**Recall: 11/11 (100%)** - All Gartner MQ vendors survived filtering.

### Top Companies by Mention Count

After filtering, the most-mentioned companies:

| Company | Mentions |
|---------|----------|
| Cisco | 29 |
| Juniper Networks | 24 |
| Arista Networks | 13 |
| HPE | 12 |
| Dell | 10 |
| Nvidia | 9 |
| Huawei | 7 |
| Intel | 6 |
| Supermicro | 5 |
| Broadcom | 5 |

### Filtering Quality

The LLM filtering was mostly accurate but not perfect:

**Correct exclusions:**
- Amazon, Microsoft (cloud providers, not hardware vendors)
- Cloudflare, Citrix (software companies)
- Digital Realty (data center operator)

**Borderline inclusions (reasonable):**
- Nvidia - included for SuperNICs/DPUs (they do sell networking hardware)
- Intel - included for NICs and Ethernet controllers

**Potential errors:**
- Equinix was incorrectly included (data center operator, not hardware vendor)
- Some duplicates remain (Cisco vs Cisco Systems, Dell vs Dell Technologies)

## Cost Comparison

### This Approach

| Step | API | Requests | Cost |
|------|-----|----------|------|
| Generate queries | Doubleword (real-time) | 2 | $0.004 |
| Search (initial) | Serper | 30 | $0.03 |
| Extract companies | Doubleword (batch) | 300 | $0.10 |
| **Filter: search** | Serper | 252 | $0.25 |
| **Filter: classify** | Doubleword (batch) | 252 | $0.08 |
| **Total** | | | **~$0.47** |

**By category:**
| Category | Cost | % of Total |
|----------|------|------------|
| Search API (Serper) | $0.28 | 60% |
| LLM (Doubleword) | $0.19 | 40% |

### Alternative Approaches

| Method | Estimated Cost | Coverage |
|--------|----------------|----------|
| Manual research (analyst) | $500-2000 | Variable |
| Commercial database (Crunchbase) | $5000+/year | Good for startups |
| **This approach** | **~$0.47** | 130 validated companies |

## Reproducing This Experiment

```bash
cd dataset-compilation && uv sync

export DOUBLEWORD_API_KEY="your-key"
export SERPER_API_KEY="your-key"

# 1. Generate search queries
uv run dataset-compilation generate-queries \
  --topic "datacenter networking hardware companies" -n 30

# 2. Run searches
uv run dataset-compilation search -n 10

# 3. Extract company data (batch)
uv run dataset-compilation extract -m 30b

# 4. Wait for results
uv run dataset-compilation status

# 5. Deduplicate
uv run dataset-compilation dedupe

# 6. Filter to validated companies (web search + LLM)
uv run dataset-compilation filter -m 30b

# 7. Get filtered results
uv run dataset-compilation filter-status

# 8. Validate (optional)
uv run dataset-compilation validate --known data/known_companies.txt
```

## Limitations

1. **Filtering accuracy**: LLM classification isn't perfect—some false positives remain (e.g., Equinix).
2. **Deduplication**: Simple name normalization misses variants (HPE vs Hewlett Packard Enterprise).
3. **Search depth**: 30 queries × 10 results limits coverage; more queries would find more companies.
4. **Prompt sensitivity**: The filter prompt just asks "does this match the category?" without criteria—results depend on LLM's interpretation.

## Conclusion

LLM + search enables dataset compilation with built-in validation:

1. **$0.47 for 130 validated companies** vs. hundreds/thousands for manual research
2. **100% recall** on Gartner MQ vendors (11/11) after filtering
3. **48% noise reduction** - filtering removed cloud providers, software vendors, operators
4. **Transparent reasoning** - each exclusion includes LLM's explanation

The filtering step is the key insight: raw extraction produces noisy results (561 "companies" including Amazon, Microsoft, Cloudflare). A second LLM pass with fresh web search context validates each candidate, cutting the list in half while preserving all true positives.
