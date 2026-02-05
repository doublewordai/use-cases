# Dataset Compilation: 188 Companies for \$1

We built a comprehensive dataset of datacenter networking hardware companies using LLM-powered search and filtering. The pipeline recursively expands a topic description into diverse search queries, extracts structured data with batch inference, then validates each candidate with a second LLM pass. The final dataset contains 188 unique companies with 100% recall against the Gartner Magic Quadrant.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Works

Suppose you're doing competitive analysis for a networking hardware company, or you're an investor trying to map the datacenter infrastructure landscape, or you're identifying acquisition targets in a fragmented market. You need a comprehensive list of companies in the space, not just the obvious names that show up on the first page of Google.

The traditional options are expensive. You can pay a junior analyst to spend a few days searching and compiling a spreadsheet, which typically costs \$500-800 and produces maybe 50 companies before they run out of search variations to try. Or you can subscribe to a commercial database like Crunchbase or PitchBook, which costs thousands per year and tends to be strong on VC-backed startups but weak on established players, international companies, and the long tail of smaller vendors.

The approach here uses LLMs for both query generation and filtering. We start with a topic description and recursively expand it into diverse search queries: the LLM takes "datacenter networking hardware companies" and generates sub-queries that approach it from different angles, and each of those expands again until we have 100+ specific queries covering geographic regions, product categories, business models, and recent events. Each query surfaces candidates the others miss.

Raw extraction produces a lot of noise. Our initial pass found over 1,700 company mentions, but many of them were cloud providers, software vendors, and data center operators that mention networking hardware in their marketing without actually manufacturing it. Equinix, CyrusOne, and Azure all showed up. So we run a second LLM pass: for each candidate, we do a fresh web search and ask the model to classify whether the company actually makes networking hardware or is something else entirely. This filtering step removed 62 false positives while preserving every company from the Gartner Magic Quadrant.

## Results

We compiled a dataset of datacenter networking hardware vendors using this pipeline:

| Stage | Count |
|-------|-------|
| Search queries generated | 125 |
| Unique URLs searched | 714 |
| Company mentions extracted | 1,762 |
| After LLM deduplication | 285 |
| After validation filtering | 223 |
| **Unique companies (final)** | **188** |

The filtering removed 22% of candidates, mostly data center operators (Equinix, CyrusOne), cloud providers (Azure), and software/services companies that showed up because their marketing mentions networking hardware. We run deduplication twice: once on the raw extraction results, and again on the filtered list. The second pass catches duplicates like "Hewlett Packard Enterprise" and "Hewlett-Packard Enterprise (HPE)" that both passed the filter independently.

Here's what got cut and what survived:

| Excluded (62 companies) | Kept (188 unique) |
|--------------------------|-------------------|
| DC operators: Equinix, CyrusOne, Aligned | Network equipment: Cisco, Juniper, Arista, Extreme |
| Cloud providers: Azure | Hardware OEMs: Dell, HPE, Supermicro |
| Software/services: DXC, Ensoft, SolveDirect | White-box vendors: Celestica, Edgecore, Accton |
| | Silicon + hardware: Nvidia, Broadcom, Intel, Marvell |
| | Enterprise: Alcatel-Lucent Enterprise, H3C, Nokia |

### Validation Against Gartner

To verify we weren't just finding well-known names, we checked recall against the [Gartner Magic Quadrant for Data Center Networking](https://www.gartner.com/doc/reprints?id=1-2FXLQWJB):

| Quadrant | Companies | Found | Passed Filter |
|----------|-----------|-------|---------------|
| Leaders | Cisco, Juniper, Arista, Huawei | ✓ | ✓ |
| Challengers | HPE, NVIDIA | ✓ | ✓ |
| Visionaries | Dell, Nokia | ✓ | ✓ |
| Niche Players | H3C, Extreme, Alcatel-Lucent Enterprise | ✓ | ✓ |

We found all 11 vendors and the filtering step classified each one correctly, giving us 100% recall against the analyst benchmark.

### Cost Breakdown

| Step | API | Requests | Cost |
|------|-----|----------|------|
| Generate queries | Doubleword (real-time) | ~50 | \$0.10 |
| Initial search | [Serper](https://serper.dev/pricing) | 125 | \$0.13 |
| Extract companies | Doubleword (batch) | 1,216 | \$0.40 |
| LLM deduplication | Doubleword (batch) | 7 | \$0.02 |
| Validation search | Serper | 285 | \$0.29 |
| Validation classify | Doubleword (batch) | 285 | \$0.09 |
| Final deduplication | Doubleword (batch) | 3 | \$0.01 |
| **Total** | | | **~\$1.05** |

Doubleword batch pricing is 50% off standard rates ([pricing](https://doubleword.ai/pricing)). Serper charges \$0.001 per search.

For comparison:

| Approach | Cost | Coverage |
|----------|------|----------|
| Junior analyst (2 days) | \$500-800 | Finds the obvious names |
| Crunchbase subscription | \$5,000+/year | Good for VC-backed startups, weak on established players |
| **LLM + search (this approach)** | **~\$1** | 188 validated companies, 100% Gartner recall |

The cost difference is roughly 500x. You can run hundreds of queries, validate every candidate, and iterate on the pipeline freely.

## Replication

The pipeline has five commands, each with explicit `--input` and `--output` flags so you can see exactly what flows where.

### Setup

```bash
cd dataset-compilation && uv sync

export DOUBLEWORD_API_KEY="your-key"  # from app.doubleword.ai
export SERPER_API_KEY="your-key"       # from serper.dev
```

### Step 1: Generate Search Queries

```bash
uv run dataset-compilation generate-queries \
  --topic "datacenter networking hardware companies" \
  --max-depth 3 \
  --output queries.json
```

The command recursively expands the topic into diverse search queries. Starting from the topic description, the LLM generates 3-5 sub-queries that approach it from different angles (geography, product type, company type, etc.). Each sub-query expands again until the queries are specific enough to search. A depth-3 tree typically yields 100-150 queries covering:

- Geographic regions: "datacenter switch manufacturers in Taiwan", "Israeli networking hardware startups"
- Product categories: "100GbE NIC vendors", "open source network operating systems"
- Business models: "white-box switch ODMs", "networking hardware for hyperscale datacenters"
- Temporal: "datacenter hardware acquisitions 2024", "emerging networking startups Series A"

Query diversity matters more than volume. The recursive expansion ensures we probe different dimensions of the space rather than generating variations on the same query.

### Step 2: Run Searches

```bash
uv run dataset-compilation search \
  --input queries.json \
  --output search_results.json \
  --results-per-query 50
```

This runs each query through Serper. We use 50 results per query to get broad coverage.

### Step 3: Extract Companies

```bash
uv run dataset-compilation extract \
  --input search_results.json \
  --output companies_raw.json
```

For each search result, we extract structured company data: name, description, products, headquarters, and website. Since we're processing 1,200+ pages, we use batch inference at 50% off real-time pricing. The command waits for the batch to complete by default.

### Step 4: First Dedupe Pass

```bash
uv run dataset-compilation dedupe \
  --input companies_raw.json \
  --output companies.json
```

Raw extraction produces multiple mentions of the same company with variant names ("Cisco" vs "Cisco Systems" vs "Cisco Systems, Inc."). The LLM clusters these duplicates semantically, which handles cases like "HPE" vs "Hewlett Packard Enterprise" that string normalization would miss.

### Step 5: Filter and Validate

```bash
uv run dataset-compilation filter \
  --input companies.json \
  --output companies_filtered.json
```

This step does the heavy lifting. Raw extraction found "Amazon" and "Microsoft" because their marketing pages mention networking hardware, even though they don't manufacture it. For each candidate, we run a fresh web search and ask the LLM to classify whether the company actually makes networking hardware or is something else entirely.

The command also saves `companies_filtered_excluded.json` with the LLM's reasoning for each rejection.

### Step 6: Second Dedupe Pass

```bash
uv run dataset-compilation dedupe \
  --input companies_filtered.json \
  --output companies_final.json
```

We run dedupe again on the smaller, cleaner filtered list. This catches remaining duplicates like "NVIDIA" and "Nvidia Corporation" that both passed the filter independently.

### Validate Against Ground Truth

```bash
uv run dataset-compilation validate \
  --input companies_final.json \
  --ground-truth ground_truth.json
```

This checks recall against known companies (like the Gartner Magic Quadrant) and identifies any remaining duplicate clusters in the output.

## Adapting to Other Domains

The pipeline works for any "find all X that match Y" task. Some examples:

| Topic | What the filter validates |
|-------|---------------------------|
| AI chip startups | Actually builds silicon, not just AI software |
| European fintech companies | Offers financial products, not B2B SaaS |
| Open source database companies | Has an open source project, not just "open" marketing |
| Climate tech hardware | Manufactures physical products, not carbon credits |

The important thing is defining a clear filter criterion that separates true positives from noise. The LLM handles this classification well when you give it fresh web search context about each candidate.

## Limitations

**Filtering accuracy.** The LLM classifier isn't perfect. We manually reviewed a sample and found a ~5% error rate, mostly false positives (companies incorrectly kept). For high-stakes applications, you might want a human review step for low-confidence classifications.

**Deduplication.** LLM-based deduplication catches most variants ("HPE" = "Hewlett Packard Enterprise"), but edge cases slip through. Running the dedupe step twice helps, but for critical applications you should spot-check the output for remaining duplicates.

**Search depth.** We used 125 queries × 50 results. More aggressive expansion (deeper trees, more results per query) would find more companies, though with diminishing returns as coverage saturates.

**Temporal coverage.** The pipeline captures what's currently indexed. Recently founded companies or those with minimal web presence will be missed.

## Conclusion

The pipeline runs LLMs at two stages. First, during query expansion and extraction: we turn a topic description into 125 diverse search queries, then extract structured company data from 1,200+ search results. Second, during filtering: we run a fresh web search for each candidate and ask the LLM to classify whether it actually matches the topic. This second pass cut 62 false positives (data center operators, cloud providers, companies with insufficient web presence) while keeping all 11 vendors from the Gartner Magic Quadrant.

Batch inference makes this practical. Processing 1,500+ LLM requests at real-time API rates would cost several dollars and take hours. Batch pricing brings the total under \$1, and results come back in minutes. At that price point, you can run the pipeline monthly to catch new entrants, or adapt it to adjacent markets.
