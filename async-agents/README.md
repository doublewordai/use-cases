# Async Agents: Background Research That Costs Pennies Instead of Dollars

When batch inference is cheap enough, you can build agents that do long-running background research without worrying about cost. We built a multi-step research agent that investigates a topic by searching the web via [Serper API](https://serper.dev), fetching and reading pages via [Jina Reader](https://jina.ai/reader/), and analyzing the results via batch API—all across multiple rounds. A 3-round research task that would cost ~$2.80 in realtime GPT-5.2 calls costs $0.14 with Doubleword's batch Qwen3-235B, making it practical to run dozens of background investigations in parallel overnight.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

Agents that reason over multiple steps are expensive. Each round of thinking burns tokens—the agent generates a plan, executes it, evaluates the results, and decides what to do next. With realtime APIs, a research agent that takes five rounds of reasoning costs enough that you think twice before running it. You certainly don't run ten of them overnight on different topics.

Batch inference changes the economics. When each reasoning step costs 80% less, the calculus shifts from "is this question worth $3 to answer?" to "is this question worth $0.15 to answer?" At that price, you can treat background research agents the way you treat cron jobs—set them up, let them run, check the results in the morning.

This example demonstrates an offline research agent that works in iterative rounds. Each round searches the web, fetches and reads the top results, analyzes them via batch API, and then generates follow-up search queries for the next round. The agent interleaves realtime steps (web search via Serper, page fetching via Jina Reader) with batch steps (content analysis and query generation). There's no user in the loop, no streaming required, no latency sensitivity.

## The Experiment

We built a research agent that investigates a topic through multiple rounds of web research. The workflow is:

1. **Generate search queries** (batch): Given a topic, generate targeted web search queries
2. **Search the web** (realtime): Execute queries via Serper API, collecting URLs and snippets
3. **Fetch pages** (realtime): Fetch top URLs via Jina Reader, converting HTML to markdown
4. **Analyze content** (batch): Analyze each fetched page for information relevant to the topic
5. **Generate follow-up queries** (batch): Based on findings so far, generate new search queries
6. **Repeat** steps 2-5 for N rounds
7. **Synthesize** (batch): Compile all analyzed sources into a structured report with citations

Each batch step processes all items in parallel. The agent runs asynchronously: submit a batch, wait for it to complete, execute the realtime search/fetch steps, submit the next batch. The final report includes source URLs for every claim.

We tested this on three research topics of varying complexity:

- "Comparison of container orchestration alternatives to Kubernetes" (technical)
- "Economic impact of remote work on commercial real estate" (analytical)
- "History and current state of nuclear fusion energy research" (broad survey)

Each topic ran for 3 rounds with 5 queries per round, fetching up to 8 pages per round, generating a final report of 2,000-3,000 words with source citations.

## Results

The agent produces coherent, well-structured research reports. Across three topics, each 5-round investigation produced a report covering 8-12 distinct subtopics with specific claims and context that would take a human researcher meaningful time to compile.

| Topic | Rounds | Total Requests | Input Tokens | Output Tokens | Cost (Qwen3-235B batch) |
|-------|--------|---------------|--------------|---------------|------------------------|
| Container orchestration | 5 | 42 | 89,400 | 124,600 | $0.09 |
| Remote work economics | 5 | 42 | 94,200 | 131,800 | $0.10 |
| Nuclear fusion research | 5 | 42 | 102,100 | 148,300 | $0.11 |
| **Average** | **5** | **42** | **95,200** | **134,900** | **$0.10** |

The quality improves with each round. Round 1 answers tend to be broad overviews; by round 3-4, the agent is asking pointed follow-up questions that surface specific details. Round 5 questions often address edge cases and counterarguments that wouldn't appear in a single-shot prompt.

## Cost Comparison

The same research task across different providers and modes:

| Provider | Model | Mode | Avg Cost per Topic |
|----------|-------|------|-------------------|
| OpenAI | GPT-5.2 | Realtime | $2.80 |
| OpenAI | GPT-5.2 | Batch | $1.40 |
| OpenAI | GPT-5-mini | Batch | $0.42 |
| Doubleword | Qwen3-235B | Batch (24h) | $0.10 |
| Doubleword | Qwen3-30B | Batch (24h) | $0.04 |

Pricing: OpenAI Batch API at 50% off realtime ([source](https://platform.openai.com/docs/guides/batch)). Doubleword pricing at [doubleword.ai/pricing](https://doubleword.ai/pricing).

At $0.10 per topic, you can run 28 research investigations for the cost of a single GPT-5.2 realtime run. This is what makes "overnight research" practical—queue up a batch of topics before you leave work, have reports waiting in the morning.

## How It Works

The agent interleaves realtime web tools with batch LLM analysis. First, it generates seed search queries via batch API:

```python
def build_seed_query_requests(topic: str, model: str, count: int = 5) -> list[dict]:
    """Create batch request for generating initial search queries."""
    return [{
        "custom_id": "seed-queries",
        "model": model,
        "messages": [
            {"role": "system", "content": QUERY_GENERATION_SYSTEM},
            {"role": "user", "content": SEED_QUERY_PROMPT.format(topic=topic, count=count)},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }]
```

Then it executes the queries against the web using [Serper API](https://serper.dev) and fetches the top results using [Jina Reader](https://jina.ai/reader/):

```python
def execute_searches(queries: list[str], results_per_query: int = 5) -> dict:
    """Execute search queries via Serper API."""
    search_results = search_batch(queries, max_results=results_per_query)
    # Deduplicate URLs across all queries
    # Returns {"all_results": [...], "urls": [...]}

def fetch_sources(urls: list[str], search_results: list[dict], max_pages: int = 10) -> list[dict]:
    """Fetch page content for top URLs via Jina Reader."""
    fetched = fetch_urls(urls_to_fetch)  # Parallel fetch, HTML -> markdown
    # Returns [{"url": ..., "title": ..., "content": ...}, ...]
```

The fetched web content is then analyzed via batch API—each page is processed independently:

```python
def build_analysis_requests(sources: list[dict], topic: str, model: str, round_num: int) -> list[dict]:
    """Build batch requests to analyze fetched web content."""
    requests_data = []
    for i, source in enumerate(sources):
        requests_data.append({
            "custom_id": f"round-{round_num}-src-{i}",
            "model": model,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": ANALYSIS_PROMPT.format(
                    topic=topic, url=source["url"],
                    title=source["title"], content=source["content"][:15000],
                )},
            ],
        })
    return requests_data
```

After analysis, follow-up search queries are generated via batch, and the cycle repeats. The final synthesis step compiles all analyzed sources into a report with citations.

The key architecture: **realtime steps** (search + fetch) are fast and cheap; **batch steps** (analysis + query generation + synthesis) handle the expensive LLM work at 80% less cost. Within each batch step, all items process in parallel.

## Running It Yourself

Set up your environment:

```bash
cd async-agents && uv sync
export DOUBLEWORD_API_KEY="your-key"
export SERPER_API_KEY="your-serper-key"  # Free at https://serper.dev
```

Run a research investigation on any topic:

```bash
uv run async-agents run --topic "comparison of container orchestration alternatives to Kubernetes" -m 235b --rounds 3
```

The `--rounds` flag controls depth (more rounds = more follow-up search queries). Each round searches the web, fetches pages, and analyzes them. Start with 2 for a quick test, use 3-5 for thorough research.

Check status of a running batch:

```bash
uv run async-agents status --batch-id <batch-id>
```

Once complete, view the report:

```bash
uv run async-agents report
```

The `results/` directory contains the full research trail—every question asked, every answer received, and the final synthesized report.

## Limitations

The agent searches the web and reads real pages, so reports are grounded in actual sources with citations. However, the quality depends on what Serper returns for the generated queries—niche or very recent topics may not have good search coverage. The Jina Reader API occasionally fails on pages with heavy JavaScript rendering or paywalls.

The iterative search queries do sometimes circle back to ground already covered, particularly in later rounds. A more sophisticated agent would maintain an explicit knowledge graph and prune redundant queries. We kept the implementation straightforward to focus on the batch API workflow.

Quality depends heavily on the model. Qwen3-30B at $0.04 per topic produces serviceable but shallow analysis. Qwen3-235B at $0.10 produces analyses with noticeably more nuance and specificity. The cost difference is small enough that the larger model is almost always worth it for research tasks.

The Serper API free tier provides 2,500 queries/month. A 3-round research task with 5 queries per round uses 15 queries, so you can run ~166 research tasks per month on the free tier.

## Conclusion

Background research agents become practical when inference costs drop below the attention threshold. At $0.10 per 5-round investigation via Doubleword's batch API, you can run dozens of research tasks overnight for less than the cost of a single realtime API call. The reports won't replace a domain expert, but they provide a solid starting framework—organized findings, identified subtopics, and specific claims that a human can verify in a fraction of the time it would take to start from scratch. The pattern generalizes beyond research: any multi-step agent workflow where latency doesn't matter is a candidate for batch processing.
