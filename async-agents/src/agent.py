"""Agent logic for multi-round web research via the batch API.

Each research round:
1. Generate search queries (batch API)
2. Execute queries via Serper (realtime)
3. Fetch top pages via Jina Reader (realtime)
4. Analyze fetched content (batch API)
5. Generate follow-up queries (batch API)
"""

import re

from .prompts import (
    ANALYSIS_PROMPT,
    ANALYSIS_SYSTEM,
    FOLLOWUP_QUERY_PROMPT,
    QUERY_GENERATION_SYSTEM,
    SEED_QUERY_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM,
)
from .scrape import fetch_urls
from .search import extract_urls, search_batch


def build_seed_query_requests(topic: str, model: str, count: int = 8) -> list[dict]:
    """Create batch request for generating initial search queries."""
    return [
        {
            "custom_id": "seed-queries",
            "model": model,
            "messages": [
                {"role": "system", "content": QUERY_GENERATION_SYSTEM},
                {
                    "role": "user",
                    "content": SEED_QUERY_PROMPT.format(topic=topic, count=count),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    ]


def build_followup_query_requests(
    findings_text: str,
    topic: str,
    model: str,
    round_num: int,
    count: int = 8,
) -> list[dict]:
    """Build batch request for follow-up search queries based on findings."""
    return [
        {
            "custom_id": f"round-{round_num}-queries",
            "model": model,
            "messages": [
                {"role": "system", "content": QUERY_GENERATION_SYSTEM},
                {
                    "role": "user",
                    "content": FOLLOWUP_QUERY_PROMPT.format(
                        topic=topic,
                        findings=findings_text,
                        count=count,
                    ),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    ]


def build_analysis_requests(
    sources: list[dict], topic: str, model: str, round_num: int
) -> list[dict]:
    """Build batch requests to analyze fetched web content.

    Args:
        sources: List of dicts with 'url', 'title', 'content' keys
        topic: Research topic
        model: Model ID
        round_num: Current round number

    Returns:
        List of batch request dicts
    """
    requests_data = []
    for i, source in enumerate(sources):
        # Truncate content to fit in context window
        content = source["content"][:15000]
        requests_data.append(
            {
                "custom_id": f"round-{round_num}-src-{i}",
                "model": model,
                "messages": [
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {
                        "role": "user",
                        "content": ANALYSIS_PROMPT.format(
                            topic=topic,
                            url=source["url"],
                            title=source.get("title", "Unknown"),
                            content=content,
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 2048,
            }
        )
    return requests_data


def build_synthesis_request(
    all_findings: dict[str, str],
    topic: str,
    model: str,
    num_rounds: int,
) -> list[dict]:
    """Build the final synthesis batch request from all analyzed sources."""
    findings_text = "\n\n".join(
        f"--- {cid} ---\n{text}" for cid, text in all_findings.items()
    )

    return [
        {
            "custom_id": "synthesis",
            "model": model,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        topic=topic,
                        num_rounds=num_rounds,
                        findings=findings_text,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
    ]


def extract_queries(text: str) -> list[str]:
    """Parse numbered queries/questions from model output."""
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        match = re.match(r"^\d+[.):\-]\s*(.+)$", line)
        if match:
            query = match.group(1).strip().strip('"').strip("'")
            if query:
                queries.append(query)
    return queries


def extract_content(result: dict) -> str:
    """Extract the assistant message content from a batch result."""
    try:
        return result["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def execute_searches(queries: list[str], results_per_query: int = 5) -> dict:
    """Execute search queries via Serper API.

    Args:
        queries: List of search query strings
        results_per_query: Max results per query

    Returns:
        Dict with 'all_results' (list of search result dicts) and
        'urls' (deduplicated list of URLs)
    """
    search_results = search_batch(queries, max_results=results_per_query)

    all_results = []
    all_urls = []
    seen_urls = set()

    for query, result in search_results.items():
        if "error" in result and result.get("results") == []:
            continue
        for r in result.get("results", []):
            all_results.append({**r, "query": query})
            url = r.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_urls.append(url)

    return {"all_results": all_results, "urls": all_urls}


def fetch_sources(
    urls: list[str], search_results: list[dict], max_pages: int = 10
) -> list[dict]:
    """Fetch page content for top URLs and combine with search metadata.

    Args:
        urls: Deduplicated list of URLs to fetch
        search_results: List of search result dicts with url/title/snippet
        max_pages: Maximum number of pages to fetch

    Returns:
        List of source dicts with url, title, content
    """
    urls_to_fetch = urls[:max_pages]
    fetched = fetch_urls(urls_to_fetch)

    # Build title lookup from search results
    title_lookup = {}
    for r in search_results:
        if r.get("url") and r.get("title"):
            title_lookup[r["url"]] = r["title"]

    sources = []
    for url in urls_to_fetch:
        content = fetched.get(url)
        if content:
            sources.append(
                {
                    "url": url,
                    "title": title_lookup.get(url, "Unknown"),
                    "content": content,
                }
            )

    return sources
