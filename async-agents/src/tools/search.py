"""Serper search API wrapper for web research."""

import os

import requests


def get_api_key():
    """Get Serper API key from environment."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable not set")
    return api_key


def search(query: str, max_results: int = 10) -> dict:
    """Run a Serper search.

    Args:
        query: Search query
        max_results: Maximum number of results (default 10)

    Returns:
        Dict with 'results' list containing url, title, snippet
    """
    api_key = get_api_key()

    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key},
        json={"q": query, "num": max_results},
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic", []):
        results.append(
            {
                "url": item.get("link"),
                "title": item.get("title"),
                "snippet": item.get("snippet"),
            }
        )

    return {"results": results}


def search_batch(queries: list[str], max_results: int = 5) -> dict[str, dict]:
    """Run multiple searches.

    Args:
        queries: List of search queries
        max_results: Max results per query

    Returns:
        Dict mapping query -> search results
    """
    results = {}
    for query in queries:
        try:
            results[query] = search(query, max_results=max_results)
        except Exception as e:
            results[query] = {"error": str(e), "results": []}
    return results


def extract_urls(search_results: dict) -> list[str]:
    """Extract unique URLs from search results."""
    urls = []
    seen = set()
    for result in search_results.get("results", []):
        if url := result.get("url"):
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def format_results_for_context(query: str, results: dict) -> str:
    """Format search results as readable text for injection into agent messages.

    Args:
        query: The search query that produced these results
        results: Search results dict with 'results' list

    Returns:
        Formatted string with numbered results including title, URL, and snippet
    """
    items = results.get("results", [])
    if not items:
        return f'Search for "{query}" returned no results.'

    lines = [f'Search results for "{query}":\n']
    for i, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        lines.append(f"{i}. [{title}]({url})")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)
