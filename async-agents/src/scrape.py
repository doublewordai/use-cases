"""Fetch full page content from URLs using Jina Reader API."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch page content using Jina Reader API (converts HTML to markdown).

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Page content as markdown text, or None if failed
    """
    try:
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text[:50000]  # Limit to 50k chars
    except Exception:
        return None


def fetch_urls(urls: list[str], max_workers: int = 5) -> dict[str, Optional[str]]:
    """Fetch multiple URLs in parallel.

    Args:
        urls: List of URLs to fetch
        max_workers: Number of parallel workers

    Returns:
        Dict mapping URL -> content (or None if failed)
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = None
    return results
