"""CLI for dataset compilation using LLM + search."""

import json
from pathlib import Path

import click
from tqdm import tqdm

from .batch import (
    count_tokens,
    create_batch,
    create_batch_file,
    download_results,
    get_client,
    get_response_content,
    parse_results,
    upload_batch_file,
    wait_for_batch,
)
from .search import search, search_batch, extract_urls

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "gpt5-nano": "gpt-5-nano",
    "gpt5-mini": "gpt-5-mini",
    "gpt5.2": "gpt-5.2",
}
DEFAULT_MODEL = "30b"


@click.group()
def main():
    """Compile exhaustive datasets using LLM + search.

    Workflow:
    1. generate-queries - Create diverse search queries for a topic
    2. search - Run queries through Serper
    3. extract - Use batch LLM to extract structured data
    4. dedupe - Merge and deduplicate results
    5. filter - Validate companies via web search + LLM classification
    6. filter-status - Get filter results
    """
    pass


@main.command()
@click.option("--topic", "-t", required=True, help="Topic to research (e.g., 'datacenter networking hardware companies')")
@click.option("--output", "-o", default="data/queries.json", help="Output file for queries")
@click.option("--count", "-n", default=20, help="Number of queries to generate")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias or full name")
def generate_queries(topic: str, output: str, count: int, model: str):
    """Generate diverse search queries for a topic."""
    from openai import OpenAI
    import os

    model = MODELS.get(model, model)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use real-time API for this small task
    client = OpenAI(
        api_key=os.environ.get("DOUBLEWORD_API_KEY"),
        base_url="https://api.doubleword.ai/v1",
    )

    prompt = f"""Generate {count} diverse search queries to find companies in this space:

Topic: {topic}

Generate queries that will find:
- Major established players
- Smaller/emerging companies
- Companies from different geographies
- Different product categories within the space
- Acquisitions and subsidiaries

Each query should be specific enough to return relevant results but different enough to find new companies.

Return a JSON object:
{{
    "topic": "{topic}",
    "queries": [
        "query 1",
        "query 2",
        ...
    ]
}}"""

    click.echo(f"Generating {count} queries for: {topic}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"Generated {len(result['queries'])} queries")
    click.echo(f"Saved to {output_path}")

    # Print queries
    for i, q in enumerate(result["queries"], 1):
        click.echo(f"  {i}. {q}")


@main.command(name="search")
@click.option("--queries", "-q", default="data/queries.json", help="Queries file from generate-queries")
@click.option("--output", "-o", default="data/search_results.json", help="Output file for search results")
@click.option("--max-results", "-n", default=10, help="Max results per query")
def search_cmd(queries: str, output: str, max_results: int):
    """Run search queries through Serper."""
    queries_path = Path(queries)
    output_path = Path(output)

    if not queries_path.exists():
        raise click.ClickException(f"Queries file not found: {queries_path}")

    with open(queries_path) as f:
        data = json.load(f)

    query_list = data.get("queries", [])
    click.echo(f"Running {len(query_list)} queries with {max_results} results each")

    all_results = {"topic": data.get("topic"), "searches": []}

    for query in tqdm(query_list, desc="Searching"):
        result = search(query, max_results=max_results)
        all_results["searches"].append({
            "query": query,
            "results": result.get("results", []),
        })

    # Count unique URLs
    all_urls = set()
    for s in all_results["searches"]:
        for r in s["results"]:
            if url := r.get("url"):
                all_urls.add(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    click.echo(f"\nFound {len(all_urls)} unique URLs across {len(query_list)} queries")
    click.echo(f"Saved to {output_path}")


@main.command()
@click.option("--search-results", "-i", default="data/search_results.json", help="Search results file")
@click.option("--output", "-o", default="data/search_results_full.json", help="Output file with full content")
@click.option("--max-workers", "-w", default=5, help="Parallel fetch workers")
def scrape(search_results: str, output: str, max_workers: int):
    """Fetch full page content for search result URLs."""
    from .scrape import fetch_url

    results_path = Path(search_results)
    output_path = Path(output)

    if not results_path.exists():
        raise click.ClickException(f"Search results not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    # Collect unique URLs
    all_urls = set()
    for s in data["searches"]:
        for r in s["results"]:
            if url := r.get("url"):
                all_urls.add(url)

    click.echo(f"Fetching full content for {len(all_urls)} URLs...")

    # Fetch content for each URL
    url_content = {}
    for url in tqdm(list(all_urls), desc="Fetching"):
        content = fetch_url(url)
        if content:
            url_content[url] = content

    click.echo(f"Successfully fetched {len(url_content)}/{len(all_urls)} pages")

    # Enrich search results with full content
    for s in data["searches"]:
        for r in s["results"]:
            url = r.get("url")
            if url and url in url_content:
                r["full_content"] = url_content[url]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    click.echo(f"Saved enriched results to {output_path}")


@main.command()
@click.option("--search-results", "-i", default="data/search_results.json", help="Search results file")
@click.option("--output", "-o", default="results/", help="Output directory")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias or full name")
@click.option("--dry-run", is_flag=True, help="Prepare batch but don't submit")
def extract(search_results: str, output: str, model: str, dry_run: bool):
    """Extract structured company data from search results using batch LLM."""
    model = MODELS.get(model, model)
    results_path = Path(search_results)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise click.ClickException(f"Search results not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    topic = data.get("topic", "unknown")

    # Collect all search results with context
    extraction_tasks = []
    for search in data["searches"]:
        query = search["query"]
        for result in search["results"]:
            # Use full_content if available (from scrape), else snippet
            content = result.get("full_content") or result.get("content", "")
            extraction_tasks.append({
                "query": query,
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": content[:30000],  # Limit content length
                "has_full_content": bool(result.get("full_content")),
            })

    click.echo(f"Preparing {len(extraction_tasks)} extraction tasks")

    # Build batch requests
    requests_data = []
    for i, task in enumerate(extraction_tasks):
        content_label = "Page content" if task.get("has_full_content") else "Content snippet"
        prompt = f"""Extract company information from this search result.

Topic we're researching: {topic}

Search query: {task['query']}
URL: {task['url']}
Title: {task['title']}
{content_label}: {task['content']}

If this page mentions one or more companies relevant to "{topic}", extract them.
If the page is not about a relevant company (e.g., it's a news aggregator, job board, or unrelated), return an empty list.

Return JSON:
{{
    "companies": [
        {{
            "name": "Company Name",
            "description": "Brief description of what they do",
            "products": ["product1", "product2"],
            "headquarters": "City, Country or null if unknown",
            "website": "company website if mentioned, or null",
            "confidence": "high/medium/low"
        }}
    ],
    "page_type": "company_website/news_article/industry_report/directory/other",
    "relevant": true/false
}}"""

        requests_data.append({
            "custom_id": f"extract-{i:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        })

    batch_file = output_dir / "batch_extract.jsonl"
    create_batch_file(requests_data, batch_file)
    click.echo(f"Created {batch_file} with {len(requests_data)} requests")

    # Save task mapping for later
    task_map = {f"extract-{i:04d}": task for i, task in enumerate(extraction_tasks)}
    with open(output_dir / "task_map.json", "w") as f:
        json.dump(task_map, f, indent=2)

    if dry_run:
        click.echo("Dry run - skipping submission")
        return

    # Submit batch
    client = get_client()
    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch ID: {batch_id}")

    # Save batch info
    info = {"batch_id": batch_id, "file_id": file_id, "task_count": len(requests_data), "model": model}
    with open(output_dir / "batch_info.json", "w") as f:
        json.dump(info, f, indent=2)

    click.echo("Run 'dataset-compilation status' to check progress")


@main.command()
@click.option("--results-dir", "-r", default="results/", help="Results directory")
@click.option("--wait/--no-wait", default=True, help="Wait for batch to complete")
def status(results_dir: str, wait: bool):
    """Check batch status and download results."""
    results_dir = Path(results_dir)
    info_file = results_dir / "batch_info.json"

    if not info_file.exists():
        click.echo("No batch info found. Run 'extract' first.")
        return

    with open(info_file) as f:
        info = json.load(f)

    batch_id = info["batch_id"]
    client = get_client()

    batch = client.batches.retrieve(batch_id)
    click.echo(f"Status: {batch.status}")
    click.echo(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}")

    if batch.status == "in_progress" and wait:
        click.echo("Waiting for completion...")
        batch = wait_for_batch(client, batch_id)

    if batch.status == "completed" and batch.output_file_id:
        results_file = results_dir / "extract_results.jsonl"
        click.echo(f"Downloading results to {results_file}...")
        download_results(client, batch.output_file_id, results_file)
        click.echo("Done!")


@main.command()
@click.option("--results-dir", "-r", default="results/", help="Results directory")
@click.option("--output", "-o", default="results/companies.json", help="Output file")
def dedupe(results_dir: str, output: str):
    """Deduplicate and merge extracted companies."""
    results_dir = Path(results_dir)
    output_path = Path(output)

    results_file = results_dir / "extract_results.jsonl"
    task_map_file = results_dir / "task_map.json"

    if not results_file.exists():
        raise click.ClickException(f"Results file not found: {results_file}")

    # Load task map for context
    task_map = {}
    if task_map_file.exists():
        with open(task_map_file) as f:
            task_map = json.load(f)

    # Parse results
    results = parse_results(results_file)

    # Extract all companies
    all_companies = []
    for custom_id, result in results.items():
        content = get_response_content(result)
        if not content:
            continue

        try:
            data = json.loads(content)
            for company in data.get("companies", []):
                company["_source_id"] = custom_id
                company["_source"] = task_map.get(custom_id, {})
                all_companies.append(company)
        except json.JSONDecodeError:
            continue

    click.echo(f"Found {len(all_companies)} company mentions")

    # Dedupe by normalized name
    def normalize_name(name: str) -> str:
        return name.lower().strip().replace(",", "").replace(".", "").replace(" inc", "").replace(" llc", "").replace(" ltd", "")

    companies_by_name = {}
    for company in all_companies:
        name = company.get("name", "")
        if not name:
            continue

        norm_name = normalize_name(name)
        if norm_name not in companies_by_name:
            companies_by_name[norm_name] = {
                "name": name,
                "descriptions": [],
                "products": set(),
                "headquarters": None,
                "website": None,
                "sources": [],
                "mention_count": 0,
            }

        entry = companies_by_name[norm_name]
        entry["mention_count"] += 1
        if desc := company.get("description"):
            entry["descriptions"].append(desc)
        if products := company.get("products"):
            entry["products"].update(products)
        if hq := company.get("headquarters"):
            entry["headquarters"] = hq
        if website := company.get("website"):
            entry["website"] = website
        entry["sources"].append(company.get("_source", {}))

    # Convert to list and clean up
    deduped = []
    for norm_name, entry in companies_by_name.items():
        deduped.append({
            "name": entry["name"],
            "description": entry["descriptions"][0] if entry["descriptions"] else None,
            "products": list(entry["products"]),
            "headquarters": entry["headquarters"],
            "website": entry["website"],
            "mention_count": entry["mention_count"],
            "source_count": len(set(s.get("url", "") for s in entry["sources"] if s)),
        })

    # Sort by mention count
    deduped.sort(key=lambda x: -x["mention_count"])

    # Try to get topic from search results
    search_results_path = results_dir / "extract_results.jsonl"
    topic = None
    for candidate in [Path("data/search_results.json"), Path("data/queries.json")]:
        if candidate.exists():
            with open(candidate) as f:
                topic = json.load(f).get("topic")
                if topic:
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"topic": topic, "companies": deduped, "total": len(deduped)}, f, indent=2)

    click.echo(f"\nDeduplicated to {len(deduped)} unique companies")
    click.echo(f"Saved to {output_path}")

    # Print top companies
    click.echo("\nTop companies by mentions:")
    for company in deduped[:15]:
        click.echo(f"  {company['mention_count']}x {company['name']}")


@main.command(name="filter")
@click.option("--companies", "-c", default="results/companies.json", help="Companies file")
@click.option("--output", "-o", default="results/", help="Output directory")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias or full name")
@click.option("--search-results", "-n", default=5, help="Search results per company")
@click.option("--topic", "-t", default=None, help="Topic to filter for (defaults to topic from data)")
@click.option("--dry-run", is_flag=True, help="Prepare batch but don't submit")
def filter_cmd(companies: str, output: str, model: str, search_results: int, topic: str, dry_run: bool):
    """Filter companies to only those matching the original topic using web search + LLM."""
    model = MODELS.get(model, model)
    companies_path = Path(companies)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not companies_path.exists():
        raise click.ClickException(f"Companies file not found: {companies_path}")

    with open(companies_path) as f:
        data = json.load(f)

    company_list = data["companies"]

    # Get topic from data or CLI
    if not topic:
        topic = data.get("topic")
    if not topic:
        raise click.ClickException("No topic found in data. Use --topic to specify.")

    click.echo(f"Filtering {len(company_list)} companies for: {topic}")

    # Step 1: Run web searches for each company
    click.echo(f"\nSearching for company info ({search_results} results each)...")
    company_contexts = []

    for company in tqdm(company_list, desc="Searching"):
        name = company["name"]
        query = f'"{name}" {topic}'

        try:
            result = search(query, max_results=search_results)
            snippets = [
                f"- {r.get('title', '')}: {r.get('content', '')}"
                for r in result.get("results", [])
            ]
            context = "\n".join(snippets) if snippets else "No search results found."
        except Exception as e:
            context = f"Search failed: {e}"

        company_contexts.append({
            "company": company,
            "search_context": context,
        })

    # Step 2: Build batch requests for classification
    click.echo(f"\nPreparing {len(company_contexts)} classification requests...")

    requests_data = []
    for i, ctx in enumerate(company_contexts):
        company = ctx["company"]
        prompt = f"""Does this company match the category "{topic}"?

Company: {company["name"]}

Web search results:
{ctx["search_context"]}

Return JSON:
{{
    "company": "{company["name"]}",
    "matches_category": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of what this company does and why you classified it this way",
    "primary_products": ["list", "of", "products"] or null
}}"""

        requests_data.append({
            "custom_id": f"filter-{i:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
        })

    # Save company mapping
    company_map = {f"filter-{i:04d}": ctx["company"] for i, ctx in enumerate(company_contexts)}
    with open(output_dir / "filter_company_map.json", "w") as f:
        json.dump(company_map, f, indent=2)

    batch_file = output_dir / "batch_filter.jsonl"
    create_batch_file(requests_data, batch_file)
    click.echo(f"Created {batch_file} with {len(requests_data)} requests")

    if dry_run:
        click.echo("Dry run - skipping submission")
        return

    # Submit batch
    client = get_client()
    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_file)

    click.echo("Creating batch...")
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch ID: {batch_id}")

    # Save batch info
    info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "task_count": len(requests_data),
        "model": model,
        "type": "filter",
        "topic": topic,
    }
    with open(output_dir / "filter_batch_info.json", "w") as f:
        json.dump(info, f, indent=2)

    click.echo("\nRun 'dataset-compilation filter-status' to check progress and get results")


@main.command()
@click.option("--results-dir", "-r", default="results/", help="Results directory")
@click.option("--output", "-o", default="results/companies_filtered.json", help="Output file")
@click.option("--wait/--no-wait", default=True, help="Wait for batch to complete")
def filter_status(results_dir: str, output: str, wait: bool):
    """Check filter batch status and compile filtered results."""
    results_dir = Path(results_dir)
    output_path = Path(output)
    info_file = results_dir / "filter_batch_info.json"

    if not info_file.exists():
        click.echo("No filter batch info found. Run 'filter' first.")
        return

    with open(info_file) as f:
        info = json.load(f)

    batch_id = info["batch_id"]
    client = get_client()

    batch = client.batches.retrieve(batch_id)
    click.echo(f"Status: {batch.status}")
    click.echo(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total}")

    if batch.status == "in_progress" and wait:
        click.echo("Waiting for completion...")
        batch = wait_for_batch(client, batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        click.echo(f"Batch not ready (status: {batch.status})")
        return

    # Download results
    results_file = results_dir / "filter_results.jsonl"
    click.echo(f"Downloading results to {results_file}...")
    download_results(client, batch.output_file_id, results_file)

    # Load company map and batch info
    with open(results_dir / "filter_company_map.json") as f:
        company_map = json.load(f)

    with open(info_file) as f:
        batch_info = json.load(f)
    topic = batch_info.get("topic", "unknown")

    # Parse and filter
    results = parse_results(results_file)

    included = []
    excluded = []
    uncertain = []

    for custom_id, result in results.items():
        company = company_map.get(custom_id, {})
        content = get_response_content(result)

        if not content:
            uncertain.append({"company": company, "error": "No response"})
            continue

        try:
            classification = json.loads(content)
            classification["_original"] = company

            if classification.get("matches_category"):
                if classification.get("confidence") == "low":
                    uncertain.append(classification)
                else:
                    included.append(classification)
            else:
                excluded.append(classification)
        except json.JSONDecodeError:
            uncertain.append({"company": company, "error": "Invalid JSON"})

    # Build output
    filtered_companies = []
    for item in included:
        orig = item["_original"]
        filtered_companies.append({
            "name": orig["name"],
            "description": orig.get("description"),
            "products": item.get("primary_products") or orig.get("products", []),
            "headquarters": orig.get("headquarters"),
            "website": orig.get("website"),
            "mention_count": orig.get("mention_count", 1),
            "source_count": orig.get("source_count", 1),
            "filter_confidence": item.get("confidence"),
            "filter_reasoning": item.get("reasoning"),
        })

    # Sort by mention count
    filtered_companies.sort(key=lambda x: -x.get("mention_count", 0))

    output_data = {
        "topic": topic,
        "companies": filtered_companies,
        "total": len(filtered_companies),
        "stats": {
            "original": len(company_map),
            "included": len(included),
            "excluded": len(excluded),
            "uncertain": len(uncertain),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Also save detailed results for review
    with open(results_dir / "filter_excluded.json", "w") as f:
        json.dump(excluded, f, indent=2)
    with open(results_dir / "filter_uncertain.json", "w") as f:
        json.dump(uncertain, f, indent=2)

    click.echo(f"\n=== Filter Results ===")
    click.echo(f"Topic: {topic}")
    click.echo(f"Original companies: {len(company_map)}")
    click.echo(f"Matches category: {len(included)}")
    click.echo(f"Excluded (not hardware): {len(excluded)}")
    click.echo(f"Uncertain (needs review): {len(uncertain)}")
    click.echo(f"\nSaved to {output_path}")
    click.echo(f"Excluded companies: {results_dir / 'filter_excluded.json'}")
    click.echo(f"Uncertain companies: {results_dir / 'filter_uncertain.json'}")

    # Show top filtered companies
    click.echo(f"\nTop matches:")
    for company in filtered_companies[:15]:
        click.echo(f"  {company['mention_count']}x {company['name']}")


@main.command()
@click.option("--companies", "-c", default="results/companies.json", help="Companies file")
@click.option("--known", "-k", help="File with known companies to check (one per line)")
def validate(companies: str, known: str):
    """Validate results against known companies."""
    companies_path = Path(companies)

    if not companies_path.exists():
        raise click.ClickException(f"Companies file not found: {companies_path}")

    with open(companies_path) as f:
        data = json.load(f)

    company_names = [c["name"].lower() for c in data["companies"]]

    if known:
        known_path = Path(known)
        if not known_path.exists():
            raise click.ClickException(f"Known companies file not found: {known_path}")

        with open(known_path) as f:
            known_companies = [line.strip().lower() for line in f if line.strip()]

        found = []
        missed = []
        for kc in known_companies:
            # Fuzzy match
            if any(kc in cn or cn in kc for cn in company_names):
                found.append(kc)
            else:
                missed.append(kc)

        click.echo(f"\nValidation against {len(known_companies)} known companies:")
        click.echo(f"  Found: {len(found)}/{len(known_companies)} ({100*len(found)/len(known_companies):.0f}%)")
        click.echo(f"  Missed: {len(missed)}")

        if missed:
            click.echo(f"\nMissed companies:")
            for m in missed:
                click.echo(f"    - {m}")
    else:
        click.echo(f"Total companies found: {len(data['companies'])}")
        click.echo("\nTo validate, create a file with known companies (one per line) and run:")
        click.echo("  dataset-compilation validate --known known_companies.txt")


if __name__ == "__main__":
    main()
