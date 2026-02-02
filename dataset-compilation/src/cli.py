"""CLI for dataset compilation using LLM + search.

Pipeline:
    generate-queries → search → extract → dedupe → filter → dedupe → validate

For more comprehensive coverage, add an expansion pass:
    expand (from first-pass companies) → search → extract → dedupe → filter → merge

Each command has explicit --input and --output flags. Batch commands wait
for completion by default.

Example (single pass):
    dataset-compilation generate-queries --topic "datacenter networking hardware" -o queries.json
    dataset-compilation search -i queries.json -o search_results.json
    dataset-compilation extract -i search_results.json -o companies_raw.json
    dataset-compilation dedupe -i companies_raw.json -o companies.json
    dataset-compilation filter -i companies.json -o companies_filtered.json
    dataset-compilation dedupe -i companies_filtered.json -o companies_final.json
    dataset-compilation validate -i companies_final.json

Example (with expansion):
    # First pass
    dataset-compilation generate-queries --topic "datacenter networking hardware" -o queries.json
    dataset-compilation search -i queries.json -o search_results.json
    dataset-compilation extract -i search_results.json -o companies_raw.json
    dataset-compilation dedupe -i companies_raw.json -o companies.json
    dataset-compilation filter -i companies.json -o companies_filtered.json

    # Expansion pass
    dataset-compilation expand -i companies_filtered.json -t "datacenter networking hardware" -o queries_exp.json
    dataset-compilation search -i queries_exp.json -o search_exp.json
    dataset-compilation extract -i search_exp.json -o companies_exp_raw.json
    dataset-compilation dedupe -i companies_exp_raw.json -o companies_exp.json
    dataset-compilation filter -i companies_exp.json -o companies_exp_filtered.json

    # Merge and final dedupe
    dataset-compilation merge -i companies_filtered.json -i companies_exp_filtered.json -o merged.json
    dataset-compilation dedupe -i merged.json -o companies_final.json
    dataset-compilation validate -i companies_final.json
"""

import json
import os
from pathlib import Path

import click
from tqdm import tqdm

from .batch import (
    create_batch,
    create_batch_file,
    download_results,
    get_client,
    get_response_content,
    parse_results,
    upload_batch_file,
    wait_for_batch,
)
from .search import search

MODELS = {
    "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
}
DEFAULT_MODEL = "30b"


@click.group()
def main():
    """Compile exhaustive datasets using LLM + search.

    Pipeline: generate-queries → search → extract → dedupe → filter → dedupe → validate

    Each command takes --input and --output. Run dedupe twice for best results
    (after extract and after filter).
    """
    pass


# -----------------------------------------------------------------------------
# generate-queries: Generate diverse search queries for a topic
# -----------------------------------------------------------------------------

@main.command()
@click.option("--topic", "-t", required=True, help="Topic to research")
@click.option("--output", "-o", default="queries.json", help="Output file")
@click.option("--max-depth", "-d", default=3, help="Max expansion depth")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
def generate_queries(topic: str, output: str, max_depth: int, model: str):
    """Generate diverse search queries by recursive expansion.

    Starts with the topic and recursively expands into diverse sub-queries.
    The LLM decides when a query is specific enough to search (returns SEARCH).
    A depth-3 tree typically yields 50-150 leaf queries.
    """
    from openai import OpenAI

    model = MODELS.get(model, model)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ.get("DOUBLEWORD_API_KEY"),
        base_url="https://api.doubleword.ai/v1",
    )

    def expand(query: str, path: list[str], depth: int) -> list[dict]:
        """Expand a query into sub-queries, or mark as ready to search."""
        if depth >= max_depth:
            return [{"query": query, "path": path, "depth": depth}]

        path_str = " → ".join(path) if path else "(root)"
        prompt = f"""Expand this into 5-8 different search queries, or reply SEARCH if it's already specific enough to search.

Query: {query}
Path so far: {path_str}

Generate diverse queries that approach this from DIFFERENT ANGLES (geography, product type, company type, time period, industry term variations). Not minor variations of the same query.

If this query is already specific enough to get good search results, just reply: SEARCH

Otherwise, reply with 5-8 different search queries, one per line. No numbering, no bullets, just the queries."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )

        content = response.choices[0].message.content.strip()

        if content.upper() == "SEARCH" or content.upper().startswith("SEARCH"):
            return [{"query": query, "path": path, "depth": depth}]

        # Parse sub-queries
        sub_queries = [line.strip() for line in content.split("\n") if line.strip()]
        sub_queries = [q for q in sub_queries if len(q) > 5 and not q.upper().startswith("SEARCH")]

        if not sub_queries:
            return [{"query": query, "path": path, "depth": depth}]

        # Recurse on each sub-query
        results = []
        for sq in sub_queries[:8]:  # Limit to 8 children
            results.extend(expand(sq, path + [query], depth + 1))
        return results

    click.echo(f"Expanding queries for: {topic}")
    click.echo(f"Max depth: {max_depth}")

    # Start expansion from the topic
    all_queries = expand(topic, [], 0)

    # Extract unique query strings
    seen = set()
    unique_queries = []
    for item in all_queries:
        q = item["query"]
        if q not in seen and q != topic:
            seen.add(q)
            unique_queries.append(q)

    output_data = {
        "topic": topic,
        "queries": unique_queries,
        "expansion_tree": all_queries,
        "stats": {
            "total_leaves": len(all_queries),
            "unique_queries": len(unique_queries),
            "max_depth": max_depth,
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"Generated {len(unique_queries)} unique queries → {output_path}")


# -----------------------------------------------------------------------------
# expand: Generate competitor/alternative queries from found entities
# -----------------------------------------------------------------------------

@main.command()
@click.option("--input", "-i", "input_file", required=True, help="Companies JSON file")
@click.option("--topic", "-t", required=True, help="Topic description (for context)")
@click.option("--output", "-o", default="queries_expanded.json", help="Output file")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
@click.option("--sample", "-s", default=50, help="Number of companies to sample (0 = all)")
def expand(input_file: str, topic: str, output: str, model: str, sample: int):
    """Generate competitor/alternative queries from found entities.

    Takes companies from a previous run and generates queries like
    "competitors of Cisco", "alternatives to Arista switches" to find
    more entities in the same space.

    Use this after a first pass to expand coverage.
    """
    from openai import OpenAI
    import random

    model = MODELS.get(model, model)
    input_path = Path(input_file)
    output_path = Path(output)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    companies = data.get("companies", [])

    # Sample if requested
    if sample and len(companies) > sample:
        # Prefer companies with more mentions (likely more prominent)
        companies_sorted = sorted(companies, key=lambda x: -x.get("mention_count", 1))
        # Take top half by mentions, sample the rest
        top_half = companies_sorted[:sample // 2]
        rest = companies_sorted[sample // 2:]
        random.shuffle(rest)
        companies = top_half + rest[:sample - len(top_half)]
        click.echo(f"Sampled {len(companies)} companies from {len(data['companies'])}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ.get("DOUBLEWORD_API_KEY"),
        base_url="https://api.doubleword.ai/v1",
    )

    # Build company list for the prompt
    company_names = [c["name"] for c in companies]
    names_sample = company_names[:30]  # Show first 30 as examples

    prompt = f"""I have a list of entities in the "{topic}" space. Generate search queries that would help find MORE entities in the same space that we might have missed.

Example entities we found:
{chr(10).join(f'- {name}' for name in names_sample)}
{"..." if len(company_names) > 30 else ""}

Generate 20-30 diverse search queries using patterns like:
- "competitors of [entity]"
- "alternatives to [entity]"
- "[entity] vs" (comparison articles often list competitors)
- "companies like [entity]"
- "top [entity type] companies 2024"
- "[industry] startups"

Pick entities that are likely to surface OTHER companies when searched (i.e., major players that appear in comparison articles).

Return one query per line. No numbering, no bullets."""

    click.echo(f"Generating expansion queries from {len(companies)} companies...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    content = response.choices[0].message.content.strip()
    queries = [line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 5]

    output_data = {
        "topic": topic,
        "queries": queries,
        "source": str(input_path),
        "stats": {
            "companies_sampled": len(companies),
            "queries_generated": len(queries),
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"Generated {len(queries)} expansion queries → {output_path}")


# -----------------------------------------------------------------------------
# search: Run queries through Serper
# -----------------------------------------------------------------------------

@main.command(name="search")
@click.option("--input", "-i", "input_file", default="queries.json", help="Queries file")
@click.option("--output", "-o", default="search_results.json", help="Output file")
@click.option("--results-per-query", "-n", default=30, help="Results per query")
def search_cmd(input_file: str, output: str, results_per_query: int):
    """Run search queries through Serper."""
    input_path = Path(input_file)
    output_path = Path(output)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    topic = data.get("topic")

    click.echo(f"Running {len(queries)} queries ({results_per_query} results each)")

    all_results = {"topic": topic, "searches": []}
    all_urls = set()

    for query in tqdm(queries, desc="Searching"):
        result = search(query, max_results=results_per_query)
        results = result.get("results", [])
        all_results["searches"].append({"query": query, "results": results})
        for r in results:
            if url := r.get("url"):
                all_urls.add(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    click.echo(f"Found {len(all_urls)} unique URLs → {output_path}")


# -----------------------------------------------------------------------------
# extract: Extract companies from search results (batch)
# -----------------------------------------------------------------------------

@main.command()
@click.option("--input", "-i", "input_file", default="search_results.json", help="Search results")
@click.option("--output", "-o", default="companies_raw.json", help="Output file")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
@click.option("--no-wait", is_flag=True, help="Submit batch without waiting")
def extract(input_file: str, output: str, model: str, no_wait: bool):
    """Extract companies from search results using batch LLM."""
    model = MODELS.get(model, model)
    input_path = Path(input_file)
    output_path = Path(output)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    topic = data.get("topic", "unknown")

    # Build extraction tasks
    tasks = []
    for s in data["searches"]:
        query = s["query"]
        for r in s["results"]:
            tasks.append({
                "query": query,
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "")[:20000],
            })

    click.echo(f"Extracting companies from {len(tasks)} pages")

    # Build batch requests
    requests_data = []
    for i, task in enumerate(tasks):
        prompt = f"""Extract company information from this search result.

Topic: {topic}
Query: {task['query']}
URL: {task['url']}
Title: {task['title']}
Content: {task['content']}

If this mentions companies relevant to "{topic}", extract them.
If it's not about relevant companies (news aggregator, job board, etc.), return empty list.

Return JSON:
{{
    "companies": [
        {{
            "name": "Company Name",
            "description": "Brief description",
            "products": ["product1", "product2"],
            "headquarters": "City, Country or null",
            "website": "website or null"
        }}
    ]
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

    # Save task mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_map = {f"extract-{i:04d}": task for i, task in enumerate(tasks)}
    task_map_path = output_path.parent / f".{output_path.stem}_tasks.json"
    with open(task_map_path, "w") as f:
        json.dump(task_map, f)

    # Submit batch
    batch_file = output_path.parent / f".{output_path.stem}_batch.jsonl"
    create_batch_file(requests_data, batch_file)

    client = get_client()
    file_id = upload_batch_file(client, batch_file)
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch submitted: {batch_id}")

    if no_wait:
        # Save batch info for later
        info = {"batch_id": batch_id, "output": str(output_path), "topic": topic}
        with open(output_path.parent / f".{output_path.stem}_pending.json", "w") as f:
            json.dump(info, f)
        click.echo(f"Run 'dataset-compilation extract-status {output}' to check progress")
        return

    # Wait and process
    click.echo("Waiting for batch...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        raise click.ClickException(f"Batch failed: {batch.status}")

    results_file = output_path.parent / f".{output_path.stem}_results.jsonl"
    download_results(client, batch.output_file_id, results_file)

    # Parse and aggregate
    results = parse_results(results_file)
    companies = []

    for custom_id, result in results.items():
        content = get_response_content(result)
        if not content:
            continue
        try:
            data = json.loads(content)
            task = task_map.get(custom_id, {})
            for company in data.get("companies", []):
                company["_source_url"] = task.get("url", "")
                companies.append(company)
        except json.JSONDecodeError:
            continue

    output_data = {"topic": topic, "companies": companies, "total": len(companies)}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"Extracted {len(companies)} company mentions → {output_path}")


# -----------------------------------------------------------------------------
# dedupe: Deduplicate company names using LLM (batch)
# -----------------------------------------------------------------------------

@main.command()
@click.option("--input", "-i", "input_file", required=True, help="Companies JSON file")
@click.option("--output", "-o", default=None, help="Output file (default: input_deduped.json)")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
@click.option("--no-wait", is_flag=True, help="Submit batch without waiting")
def dedupe(input_file: str, output: str, model: str, no_wait: bool):
    """Deduplicate company names using LLM.

    Run this twice: after extract (on raw mentions) and after filter (on validated list).
    The LLM clusters variants like "Cisco" / "Cisco Systems" / "Cisco Systems, Inc."
    """
    model = MODELS.get(model, model)
    input_path = Path(input_file)

    if not output:
        output = str(input_path.parent / f"{input_path.stem}_deduped.json")
    output_path = Path(output)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    companies = data.get("companies", [])
    topic = data.get("topic", "unknown")

    # Get unique names
    unique_names = list(set(c.get("name", "") for c in companies if c.get("name")))
    click.echo(f"Deduplicating {len(unique_names)} unique names from {len(companies)} mentions")

    if len(unique_names) <= 1:
        # Nothing to dedupe
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"Nothing to dedupe → {output_path}")
        return

    # Batch names into chunks
    chunk_size = 100
    chunks = [unique_names[i:i + chunk_size] for i in range(0, len(unique_names), chunk_size)]

    requests_data = []
    for i, chunk in enumerate(chunks):
        names_list = "\n".join(f"- {name}" for name in chunk)
        prompt = f"""These are company names in the "{topic}" space. Many are duplicates or variants.

Company names:
{names_list}

Group into clusters where each cluster is THE SAME company:
- "Cisco" and "Cisco Systems" → same
- "HPE" and "Hewlett Packard Enterprise" → same
- "Dell" and "Dell Technologies" → same

Pick the most common/canonical name for each cluster.

Return JSON:
{{
    "clusters": [
        {{"canonical": "Cisco", "variants": ["Cisco", "Cisco Systems", "Cisco Systems, Inc."]}}
    ]
}}

Only include clusters with 2+ names. Skip singletons."""

        requests_data.append({
            "custom_id": f"dedupe-{i:03d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        })

    # Submit batch
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_file = output_path.parent / f".{output_path.stem}_batch.jsonl"
    create_batch_file(requests_data, batch_file)

    client = get_client()
    file_id = upload_batch_file(client, batch_file)
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch submitted: {batch_id}")

    if no_wait:
        info = {"batch_id": batch_id, "input": str(input_path), "output": str(output_path), "topic": topic}
        with open(output_path.parent / f".{output_path.stem}_pending.json", "w") as f:
            json.dump(info, f)
        click.echo("Use --no-wait only if you want to check status manually")
        return

    # Wait and process
    click.echo("Waiting for batch...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        raise click.ClickException(f"Batch failed: {batch.status}")

    results_file = output_path.parent / f".{output_path.stem}_results.jsonl"
    download_results(client, batch.output_file_id, results_file)

    # Build name mapping
    results = parse_results(results_file)
    name_to_canonical = {}

    for custom_id, result in results.items():
        content = get_response_content(result)
        if not content:
            continue
        try:
            result_data = json.loads(content)
            for cluster in result_data.get("clusters", []):
                canonical = cluster.get("canonical", "")
                for variant in cluster.get("variants", []):
                    name_to_canonical[variant] = canonical
        except json.JSONDecodeError:
            continue

    click.echo(f"LLM identified {len(name_to_canonical)} variants to merge")

    # Group by canonical name
    by_canonical = {}
    for company in companies:
        name = company.get("name", "")
        if not name:
            continue

        canonical = name_to_canonical.get(name, name)
        if canonical not in by_canonical:
            by_canonical[canonical] = {
                "name": canonical,
                "variants": set(),
                "descriptions": [],
                "products": set(),
                "headquarters": None,
                "website": None,
                "mention_count": 0,
            }

        entry = by_canonical[canonical]
        entry["variants"].add(name)
        entry["mention_count"] += 1
        if desc := company.get("description"):
            entry["descriptions"].append(desc)
        if products := company.get("products"):
            if isinstance(products, list):
                entry["products"].update(products)
        if hq := company.get("headquarters"):
            entry["headquarters"] = hq
        if website := company.get("website"):
            entry["website"] = website

    # Build output
    deduped = []
    for canonical, entry in by_canonical.items():
        deduped.append({
            "name": entry["name"],
            "variants": list(entry["variants"]) if len(entry["variants"]) > 1 else [],
            "description": entry["descriptions"][0] if entry["descriptions"] else None,
            "products": list(entry["products"])[:10],
            "headquarters": entry["headquarters"],
            "website": entry["website"],
            "mention_count": entry["mention_count"],
        })

    deduped.sort(key=lambda x: -x["mention_count"])

    output_data = {"topic": topic, "companies": deduped, "total": len(deduped)}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"Deduplicated {len(companies)} → {len(deduped)} companies → {output_path}")


# -----------------------------------------------------------------------------
# filter: Validate companies via web search + LLM classification (batch)
# -----------------------------------------------------------------------------

@main.command(name="filter")
@click.option("--input", "-i", "input_file", default="companies.json", help="Companies file")
@click.option("--output", "-o", default="companies_filtered.json", help="Output file")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
@click.option("--search-results", "-n", default=5, help="Search results per company")
@click.option("--no-wait", is_flag=True, help="Submit batch without waiting")
def filter_cmd(input_file: str, output: str, model: str, search_results: int, no_wait: bool):
    """Validate companies via web search + LLM classification.

    For each company, runs a fresh web search and asks the LLM to classify
    whether it truly matches the topic (e.g., actually manufactures hardware
    vs just mentions it in marketing).
    """
    model = MODELS.get(model, model)
    input_path = Path(input_file)
    output_path = Path(output)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    companies = data.get("companies", [])
    topic = data.get("topic")

    if not topic:
        raise click.ClickException("No topic in input. Specify --topic or use file with topic field.")

    click.echo(f"Filtering {len(companies)} companies for: {topic}")

    # Run web searches
    click.echo(f"Searching ({search_results} results per company)...")
    contexts = []

    for company in tqdm(companies, desc="Searching"):
        name = company["name"]
        query = f'"{name}" {topic}'

        try:
            result = search(query, max_results=search_results)
            snippets = [
                f"- {r.get('title', '')}: {r.get('content', '')}"
                for r in result.get("results", [])
            ]
            context = "\n".join(snippets) if snippets else "No results found."
        except Exception as e:
            context = f"Search failed: {e}"

        contexts.append({"company": company, "context": context})

    # Build batch requests
    requests_data = []
    for i, ctx in enumerate(contexts):
        company = ctx["company"]
        prompt = f"""Does this company match "{topic}"?

Company: {company["name"]}

Web search results:
{ctx["context"]}

Return JSON:
{{
    "matches": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation",
    "products": ["list", "of", "products"] or null
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

    # Save mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)
    company_map = {f"filter-{i:04d}": ctx["company"] for i, ctx in enumerate(contexts)}
    map_path = output_path.parent / f".{output_path.stem}_map.json"
    with open(map_path, "w") as f:
        json.dump(company_map, f)

    # Submit batch
    batch_file = output_path.parent / f".{output_path.stem}_batch.jsonl"
    create_batch_file(requests_data, batch_file)

    client = get_client()
    file_id = upload_batch_file(client, batch_file)
    batch_id = create_batch(client, file_id)
    click.echo(f"Batch submitted: {batch_id}")

    if no_wait:
        info = {"batch_id": batch_id, "output": str(output_path), "topic": topic}
        with open(output_path.parent / f".{output_path.stem}_pending.json", "w") as f:
            json.dump(info, f)
        click.echo("Use --no-wait only if you want to check status manually")
        return

    # Wait and process
    click.echo("Waiting for batch...")
    batch = wait_for_batch(client, batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        raise click.ClickException(f"Batch failed: {batch.status}")

    results_file = output_path.parent / f".{output_path.stem}_results.jsonl"
    download_results(client, batch.output_file_id, results_file)

    # Parse and filter
    results = parse_results(results_file)
    included = []
    excluded = []

    for custom_id, result in results.items():
        company = company_map.get(custom_id, {})
        content = get_response_content(result)

        if not content:
            continue

        try:
            classification = json.loads(content)
            classification["_company"] = company

            if classification.get("matches") and classification.get("confidence") != "low":
                included.append(classification)
            else:
                excluded.append(classification)
        except json.JSONDecodeError:
            continue

    # Build output
    filtered = []
    for item in included:
        orig = item["_company"]
        filtered.append({
            "name": orig["name"],
            "description": orig.get("description"),
            "products": item.get("products") or orig.get("products", []),
            "headquarters": orig.get("headquarters"),
            "website": orig.get("website"),
            "mention_count": orig.get("mention_count", 1),
            "filter_reasoning": item.get("reasoning"),
        })

    filtered.sort(key=lambda x: -x.get("mention_count", 0))

    output_data = {
        "topic": topic,
        "companies": filtered,
        "total": len(filtered),
        "stats": {"input": len(companies), "included": len(included), "excluded": len(excluded)},
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Save excluded for review
    excluded_path = output_path.parent / f"{output_path.stem}_excluded.json"
    with open(excluded_path, "w") as f:
        json.dump(excluded, f, indent=2)

    click.echo(f"Filtered {len(companies)} → {len(filtered)} companies → {output_path}")
    click.echo(f"Excluded: {len(excluded)} (see {excluded_path})")


# -----------------------------------------------------------------------------
# merge: Merge multiple company files
# -----------------------------------------------------------------------------

@main.command()
@click.option("--inputs", "-i", required=True, multiple=True, help="Input files to merge (use multiple -i flags)")
@click.option("--output", "-o", default="companies_merged.json", help="Output file")
def merge(inputs: tuple, output: str):
    """Merge multiple company files into one.

    Use this to combine results from multiple passes (e.g., initial + expansion).
    Run dedupe afterwards to remove duplicates across the merged set.

    Example:
        dataset-compilation merge -i pass1.json -i pass2.json -o merged.json
        dataset-compilation dedupe -i merged.json -o final.json
    """
    output_path = Path(output)
    all_companies = []
    topic = None

    for input_file in inputs:
        input_path = Path(input_file)
        if not input_path.exists():
            raise click.ClickException(f"Input not found: {input_path}")

        with open(input_path) as f:
            data = json.load(f)

        companies = data.get("companies", [])
        all_companies.extend(companies)

        if not topic and data.get("topic"):
            topic = data["topic"]

        click.echo(f"  {input_path}: {len(companies)} companies")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "topic": topic,
        "companies": all_companies,
        "total": len(all_companies),
        "sources": list(inputs),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"Merged {len(all_companies)} companies → {output_path}")
    click.echo("Run dedupe to remove duplicates across merged files")


# -----------------------------------------------------------------------------
# validate: Check results against ground truth
# -----------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Normalize company name for matching."""
    name = name.lower().strip()
    for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", ltd", " ltd",
                   " corporation", " corp.", " corp", " technologies", " technology",
                   " networks", " systems"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return " ".join(name.replace(",", "").replace(".", "").replace("-", " ").split())


def matches(name: str, candidates: list[str], aliases: list[str] = None) -> bool:
    """Check if name matches any candidate."""
    norm = normalize_name(name)
    all_norms = [norm] + [normalize_name(a) for a in (aliases or [])]

    for candidate in candidates:
        norm_cand = normalize_name(candidate)
        for n in all_norms:
            if n == norm_cand or (len(n) >= 4 and len(norm_cand) >= 4 and (n in norm_cand or norm_cand in n)):
                return True
    return False


@main.command()
@click.option("--input", "-i", "input_file", default="companies_filtered.json", help="Companies file")
@click.option("--ground-truth", "-g", default="ground_truth.json", help="Ground truth JSON")
def validate(input_file: str, ground_truth: str):
    """Validate results against ground truth."""
    input_path = Path(input_file)
    gt_path = Path(ground_truth)

    if not input_path.exists():
        raise click.ClickException(f"Input not found: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    our_companies = [c["name"] for c in data["companies"]]

    if not gt_path.exists():
        click.echo(f"No ground truth file: {gt_path}")
        click.echo(f"Companies in dataset: {len(our_companies)}")
        return

    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_companies = gt_data["companies"]

    click.echo(f"=== Validation ===")
    click.echo(f"Our dataset: {len(our_companies)} companies")
    click.echo(f"Ground truth: {len(gt_companies)} companies")

    # Check recall
    found = []
    missed = []

    for gt in gt_companies:
        if matches(gt["name"], our_companies, gt.get("aliases")):
            found.append(gt["name"])
        else:
            missed.append(gt)

    recall = 100 * len(found) / len(gt_companies) if gt_companies else 0
    click.echo(f"\nRecall: {len(found)}/{len(gt_companies)} ({recall:.0f}%)")

    if missed:
        click.echo(f"\nMissed ({len(missed)}):")
        for m in missed:
            aliases = f" (aliases: {', '.join(m.get('aliases', []))})" if m.get('aliases') else ""
            click.echo(f"  - {m['name']}{aliases}")

    # Check for duplicates
    norm_names = {}
    for name in our_companies:
        norm = normalize_name(name)
        norm_names.setdefault(norm, []).append(name)

    duplicates = {k: v for k, v in norm_names.items() if len(v) > 1}
    if duplicates:
        click.echo(f"\nDuplicate clusters ({len(duplicates)}):")
        for norm, names in list(duplicates.items())[:10]:
            click.echo(f"  {names}")

    unique = len(norm_names)
    click.echo(f"\nUnique companies: {unique} (reported: {len(our_companies)})")


if __name__ == "__main__":
    main()
