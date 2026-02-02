"""
CLI for vulnerability scanning.

Scans code samples for security vulnerabilities using LLMs via batch or real-time APIs.
Supports both Doubleword (Qwen) and OpenAI (GPT-5) models.
"""

import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from .batch import (
    MODELS,
    DEFAULT_MODEL,
    resolve_model,
    is_openai_model,
    create_batch,
    create_batch_file,
    download_results,
    get_doubleword_client,
    get_openai_client,
    parse_results,
    upload_batch_file,
    wait_for_batch,
    count_tokens,
)
from .preprocess import preprocess_code


# Single prompt for vulnerability scanning
SCAN_PROMPT = """Review this code for security vulnerabilities.

Check for:
- Buffer overflows and memory corruption
- Injection vulnerabilities (SQL, command, XSS)
- Integer overflow/underflow
- Null pointer dereferences
- Input validation issues
- Authentication/authorization flaws

Code:
```
{code}
```

Respond with JSON: {{"has_bug": true/false, "bugs": [{{"type": "...", "description": "...", "line": N}}]}}"""


@click.group()
def cli():
    """Vulnerability scanner - detect security bugs in code at scale."""
    pass


@cli.command()
@click.option("--dataset", "-d", default="cvefixes",
              type=click.Choice(["cvefixes", "juliet"]),
              help="Dataset to load samples from")
@click.option("--max-samples", "-n", default=1000, type=int,
              help="Maximum samples to scan")
@click.option("--model", "-m", default=DEFAULT_MODEL,
              help=f"Model to use. Aliases: {', '.join(MODELS.keys())}")
@click.option("--output", "-o", default="results", help="Output directory")
@click.option("--window", "-w", default="24h", help="Batch completion window (1h or 24h)")
def scan(dataset: str, max_samples: int, model: str, output: str, window: str):
    """Scan code samples for vulnerabilities using batch API."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model(model)
    model_tag = model.replace("/", "_").replace(".", "_")

    # Load samples
    if dataset == "cvefixes":
        from .datasets import load_cvefixes
        db_path = Path("data/CVEfixes.db")
        if not db_path.exists():
            raise click.ClickException(
                f"CVEfixes database not found: {db_path}\n"
                "Run: bug-ensemble fetch-cvefixes to download"
            )
        samples = load_cvefixes(db_path, max_samples=max_samples, strip_comments=True)
    elif dataset == "juliet":
        from .juliet import load_juliet_samples
        juliet_path = Path("data/juliet")
        if not juliet_path.exists():
            raise click.ClickException(f"Juliet directory not found: {juliet_path}")
        samples = load_juliet_samples(juliet_path, max_per_cwe=max_samples // 5)
        for sample in samples:
            sample["code"] = preprocess_code(sample["code"], strip_comments=True)

    click.echo(f"Loaded {len(samples)} samples from {dataset}")
    buggy = sum(1 for s in samples if s["has_bug"])
    click.echo(f"  Vulnerable: {buggy}, Clean: {len(samples) - buggy}")

    # Create batch requests
    requests_list = []
    id_mapping = {}

    for i, sample in enumerate(samples):
        short_id = f"s{i:04d}"
        id_mapping[short_id] = sample["id"]

        requests_list.append({
            "custom_id": short_id,
            "model": resolved_model,
            "messages": [{"role": "user", "content": SCAN_PROMPT.format(code=sample["code"])}],
            "temperature": 0.0,
        })

    # Save samples for later analysis
    samples_path = output_dir / f"samples_{model_tag}.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Save ID mapping
    mapping_path = output_dir / f"id_mapping_{model_tag}.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    # Create batch file
    batch_path = output_dir / f"batch_input_{model_tag}.jsonl"
    create_batch_file(requests_list, batch_path)
    click.echo(f"Created batch file: {batch_path} ({len(requests_list)} requests)")

    # Determine provider
    if is_openai_model(model):
        client = get_openai_client()
        provider = "openai"
    else:
        client = get_doubleword_client()
        provider = "doubleword"

    click.echo(f"Using {provider} API with model {resolved_model}")

    # Upload and create batch
    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_path)
    click.echo(f"File ID: {file_id}")

    click.echo(f"Creating batch with {window} window...")
    batch_id = create_batch(client, file_id, window)
    click.echo(f"Batch ID: {batch_id}")

    # Save batch info
    batch_info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "model": resolved_model,
        "model_alias": model,
        "provider": provider,
        "samples_count": len(samples),
        "dataset": dataset,
    }
    batch_info_path = output_dir / f"batch_info_{model_tag}.json"
    with open(batch_info_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    click.echo(f"\nBatch submitted. Check status with:")
    click.echo(f"  bug-ensemble status -o {output} -m {model}")


@cli.command()
@click.option("--output", "-o", default="results", help="Output directory")
@click.option("--model", "-m", default=None, help="Model to check (checks all if not specified)")
@click.option("--wait/--no-wait", default=False, help="Wait for completion")
@click.option("--poll", "-p", default=30, help="Poll interval in seconds")
def status(output: str, model: str, wait: bool, poll: int):
    """Check batch status and optionally wait for completion."""
    output_dir = Path(output)

    # Find batch info files
    if model:
        model_tag = model.replace("/", "_").replace(".", "_")
        patterns = [f"batch_info_{model_tag}.json"]
    else:
        patterns = list(output_dir.glob("batch_info_*.json"))
        if not patterns:
            raise click.ClickException(f"No batch info files found in {output_dir}")

    for batch_info_path in (patterns if isinstance(patterns[0], Path) else [output_dir / p for p in patterns]):
        if not batch_info_path.exists():
            click.echo(f"Batch info not found: {batch_info_path}")
            continue

        with open(batch_info_path) as f:
            batch_info = json.load(f)

        batch_id = batch_info["batch_id"]
        provider = batch_info.get("provider", "doubleword")
        model_alias = batch_info.get("model_alias", batch_info["model"])

        if provider == "openai":
            client = get_openai_client()
        else:
            client = get_doubleword_client()

        click.echo(f"\n=== {model_alias} ({provider}) ===")
        click.echo(f"Batch ID: {batch_id}")

        if wait:
            batch = wait_for_batch(client, batch_id, poll)
        else:
            batch = client.batches.retrieve(batch_id)

        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0

        click.echo(f"Status: {batch.status}")
        click.echo(f"Progress: {completed}/{total}")

        if batch.status == "completed" and batch.output_file_id:
            model_tag = model_alias.replace("/", "_").replace(".", "_")
            results_path = output_dir / f"results_{model_tag}.jsonl"

            click.echo(f"Downloading results to {results_path}...")
            download_results(client, batch.output_file_id, results_path, provider)
            click.echo("Done!")


@cli.command()
@click.option("--dataset", "-d", default="cvefixes",
              type=click.Choice(["cvefixes", "juliet"]),
              help="Dataset to load samples from")
@click.option("--max-samples", "-n", default=200, type=int,
              help="Maximum samples to scan (default lower for real-time)")
@click.option("--model", "-m", default="gpt5-mini",
              help="Model to use (typically GPT-5 variants)")
@click.option("--output", "-o", default="results", help="Output directory")
@click.option("--concurrency", "-c", default=20, type=int, help="Concurrent requests")
def realtime(dataset: str, max_samples: int, model: str, output: str, concurrency: int):
    """Scan code using real-time API (for OpenAI comparison)."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model(model)
    model_tag = model.replace("/", "_").replace(".", "_")

    # Load samples
    if dataset == "cvefixes":
        from .datasets import load_cvefixes
        db_path = Path("data/CVEfixes.db")
        if not db_path.exists():
            raise click.ClickException(f"CVEfixes database not found: {db_path}")
        samples = load_cvefixes(db_path, max_samples=max_samples, strip_comments=True)
    elif dataset == "juliet":
        from .juliet import load_juliet_samples
        juliet_path = Path("data/juliet")
        if not juliet_path.exists():
            raise click.ClickException(f"Juliet directory not found: {juliet_path}")
        samples = load_juliet_samples(juliet_path, max_per_cwe=max_samples // 5)
        for sample in samples:
            sample["code"] = preprocess_code(sample["code"], strip_comments=True)

    click.echo(f"Loaded {len(samples)} samples from {dataset}")

    # Get client
    if is_openai_model(model):
        client = get_openai_client()
        provider = "openai"
    else:
        client = get_doubleword_client()
        provider = "doubleword"

    click.echo(f"Using {provider} real-time API with model {resolved_model}")
    click.echo(f"Concurrency: {concurrency}")

    # Process samples
    results = {}
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    def process_sample(idx_sample):
        idx, sample = idx_sample
        short_id = f"s{idx:04d}"
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": SCAN_PROMPT.format(code=sample["code"])}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return short_id, sample["id"], {
                "response": {
                    "body": {
                        "choices": [{"message": {"content": response.choices[0].message.content}}],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                    }
                }
            }
        except Exception as e:
            return short_id, sample["id"], {"error": str(e)}

    with click.progressbar(length=len(samples), label="Scanning") as bar:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_sample, (i, s)): i for i, s in enumerate(samples)}

            for future in as_completed(futures):
                short_id, full_id, result = future.result()
                results[short_id] = result

                if "error" in result:
                    errors += 1
                else:
                    usage = result["response"]["body"].get("usage", {})
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)

                bar.update(1)

    # Save results
    results_path = output_dir / f"results_{model_tag}.jsonl"
    with open(results_path, "w") as f:
        for custom_id, result in results.items():
            f.write(json.dumps({"custom_id": custom_id, **result}) + "\n")

    # Save samples and mapping
    samples_path = output_dir / f"samples_{model_tag}.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)

    id_mapping = {f"s{i:04d}": s["id"] for i, s in enumerate(samples)}
    mapping_path = output_dir / f"id_mapping_{model_tag}.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    click.echo(f"\nResults saved to {results_path}")
    click.echo(f"Total: {len(results)} samples, {errors} errors")
    click.echo(f"Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")


@cli.command()
@click.option("--output", "-o", default="results", help="Results directory")
@click.option("--models", "-m", default=None, help="Comma-separated model aliases to analyze (default: all)")
def analyze(output: str, models: str):
    """Analyze results and compare models."""
    output_dir = Path(output)

    # Find all result files
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        # Auto-detect from files
        result_files = list(output_dir.glob("results_*.jsonl"))
        model_list = [f.stem.replace("results_", "") for f in result_files]

    if not model_list:
        raise click.ClickException(f"No results found in {output_dir}")

    click.echo(f"Analyzing {len(model_list)} model(s): {', '.join(model_list)}")

    all_results = {}

    for model_tag in model_list:
        results_path = output_dir / f"results_{model_tag}.jsonl"
        samples_path = output_dir / f"samples_{model_tag}.json"
        mapping_path = output_dir / f"id_mapping_{model_tag}.json"

        if not results_path.exists():
            click.echo(f"  Skipping {model_tag}: results not found")
            continue

        # Load ground truth
        if samples_path.exists():
            with open(samples_path) as f:
                samples = json.load(f)
            ground_truth = {s["id"]: s["has_bug"] for s in samples}
        else:
            click.echo(f"  Warning: samples file not found for {model_tag}")
            ground_truth = {}

        # Load ID mapping
        if mapping_path.exists():
            with open(mapping_path) as f:
                id_mapping = json.load(f)
        else:
            id_mapping = {}

        # Parse results
        raw_results = parse_results(results_path)

        # Calculate metrics
        tp = fp = tn = fn = 0

        for custom_id, result in raw_results.items():
            full_id = id_mapping.get(custom_id, custom_id)
            actual = ground_truth.get(full_id)

            if actual is None:
                continue

            # Parse prediction
            try:
                content = result.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = json.loads(content)
                predicted = parsed.get("has_bug", False)
            except (json.JSONDecodeError, TypeError, KeyError):
                predicted = False

            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        tokens = count_tokens(raw_results)

        all_results[model_tag] = {
            "samples": total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
        }

    # Print comparison table
    click.echo("\n" + "=" * 80)
    click.echo("MODEL COMPARISON")
    click.echo("=" * 80)

    click.echo(f"\n{'Model':<15} {'Samples':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    click.echo("-" * 65)

    for model_tag, metrics in sorted(all_results.items(), key=lambda x: -x[1]["f1"]):
        click.echo(
            f"{model_tag:<15} {metrics['samples']:>8} "
            f"{metrics['precision']:>9.1%} {metrics['recall']:>9.1%} "
            f"{metrics['f1']:>9.1%} {metrics['accuracy']:>9.1%}"
        )

    click.echo("\n" + "-" * 65)
    click.echo(f"{'Model':<15} {'Input Tok':>12} {'Output Tok':>12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    click.echo("-" * 65)

    for model_tag, metrics in sorted(all_results.items(), key=lambda x: -x[1]["f1"]):
        click.echo(
            f"{model_tag:<15} {metrics['input_tokens']:>12,} {metrics['output_tokens']:>12,} "
            f"{metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6} {metrics['tn']:>6}"
        )

    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(all_results, f, indent=2)
    click.echo(f"\nAnalysis saved to {analysis_path}")


@cli.command()
@click.option("--output-dir", "-o", default="data", help="Directory to save database")
def fetch_cvefixes(output_dir: str):
    """Download CVEfixes database from Zenodo (~2GB)."""
    from .datasets import download_cvefixes, get_cvefixes_stats

    db_path = download_cvefixes(output_dir)

    click.echo("\n=== Database Statistics ===")
    stats = get_cvefixes_stats(str(db_path))
    click.echo(f"CVEs: {stats['cve_count']:,}")
    click.echo(f"Commits: {stats['commit_count']:,}")
    click.echo(f"Methods: {stats['method_count']:,}")

    click.echo("\nTop CWEs:")
    for cwe, count in stats['top_cwes'][:10]:
        click.echo(f"  {cwe}: {count:,}")


@cli.command()
def list_models():
    """List available model aliases."""
    click.echo("=== Available Models ===\n")
    click.echo(f"{'Alias':<12} {'Full Name':<45} {'Provider':<12}")
    click.echo("-" * 70)
    for alias, full_name in MODELS.items():
        provider = "OpenAI" if full_name.startswith("gpt-") else "Doubleword"
        click.echo(f"{alias:<12} {full_name:<45} {provider:<12}")
    click.echo(f"\nDefault: {DEFAULT_MODEL}")


# Keep ensemble commands for backward compatibility and calibration experiments
@cli.group()
def ensemble():
    """Ensemble commands for calibration experiments."""
    pass


@ensemble.command("prepare")
@click.option("--output", "-o", default="results/batch_input.jsonl", help="Output JSONL file")
@click.option("--model", default="30b", help="Model to use")
@click.option("--samples-file", default=None, help="JSON file with external samples")
@click.option("--prompt-subset", default="full",
              type=click.Choice(["security_only", "security_logic", "core", "expanded", "full"]),
              help="Prompt subset to use")
@click.option("--dataset", default="cvefixes",
              type=click.Choice(["juliet", "cvefixes"]),
              help="Dataset to load samples from")
@click.option("--max-samples", default=1000, type=int, help="Maximum samples")
def ensemble_prepare(output: str, model: str, samples_file: str, prompt_subset: str,
                     dataset: str, max_samples: int):
    """Prepare ensemble batch (multiple prompts per sample)."""
    from .prompts import REVIEW_PROMPTS, PROMPT_SUBSETS, format_prompt

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model(model)

    # Load samples
    if dataset == "cvefixes":
        from .datasets import load_cvefixes
        db_path = Path("data/CVEfixes.db")
        if not db_path.exists():
            raise click.ClickException(f"CVEfixes database not found: {db_path}")
        samples = load_cvefixes(db_path, max_samples=max_samples, strip_comments=True)
    elif dataset == "juliet":
        from .juliet import load_juliet_samples
        juliet_path = Path("data/juliet")
        samples = load_juliet_samples(juliet_path, max_per_cwe=max_samples // 5)
        for sample in samples:
            sample["code"] = preprocess_code(sample["code"], strip_comments=True)
    elif samples_file:
        with open(samples_file) as f:
            samples = json.load(f)

    click.echo(f"Loaded {len(samples)} samples")

    # Get prompts
    prompt_ids = PROMPT_SUBSETS[prompt_subset]
    prompt_list = [p for p in REVIEW_PROMPTS if p["id"] in prompt_ids]

    click.echo(f"Using {len(prompt_list)} prompts ({prompt_subset})")
    click.echo(f"Total requests: {len(samples) * len(prompt_list)}")

    # Create requests
    requests_list = []
    id_mapping = {}

    for i, sample in enumerate(samples):
        short_id = f"s{i:04d}"
        id_mapping[short_id] = sample["id"]

        for prompt in prompt_list:
            custom_id = f"{short_id}__{prompt['id']}"
            formatted_prompt = format_prompt(prompt, sample["code"])

            requests_list.append({
                "custom_id": custom_id,
                "model": resolved_model,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "temperature": 0.0,
            })

    # Save mapping
    mapping_path = output_path.parent / "id_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    create_batch_file(requests_list, output_path)
    click.echo(f"Wrote {len(requests_list)} requests to {output_path}")


# ============================================================================
# CWE Classification Commands
# ============================================================================

@cli.command()
@click.option("--max-per-cwe", "-n", default=None, type=int,
              help="Maximum samples per CWE class (default: all)")
@click.option("--model", "-m", default=DEFAULT_MODEL,
              help=f"Model to use. Aliases: {', '.join(MODELS.keys())}")
@click.option("--output", "-o", default="results/classify", help="Output directory")
@click.option("--window", "-w", default="24h", help="Batch completion window")
def classify(max_per_cwe: int | None, model: str, output: str, window: str):
    """Classify vulnerable code into CWE categories."""
    from .classify import (
        CWE_CLASSES,
        load_classification_samples,
        format_classification_prompt,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model(model)
    model_tag = model.replace("/", "_").replace(".", "_")

    # Load samples
    db_path = Path("data/CVEfixes.db")
    if not db_path.exists():
        raise click.ClickException(f"CVEfixes database not found: {db_path}")

    samples = load_classification_samples(db_path, max_per_cwe=max_per_cwe)
    click.echo(f"Loaded {len(samples)} samples across {len(CWE_CLASSES)} CWE classes")

    # Show distribution
    from collections import Counter
    cwe_dist = Counter(s["cwe"] for s in samples)
    for cwe, count in sorted(cwe_dist.items()):
        click.echo(f"  {cwe}: {count}")

    # Create batch requests
    requests_list = []
    id_mapping = {}

    for i, sample in enumerate(samples):
        short_id = f"c{i:04d}"
        id_mapping[short_id] = sample["id"]

        prompt = format_classification_prompt(sample["code"])
        requests_list.append({
            "custom_id": short_id,
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        })

    # Save samples for later analysis
    samples_path = output_dir / f"samples_{model_tag}.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Save ID mapping
    mapping_path = output_dir / f"id_mapping_{model_tag}.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    # Create batch file
    batch_path = output_dir / f"batch_input_{model_tag}.jsonl"
    create_batch_file(requests_list, batch_path)
    click.echo(f"Created batch file: {batch_path} ({len(requests_list)} requests)")

    # Determine provider
    if is_openai_model(model):
        client = get_openai_client()
        provider = "openai"
    else:
        client = get_doubleword_client()
        provider = "doubleword"

    click.echo(f"Using {provider} API with model {resolved_model}")

    # Upload and create batch
    click.echo("Uploading batch file...")
    file_id = upload_batch_file(client, batch_path)
    click.echo(f"File ID: {file_id}")

    click.echo(f"Creating batch with {window} window...")
    batch_id = create_batch(client, file_id, window)
    click.echo(f"Batch ID: {batch_id}")

    # Save batch info
    batch_info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "model": resolved_model,
        "model_alias": model,
        "provider": provider,
        "samples_count": len(samples),
        "task": "classification",
    }
    batch_info_path = output_dir / f"batch_info_{model_tag}.json"
    with open(batch_info_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    click.echo(f"\nBatch submitted. Check status with:")
    click.echo(f"  bug-ensemble status -o {output} -m {model}")


@cli.command("classify-realtime")
@click.option("--max-per-cwe", "-n", default=None, type=int,
              help="Maximum samples per CWE class (default: all)")
@click.option("--model", "-m", default="gpt5-mini",
              help="Model to use")
@click.option("--output", "-o", default="results/classify", help="Output directory")
@click.option("--concurrency", "-c", default=20, type=int, help="Concurrent requests")
def classify_realtime(max_per_cwe: int | None, model: str, output: str, concurrency: int):
    """Classify vulnerabilities using real-time API."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .classify import (
        CWE_CLASSES,
        load_classification_samples,
        format_classification_prompt,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model(model)
    model_tag = model.replace("/", "_").replace(".", "_")

    # Load samples
    db_path = Path("data/CVEfixes.db")
    if not db_path.exists():
        raise click.ClickException(f"CVEfixes database not found: {db_path}")

    samples = load_classification_samples(db_path, max_per_cwe=max_per_cwe)
    click.echo(f"Loaded {len(samples)} samples across {len(CWE_CLASSES)} CWE classes")

    # Get client
    if is_openai_model(model):
        client = get_openai_client()
        provider = "openai"
    else:
        client = get_doubleword_client()
        provider = "doubleword"

    click.echo(f"Using {provider} real-time API with model {resolved_model}")
    click.echo(f"Concurrency: {concurrency}")

    results = {}
    id_mapping = {}
    errors = 0

    def process_sample(idx_sample):
        idx, sample = idx_sample
        short_id = f"c{idx:04d}"
        prompt = format_classification_prompt(sample["code"])
        try:
            # GPT-5 models don't support temperature=0
            kwargs = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
            if not resolved_model.startswith("gpt-5"):
                kwargs["temperature"] = 0.0
            response = client.chat.completions.create(**kwargs)
            return short_id, sample["id"], {
                "response": {
                    "body": {
                        "choices": [{"message": {"content": response.choices[0].message.content}}],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                    }
                }
            }
        except Exception as e:
            return short_id, sample["id"], {"error": str(e)}

    with click.progressbar(length=len(samples), label="Classifying") as bar:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_sample, (i, s)): i for i, s in enumerate(samples)}

            for future in as_completed(futures):
                short_id, full_id, result = future.result()
                results[short_id] = result
                id_mapping[short_id] = full_id

                if "error" in result:
                    errors += 1

                bar.update(1)

    # Save results
    results_path = output_dir / f"results_{model_tag}.jsonl"
    with open(results_path, "w") as f:
        for custom_id, result in results.items():
            f.write(json.dumps({"custom_id": custom_id, **result}) + "\n")

    samples_path = output_dir / f"samples_{model_tag}.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)

    mapping_path = output_dir / f"id_mapping_{model_tag}.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)

    click.echo(f"\nResults saved to {results_path}")
    click.echo(f"Total: {len(results)} samples, {errors} errors")
    click.echo(f"\nAnalyze with: bug-ensemble classify-analyze -o {output}")


@cli.command("classify-analyze")
@click.option("--output", "-o", default="results/classify", help="Results directory")
@click.option("--model", "-m", default=None, help="Model to analyze (default: all)")
def classify_analyze(output: str, model: str):
    """Analyze CWE classification results."""
    from .classify import CWE_CLASSES, analyze_classification_results

    output_dir = Path(output)

    # Find result files
    if model:
        model_tag = model.replace("/", "_").replace(".", "_")
        model_tags = [model_tag]
    else:
        model_tags = [f.stem.replace("results_", "") for f in output_dir.glob("results_*.jsonl")]

    if not model_tags:
        raise click.ClickException(f"No results found in {output_dir}")

    for model_tag in model_tags:
        results_path = output_dir / f"results_{model_tag}.jsonl"
        samples_path = output_dir / f"samples_{model_tag}.json"
        mapping_path = output_dir / f"id_mapping_{model_tag}.json"

        if not results_path.exists():
            click.echo(f"Skipping {model_tag}: results not found")
            continue

        with open(samples_path) as f:
            samples = json.load(f)
        with open(mapping_path) as f:
            id_mapping = json.load(f)

        analysis = analyze_classification_results(results_path, samples, id_mapping)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"CWE CLASSIFICATION RESULTS: {model_tag}")
        click.echo(f"{'=' * 60}")

        click.echo(f"\nFine-grained (24 classes): {analysis['accuracy']:.1%} ({analysis['correct']}/{analysis['total']})")
        click.echo(f"Grouped (8 categories):    {analysis['grouped_accuracy']:.1%} ({analysis['grouped_correct']}/{analysis['total']})")

        # Per-group accuracy
        click.echo(f"\n{'Group':<18} {'Accuracy':>10} {'Support':>10}")
        click.echo("-" * 40)
        for group_name, metrics in sorted(analysis.get('per_group', {}).items(), key=lambda x: -x[1]['accuracy']):
            click.echo(f"{group_name:<18} {metrics['accuracy']:>9.1%} {metrics['support']:>10}")

        click.echo(f"\n{'CWE':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        click.echo("-" * 55)
        for cwe in CWE_CLASSES:
            m = analysis['per_class'].get(cwe, {})
            click.echo(f"{cwe:<12} {m.get('precision', 0):>9.1%} {m.get('recall', 0):>9.1%} "
                      f"{m.get('f1', 0):>9.1%} {m.get('support', 0):>10}")

        # Calibration by confidence
        click.echo(f"\nCalibration by Confidence:")
        for conf in ["high", "medium", "low", "unknown"]:
            cal = analysis['calibration'].get(conf, {})
            if cal:
                click.echo(f"  {conf}: {cal['accuracy']:.1%} accuracy ({cal['count']} samples)")

        # Confusion matrix summary
        click.echo(f"\nTop Confusions:")
        confusions = []
        for actual, preds in analysis['confusion'].items():
            for pred, count in preds.items():
                if pred != actual and pred != "NONE":
                    confusions.append((actual, pred, count))
        confusions.sort(key=lambda x: -x[2])
        for actual, pred, count in confusions[:5]:
            click.echo(f"  {actual} → {pred}: {count}")

        # Save analysis
        analysis_path = output_dir / f"analysis_{model_tag}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        click.echo(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    cli()
