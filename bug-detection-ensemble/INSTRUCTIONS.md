# Vulnerability Scanning at Scale

Scan a codebase for security vulnerabilities using LLMs. Compare accuracy and cost across models.

## The Use Case

Security vulnerability detection is a natural fit for LLMs and batch APIs:
- **High volume**: Real codebases have thousands of functions to review
- **Clear task**: "Does this code have a security bug?" is well-defined
- **High stakes**: Missed vulnerabilities are costly
- **Human review is expensive**: LLM pre-screening can prioritize human attention

This use case demonstrates scanning 1,000 real-world functions from CVEfixes and comparing detection accuracy across Qwen and GPT-5 models.

## The Dataset

**CVEfixes** contains 138,000+ functions extracted from real vulnerability-fixing commits. Each entry pairs:
- **Vulnerable code** (before the fix)
- **Patched code** (after the fix)

Labeled with CVE IDs and CWE types. This is messy real-world data from Linux kernel to hobby projects.

## Quick Start

```bash
cd bug-detection-ensemble
uv sync

# Download CVEfixes (~2GB)
uv run bug-ensemble fetch-cvefixes

# Scan with Qwen3-30B (Doubleword batch)
export DOUBLEWORD_API_KEY="your-key"
uv run bug-ensemble scan -m 30b -n 1000

# Check status and download results
uv run bug-ensemble status --wait

# Scan with GPT-5-mini (OpenAI real-time)
export OPENAI_API_KEY="your-key"
uv run bug-ensemble realtime -m gpt5-mini -n 1000

# Compare all models
uv run bug-ensemble analyze
```

## Available Models

| Alias | Model | Provider | Tier |
|-------|-------|----------|------|
| `30b` | Qwen3-VL-30B-A3B-Instruct | Doubleword | Mid-size |
| `235b` | Qwen3-VL-235B-A22B-Instruct | Doubleword | Flagship |
| `gpt5-nano` | GPT-5-nano | OpenAI | Budget |
| `gpt5-mini` | GPT-5-mini | OpenAI | Mid-tier |
| `gpt5.2` | GPT-5.2 | OpenAI | Flagship |

## Commands

### `scan` - Batch vulnerability scanning

```bash
# Scan 1000 samples with Qwen3-235B
uv run bug-ensemble scan -m 235b -n 1000

# Scan with GPT-5.2 (also supports batch)
uv run bug-ensemble scan -m gpt5.2 -n 1000
```

### `realtime` - Real-time API scanning

```bash
# Faster but more expensive
uv run bug-ensemble realtime -m gpt5-mini -n 500 -c 50
```

### `analyze` - Compare results

```bash
# Compare all models
uv run bug-ensemble analyze

# Compare specific models
uv run bug-ensemble analyze -m 30b,235b,gpt5-mini
```

## What Success Looks Like

A comparison table showing:

| Model | Precision | Recall | F1 | Cost/1K samples |
|-------|-----------|--------|----|----|
| Qwen3-235B | 85% | 82% | 83% | $X |
| Qwen3-30B | 80% | 78% | 79% | $Y |
| GPT-5-mini | 82% | 80% | 81% | $Z |

The key insight: Which model offers the best accuracy-per-dollar for security scanning?

## Ensemble Mode (Advanced)

For calibrated confidence scores, run multiple prompts per sample:

```bash
# 20 prompts × 1000 samples = 20,000 requests
uv run bug-ensemble ensemble prepare --prompt-subset full -n 1000
uv run bug-ensemble scan -i results/batch_input.jsonl
```

Vote agreement predicts accuracy: unanimous votes are trustworthy, split votes need human review.

## See Also

- [report.md](./report.md) - Full results and analysis
- [RUBRIC.md](../RUBRIC.md) - Evaluation criteria
