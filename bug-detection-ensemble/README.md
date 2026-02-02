# Vulnerability Classification at Scale

Classify security vulnerabilities by type using LLMs. Compare accuracy and cost across models, with ensemble calibration for confidence estimation.

## The Use Case

Security vulnerability classification is a natural fit for LLMs and batch APIs:
- **High volume**: Real codebases have thousands of functions to review
- **Clear task**: "What type of vulnerability is this?" is well-defined
- **High stakes**: Misclassification wastes triage time
- **Calibration matters**: Knowing when to trust predictions enables automation

This use case demonstrates classifying 600 real-world vulnerable functions from CVEfixes and comparing accuracy across Qwen and GPT-5 models.

## Key Results

| Model | Accuracy (3-way) | Cost (600 samples) |
|-------|------------------|-------------------|
| **Qwen3-235B** | **60.0%** | ~$0.30 |
| Qwen3-30B | 55.1% | ~$0.10 |
| GPT-5.2 | 39.2% | ~$4.00 |
| GPT-5-mini | 36.3% | ~$0.20 |

**Key findings:**
1. Qwen significantly outperforms GPT-5 (60% vs 39%) on security classification
2. Ensemble agreement predicts accuracy—when 4 models agree, accuracy jumps to 68.5%
3. 71% of samples can be auto-triaged based on model agreement
4. Two Qwen-235B runs cost 73% less than one GPT-5.2 run while achieving better accuracy

## The Dataset

**CVEfixes** contains 138,000+ functions extracted from real vulnerability-fixing commits. Each sample is labeled with CWE types. We selected 600 samples across 6 CWEs:

| CWE | Description | Category |
|-----|-------------|----------|
| CWE-119 | Buffer Overflow | Memory Safety |
| CWE-125 | Out-of-bounds Read | Memory Safety |
| CWE-787 | Out-of-bounds Write | Memory Safety |
| CWE-190 | Integer Overflow | Integer |
| CWE-416 | Use After Free | Pointer |
| CWE-476 | NULL Pointer Dereference | Pointer |

## Quick Start

```bash
cd bug-detection-ensemble
uv sync

# Download CVEfixes (~2GB)
uv run bug-ensemble fetch-cvefixes

# Set API keys
export DOUBLEWORD_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run classification (batch - cheaper)
uv run bug-ensemble classify -n 100 -m 235b -o results/classify_235b

# Or realtime (faster)
uv run bug-ensemble classify-realtime -n 100 -m gpt5.2 -o results/classify_gpt52

# Analyze results
uv run bug-ensemble classify-analyze -o results/classify_235b
```

## Available Models

| Alias | Model | Provider | Tier |
|-------|-------|----------|------|
| `30b` | Qwen3-VL-30B-A3B-Instruct | Doubleword | Mid-size |
| `235b` | Qwen3-VL-235B-A22B-Instruct | Doubleword | Flagship |
| `gpt5-nano` | GPT-5-nano | OpenAI | Budget |
| `gpt5-mini` | GPT-5-mini | OpenAI | Mid-tier |
| `gpt5.2` | GPT-5.2 | OpenAI | Flagship |

## Results

### Fine-grained Classification (6 classes)

| Model | Accuracy | Random Baseline |
|-------|----------|-----------------|
| Qwen3-235B | 26.3% | 16.7% |
| Qwen3-30B | 25.9% | 16.7% |
| GPT-5.2 | 25.2% | 16.7% |
| GPT-5-mini | 21.2% | 16.7% |

All models are ~1.5x better than random. Fine-grained CWE classification is hard.

### Grouped Classification (3 categories)

| Model | Accuracy | Memory Safety | Integer | Pointer |
|-------|----------|---------------|---------|---------|
| **Qwen3-235B** | **60.0%** | 75.0% | 11.0% | 62.0% |
| Qwen3-30B | 55.1% | 73.0% | 8.3% | 52.1% |
| GPT-5.2 | 39.2% | 35.7% | 18.0% | 55.0% |
| GPT-5-mini | 36.3% | 40.7% | 12.0% | 42.0% |
| *Random* | *33.3%* | - | - | - |

**Key findings:**
1. **Qwen significantly outperforms GPT-5** - 60% vs 39% on grouped classification
2. **Memory safety detection is strong** - Qwen achieves 75% accuracy on buffer overflow variants
3. **Integer overflow is hard for everyone** - 8-18% across all models

### Top Confusions

| Actual | Predicted As | Count |
|--------|--------------|-------|
| CWE-416 (Use After Free) | CWE-476 (NULL Deref) | 60 |
| CWE-190 (Integer Overflow) | CWE-476 (NULL Deref) | 42 |
| CWE-787 (OOB Write) | CWE-125 (OOB Read) | 40 |
| CWE-125 (OOB Read) | CWE-119 (Buffer Overflow) | 35 |

## Cheap Calibration: Run Twice for Confidence

Doubleword's batch pricing is low enough that you can run the same model multiple times for calibration—and still pay less than a single GPT-5.2 run.

### Results (Grouped 3-way Classification)

| Signal | Samples | Accuracy |
|--------|---------|----------|
| High confidence + runs agree | 417 (71%) | **61.4%** |
| High confidence + runs *disagree* | 21 (4%) | **23.8%** |
| Medium/low confidence | 148 (25%) | 55.4% |

When high-confidence runs disagree, accuracy drops from 61% to 24%. That's the calibration signal.

### Cost Comparison

| Approach | Cost | Accuracy |
|----------|------|----------|
| 2× Qwen-235B runs | **$0.28** | 61% (on agreed samples) |
| 1× GPT-5.2 run | $1.05 | 39% |

Two Qwen runs cost **73% less** than one GPT run, while achieving better accuracy and providing a calibration signal.

### Triage Workflow

| Confidence | Definition | Samples | Accuracy | Action |
|------------|------------|---------|----------|--------|
| High | Both runs agree + high confidence | 417 (71%) | 61.4% | Auto-classify |
| Review | Runs disagree OR low confidence | 169 (29%) | ~45% | Human review |

**71% of samples can be auto-triaged** with the double-run approach.

## Failure Analysis

### Example 1: Qwen Right, GPT Wrong

**CVE-2020-28097** - Linux VGA console scrollback buffer
**Actual:** CWE-125 (Out-of-bounds Read)

**Qwen-235B (correct):**
> "The code uses 'soff' as an index without validating it stays within bounds."

**GPT-5.2 (wrong):**
> "A negative computed scrollback offset can be used as an index." *(Predicted CWE-787 Out-of-bounds Write)*

Both identified the bounds issue, but GPT incorrectly classified it as a *write* vulnerability when the actual CVE is about *reading* beyond buffer boundaries.

### Example 2: Both Models Wrong

**CVE-2016-3841** - Linux kernel DCCP IPv6 use-after-free
**Actual:** CWE-416 (Use After Free) | **Both predicted:** CWE-476 (NULL Pointer)

The real vulnerability is that `dst` can be used after being freed in a race condition. Both models focused on potential NULL dereferences rather than the subtle lifetime/concurrency issue.

### Common Failure Modes

1. **NULL pointer bias** - Models over-predict CWE-476 because "check for NULL" is such a common code review pattern
2. **Read/Write confusion** - The code patterns for OOB reads and writes are nearly identical
3. **Lifetime issues are subtle** - Use-after-free bugs require reasoning about object lifetimes across function calls

## Limitations

1. **Label quality** - CVEfixes labels come from commit metadata, not expert annotation
2. **Related CWEs** - Some vulnerabilities legitimately belong to multiple categories
3. **Code context** - Snippets may be too short to fully determine vulnerability type
4. **Model versions** - Results may vary with model updates

---

*Data: CVEfixes v1.0.7, 600 C/C++ functions across 6 CWE types. Models: Qwen3-30B, Qwen3-235B (Doubleword), GPT-5-mini, GPT-5.2 (OpenAI). All costs shown use batch API pricing.*
