# Classifying 4,600 Security Vulnerabilities for $1

LLMs can classify security vulnerabilities by type, but accuracy varies widely depending on how you frame the task. This walkthrough shows how to run classification at scale using batch APIs, what accuracy to expect, and how running twice gives you a useful calibration signal.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## The Task

Given a C/C++ function with a known vulnerability, classify what type it is. We used CVEfixes, a dataset of 138,000+ functions extracted from real vulnerability-fixing commits. Each function is labeled with a CWE (Common Weakness Enumeration) type.

We tested two setups:
- **Fine-grained**: 24 CWE classes (e.g., CWE-125 Out-of-bounds Read vs CWE-787 Out-of-bounds Write)
- **Grouped**: 8 broader categories (e.g., Memory Safety, Pointer/Lifetime, Input Validation)

## Results

On 4,642 samples:

| Granularity | Best Accuracy | Random Baseline |
|-------------|---------------|-----------------|
| 24 classes | 19.5% | 4.2% |
| 8 categories | 46.5% | 12.5% |

Fine-grained CWE classification is hard—even the best model is only ~5x better than random. Grouping into broader categories helps significantly, reaching 46% accuracy (3.7x random).

### Per-Category Accuracy (Qwen-30B)

| Category | Accuracy | Samples |
|----------|----------|---------|
| Memory Safety | 82.2% | 2,287 |
| Pointer/Lifetime | 34.9% | 602 |
| Integer | 14.3% | 336 |
| Input Validation | 3.1% | 655 |
| Other categories | <10% | 762 |

The model excels at Memory Safety (buffer overflows, out-of-bounds access) but struggles with other categories. This reflects both the training data distribution and the inherent difficulty—distinguishing a race condition from an integer overflow requires different reasoning than spotting buffer issues.

## Calibration: Run Twice

When inference is cheap, you can run the same model twice and use agreement as a confidence signal.

| Agreement | Samples | Accuracy |
|-----------|---------|----------|
| Both runs agree | 58% | 66% |
| Runs disagree | 42% | — |

When both runs agree, accuracy is higher. When they disagree, flag for human review. This gives you a simple triage workflow without any additional infrastructure.

## Cost

| Model | 4,642 samples | Per sample |
|-------|---------------|------------|
| Qwen3-30B | $0.40 | $0.00009 |
| Qwen3-235B | $1.20 | $0.00026 |

Running twice for calibration doubles the cost but still keeps you under $1 for the smaller model. At these prices, you can iterate freely—try different prompts, test grouping strategies, or run multiple models.

## What Works and What Doesn't

**Works well:**
- Memory safety vulnerabilities (buffer overflows, out-of-bounds access): 80%+ accuracy
- Coarse categorization (8 groups): ~45% accuracy
- Agreement-based calibration for flagging uncertain predictions

**Doesn't work well:**
- Fine-grained CWE classification (24 classes): ~20% accuracy
- Categories with subtle distinctions (race conditions, integer issues): <15%
- Small categories with few training examples

**The pattern:** LLMs are good at recognizing broad vulnerability patterns but struggle to distinguish similar-looking issues. CWE-125 (out-of-bounds read) vs CWE-787 (out-of-bounds write) requires understanding whether the bug allows reading or writing—a distinction the model often misses.

## Replication

```bash
cd bug-detection-ensemble
uv sync

# Download CVEfixes (~2GB SQLite database)
uv run bug-ensemble fetch-cvefixes

# Set API key
export DOUBLEWORD_API_KEY="your-key"

# Run classification (batch mode)
uv run bug-ensemble classify -m 30b -o results/run1
uv run bug-ensemble classify -m 30b -o results/run2  # second run for calibration

# Check batch status
uv run bug-ensemble status -o results/run1 --wait

# Analyze results
uv run bug-ensemble classify-analyze -o results/run1
```

### Available Models

| Alias | Model | Provider |
|-------|-------|----------|
| `30b` | Qwen3-30B-A3B-Instruct | Doubleword |
| `235b` | Qwen3-235B-A22B-Instruct | Doubleword |

### Customizing the Task

The CWE classes and groupings are defined in `src/classify.py`. To test different categorizations:

```python
# In src/classify.py
CWE_CLASSES = {
    "CWE-125": "Out-of-bounds Read",
    "CWE-787": "Out-of-bounds Write",
    # Add or remove CWEs as needed
}

CWE_GROUPS = {
    "Memory Safety": ["CWE-125", "CWE-787", "CWE-119"],
    # Define your own groupings
}
```

## Dataset Details

CVEfixes contains functions from real vulnerability-fixing commits. We filtered to:
- C/C++ code only
- 100-3000 characters (enough context, fits in context window)
- 24 CWE types with 25+ samples each

| Category | CWEs | Samples |
|----------|------|---------|
| Memory Safety | CWE-119, CWE-120, CWE-122, CWE-125, CWE-787 | 2,287 |
| Pointer/Lifetime | CWE-415, CWE-416, CWE-476, CWE-763 | 602 |
| Input Validation | CWE-20, CWE-22, CWE-78 | 655 |
| Integer | CWE-190 | 336 |
| Resource | CWE-400, CWE-401, CWE-772 | 207 |
| Concurrency | CWE-362 | 164 |
| Control Flow | CWE-617, CWE-674, CWE-835 | 127 |
| Other | CWE-59, CWE-200, CWE-269, CWE-295 | 264 |

## Limitations

**Label quality.** CVEfixes labels come from CVE metadata, not expert annotation. Some vulnerabilities span multiple CWE categories.

**Code context.** Extracted snippets may lack context needed for accurate classification—the vulnerability type sometimes depends on how the function is called.

**Category imbalance.** Memory Safety dominates the dataset (49% of samples), which biases overall accuracy numbers.

---

*Data: CVEfixes v1.0.7, 4,642 C/C++ functions across 24 CWE types.*
