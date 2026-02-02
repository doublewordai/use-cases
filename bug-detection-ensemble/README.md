# LLM Classification at Scale for $1

Classification is everywhere: categorizing support tickets, labeling documents, tagging content, sorting alerts by severity. It's often the first thing teams try to automate. The question is whether LLMs can do it well enough, and whether it's economical at scale.

We tested this on a hard problem—classifying security vulnerabilities by type—using 4,642 real vulnerabilities from CVEfixes. Qwen3-30B achieved 46.5% accuracy on grouped classification (Memory Safety, Pointer, Integer, etc.) at a cost of $0.40. Running twice and using agreement as a calibration signal pushes accuracy to 66% on the 58% of samples where both runs agree.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

At $0.40 per run, you can afford to try things. Fine-grained CWE classification doesn't work—20% accuracy isn't useful. Grouped classification does, at least for Memory Safety where the model hits 82%. Running twice gives you a confidence signal. These are things you'd want to know before committing to a classification approach, and batch inference lets you learn them for under a dollar.

## Results

We used CVEfixes, a dataset of functions from real vulnerability-fixing commits. Each function is labeled with a CWE type—4,642 samples across 24 CWEs.

### Fine-grained classification is hard

| Model | Accuracy (24 classes) | Cost |
|-------|----------------------|------|
| GPT-5.2 | 19.5% | $8.00 |
| Qwen3-30B | 19.2% | $0.40 |
| GPT-5-mini | 17.7% | $0.80 |
| Qwen3-235B | 16.2% | $1.20 |

Random baseline: 4.2%. All models are ~4-5x better than random, but 20% accuracy isn't useful for production. The models confuse similar CWEs—CWE-125 (out-of-bounds read) vs CWE-787 (out-of-bounds write) requires understanding whether the bug allows reading or writing, and they often get this wrong.

### Grouped classification works

Grouping into broader categories improves accuracy substantially:

| Model | Accuracy (8 groups) | Memory Safety | Pointer | Integer | Cost |
|-------|---------------------|---------------|---------|---------|------|
| **Qwen3-30B** | **46.5%** | **82.2%** | 34.9% | 14.3% | **$0.40** |
| Qwen3-235B | 38.6% | 59.5% | 33.2% | 10.7% | $1.20 |
| GPT-5-mini | 38.3% | 56.0% | 47.0% | 21.1% | $0.80 |
| GPT-5.2 | 35.2% | 44.6% | 41.5% | 27.7% | $8.00 |

Random baseline: 12.5%. Qwen3-30B hits 46.5%, driven by 82% accuracy on Memory Safety (buffer overflows, out-of-bounds access). Memory Safety is half the dataset, so this specialization pays off in the aggregate numbers.

### Which model to use

| Need | Model | Accuracy | Cost |
|------|-------|----------|------|
| Best value | Qwen3-30B | 46.5% grouped | $0.40 |
| Balanced across categories | GPT-5.2 | 35.2% grouped | $8.00 |

Qwen3-30B is best if your vulnerabilities are mostly memory safety issues—common in C/C++ codebases. GPT-5.2 is more balanced across categories but costs 20x more with lower overall accuracy.

## Calibration: Run Twice

At $0.40 for 4,600 samples, you can run twice and use agreement as a confidence signal.

| Agreement | Samples | Accuracy |
|-----------|---------|----------|
| Both runs agree | 58% | 66% |
| Runs disagree | 42% | Flag for review |

When both runs agree, accuracy jumps to 66%. When they disagree, flag for manual review. Two runs cost $0.80 total—still under a dollar for 4,600 samples with a calibration signal included.

## Scaling

| Volume | Qwen3-30B (1 run) | Qwen3-30B (2 runs) |
|--------|-------------------|---------------------|
| 1,000 samples | $0.09 | $0.18 |
| 10,000 samples | $0.86 | $1.72 |
| 100,000 samples | $8.60 | $17.20 |

## Error Analysis

Top confusions on fine-grained classification:

| Actual | Predicted As | Count |
|--------|--------------|-------|
| CWE-787 (OOB Write) | CWE-125 (OOB Read) | 370 |
| CWE-20 (Input Validation) | CWE-125 (OOB Read) | 362 |
| CWE-119 (Buffer Overflow) | CWE-125 (OOB Read) | 313 |

The model over-predicts CWE-125, the most common class in the dataset. These confusions all fall within Memory Safety, which is why grouped classification works better.

## Replication

```bash
cd bug-detection-ensemble
uv sync

# Download CVEfixes (~2GB SQLite database)
uv run bug-ensemble fetch-cvefixes

# Set API key
export DOUBLEWORD_API_KEY="your-key"

# Run classification
uv run bug-ensemble classify -m 30b -o results/run1
uv run bug-ensemble classify -m 30b -o results/run2

# Check batch status
uv run bug-ensemble status -o results/run1 --wait

# Analyze results
uv run bug-ensemble classify-analyze -o results/run1
```

### Available Models

| Alias | Model |
|-------|-------|
| `30b` | Qwen3-30B-A3B-Instruct |
| `235b` | Qwen3-235B-A22B-Instruct |

### Customizing Categories

Edit `src/classify.py` to change the groupings:

```python
CWE_GROUPS = {
    "Memory Safety": ["CWE-125", "CWE-787", "CWE-119", "CWE-120", "CWE-122"],
    "Pointer/Lifetime": ["CWE-476", "CWE-416", "CWE-415", "CWE-763"],
    "Integer": ["CWE-190"],
    "Resource": ["CWE-400", "CWE-401", "CWE-772"],
    "Input Validation": ["CWE-20", "CWE-22", "CWE-78"],
    "Concurrency": ["CWE-362"],
    "Control Flow": ["CWE-617", "CWE-835", "CWE-674"],
    "Other": ["CWE-59", "CWE-295", "CWE-269", "CWE-200"],
}
```

## Limitations

**Category imbalance.** Memory Safety is 49% of the dataset. Models that specialize in Memory Safety look better than balanced models.

**Label quality.** CVEfixes labels come from CVE metadata, not expert annotation.

**Generalization.** Results are specific to C/C++ vulnerabilities from open-source projects.

---

*Data: CVEfixes v1.0.7, 4,642 C/C++ functions across 24 CWE types. Costs use Doubleword batch pricing.*
