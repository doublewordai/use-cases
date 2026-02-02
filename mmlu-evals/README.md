# MMLU Evaluation at Scale

Evaluating language models across 57 subjects using the MMLU (Massive Multitask Language Understanding) benchmark via the Doubleword Batch API.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why MMLU

MMLU tests broad knowledge across STEM, humanities, social sciences, and professional domains. Unlike GSM8K (which tests math reasoning), MMLU measures what models "know" across 57 subjects from abstract algebra to world religions. It's become a standard benchmark for comparing model capabilities.

The full test set contains ~14,000 questions. At batch pricing, running the full evaluation costs around $3-6 depending on the model—cheap enough to compare multiple models comprehensively.

## Quick Start

```bash
cd mmlu-evals && uv sync

export DOUBLEWORD_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # optional, for GPT models

# Prepare a sample (100 questions across all subjects)
uv run mmlu-evals prepare -n 100

# Run evaluation
uv run mmlu-evals run --model 235b

# Score results
uv run mmlu-evals score -r data/results_*.jsonl
```

## Commands

### prepare

Download and prepare MMLU questions.

```bash
# 100 questions across all subjects
uv run mmlu-evals prepare -n 100

# Full dataset (~14K questions)
uv run mmlu-evals prepare

# Only STEM subjects
uv run mmlu-evals prepare --category STEM

# Specific subjects
uv run mmlu-evals prepare --subjects "abstract_algebra,anatomy,astronomy"
```

### list-subjects

See all available MMLU subjects organized by category.

```bash
uv run mmlu-evals list-subjects
```

### run

Run evaluation against a model.

```bash
# Batch mode (default, 50% cheaper)
uv run mmlu-evals run --model 235b

# Realtime mode (faster)
uv run mmlu-evals run --model gpt5.2 --realtime -c 20

# Available model aliases: 30b, 235b, gpt5-nano, gpt5-mini, gpt5.2
```

### score

Score results against ground truth.

```bash
uv run mmlu-evals score -r data/results_*.jsonl
```

## MMLU Categories

| Category | Subjects | Examples |
|----------|----------|----------|
| STEM | 19 | abstract_algebra, anatomy, astronomy, college_physics |
| Humanities | 13 | philosophy, world_religions, formal_logic, prehistory |
| Social Sciences | 12 | econometrics, sociology, psychology, public_relations |
| Other | 13 | business_ethics, clinical_knowledge, marketing, nutrition |

## Cost Estimates

Using batch pricing (50% off realtime):

| Model | Est. Cost (full MMLU) |
|-------|----------------------|
| GPT-5.2 | ~$6.00 |
| GPT-5-mini | ~$2.50 |
| Qwen3-235B | ~$1.20 |
| Qwen3-30B | ~$0.40 |

## Example Output

```
==================================================
MMLU EVALUATION RESULTS
==================================================
Overall Accuracy: 84.2%
Correct: 11,790/14,000

By Category:
  STEM: 81.5% (4,892/6,000)
  Humanities: 86.1% (3,444/4,000)
  Social Sciences: 85.8% (2,574/3,000)
  Other: 88.0% (880/1,000)

Tokens Used:
  Input: 2,450,000
  Output: 890,000
```
