# Model Evaluation at Scale

Run large-scale model evaluations affordably. Thousands of test cases across multiple models to understand capabilities and track regression.

## The Idea

Thorough model evaluation requires many test cases. Batch makes it economical to:
- Run full benchmark suites (MMLU, HumanEval, GSM8K)
- Compare models head-to-head
- Find category-specific weaknesses
- Track performance over time

## Key Results

We evaluated multiple models on GSM8K (Grade School Math) - 500 questions requiring 2-8 reasoning steps.

| Model | Accuracy | Cost (batch) |
|-------|----------|--------------|
| GPT-5.2 | 96.4% | $0.40 |
| GPT-5-mini | 96.4% | $0.19 |
| **Qwen3-235B** | **96.2%** | **$0.08** |
| Qwen3-30B | 94.6% | $0.03 |

**Key finding:** Qwen3-235B matches GPT-5 flagship accuracy at 5x lower cost.

## Quick Start

```bash
cd model-evals && uv sync

export DOUBLEWORD_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run evaluation
uv run model-evals run --dataset gsm8k --model 235b -n 500

# Check status
uv run model-evals status

# Analyze results
uv run model-evals analyze
```

## Available Models

| Alias | Model | Provider |
|-------|-------|----------|
| `30b` | Qwen3-VL-30B-A3B-Instruct | Doubleword |
| `235b` | Qwen3-VL-235B-A22B-Instruct | Doubleword |
| `gpt5-nano` | GPT-5-nano | OpenAI |
| `gpt5-mini` | GPT-5-mini | OpenAI |
| `gpt5.2` | GPT-5.2 | OpenAI |

## Results

### GSM8K Benchmark (500 questions)

| Model | Accuracy | Correct | Provider |
|-------|----------|---------|----------|
| GPT-5.2 | 96.4% | 482/500 | OpenAI |
| GPT-5-mini | 96.4% | 482/500 | OpenAI |
| **Qwen3-235B** | **96.2%** | 481/500 | Doubleword |
| Qwen3-30B | 94.6% | 473/500 | Doubleword |

### Token Usage

| Model | Input Tokens | Output Tokens |
|-------|--------------|---------------|
| GPT-5.2 | 51,900 | 50,113 |
| GPT-5-mini | 51,900 | 180,316 |
| Qwen3-235B | 54,828 | 111,155 |
| Qwen3-30B | 54,828 | 124,629 |

### Baseline Comparison

| Model | Published GSM8K Accuracy | Source |
|-------|--------------------------|--------|
| GPT-5.2 | ~95%+ | Vellum |
| Claude Opus 4.5 | ~95%+ | DataStudios |
| GPT-5-mini | ~93% (published) | OpenAI |

Our results align with published benchmarks.

## Cost Comparison

### Model Pricing (per 1M tokens, batch)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | GPT-5.2 | $0.875 | $7.00 |
| OpenAI | GPT-5-mini | $0.125 | $1.00 |
| Doubleword | Qwen3-235B | $0.20 | $0.60 |
| Doubleword | Qwen3-30B | $0.05 | $0.20 |

### Actual Cost for This Evaluation

| Provider | Model | This Eval (500q) | Full GSM8K (1,319q) |
|----------|-------|------------------|---------------------|
| OpenAI | GPT-5.2 | $0.40 | $1.05 |
| OpenAI | GPT-5-mini | $0.19 | $0.50 |
| **Doubleword** | **Qwen3-235B** | **$0.08** | **$0.21** |
| Doubleword | Qwen3-30B | $0.03 | $0.08 |

## Key Findings

1. **Flagship models are tied on accuracy**: GPT-5.2, GPT-5-mini, and Qwen3-235B all achieve ~96% on GSM8K. The 0.2% difference is within noise.

2. **Qwen3-235B matches GPT-5 at 5x lower cost**:
   - GPT-5.2 batch: $0.40 for 96.4% accuracy
   - Qwen3-235B batch: $0.08 for 96.2% accuracy

3. **GPT-5-mini is surprisingly strong**: Matches the flagship GPT-5.2 on this benchmark while being 7x cheaper.

4. **Qwen3-30B offers best value for cost-sensitive workloads**: At $0.03 per 500 questions, it achieves 94.6% accuracy—only 1.8 points below the flagships at 13x lower cost.

## When to Use What

| Use Case | Recommendation | Cost (500q) |
|----------|----------------|-------------|
| Maximum accuracy | GPT-5.2, GPT-5-mini, or Qwen3-235B | $0.08–$0.40 |
| Best value at flagship tier | Qwen3-235B (24h batch) | $0.08 |
| Cost-sensitive bulk evaluation | Qwen3-30B (24h batch) | $0.03 |
| Quick iteration | GPT-5-mini (realtime) | $0.37 |

## Conclusion

The flagship models have converged on GSM8K—GPT-5.2, GPT-5-mini, and Qwen3-235B all hit ~96% accuracy. The differentiator is cost:

- **Qwen3-235B** delivers flagship-tier accuracy at $0.08/500 questions (batch), making it **5x cheaper than GPT-5.2** and **2.4x cheaper than GPT-5-mini**.
- **Qwen3-30B** at $0.03/500 questions offers 94.6% accuracy for workloads where the 1.8 point gap doesn't matter.

**Bottom line**: For model evaluation workloads, Doubleword's batch pricing makes comprehensive testing economically viable. A full GSM8K benchmark costs $0.21 with Qwen3-235B versus $1.05 with GPT-5.2—same accuracy, 5x savings.
