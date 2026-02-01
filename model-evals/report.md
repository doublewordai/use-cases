# Model Evaluation Report

## What we did

Evaluated multiple models on the GSM8K benchmark (Grade School Math) to compare accuracy and cost across providers. GSM8K tests mathematical reasoning with multi-step word problems requiring 2-8 reasoning steps.

## Data

- **Dataset**: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (test split)
- **Questions evaluated**: 500 (of 1,319 total)
- **Task**: Solve grade-school math word problems with step-by-step reasoning
- **Source**: HuggingFace `openai/gsm8k`

## Results

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
| GPT-5.2 | ~95%+ | [Vellum](https://www.vellum.ai/blog/gpt-5-2-benchmarks) |
| Claude Opus 4.5 | ~95%+ | [DataStudios](https://www.datastudios.org/post/claude-opus-4-5-vs-chatgpt-5-1-full-report-and-comparison-of-models-features-performance-pricin) |
| GPT-5-mini | ~93% (published) | [OpenAI](https://openai.com/index/introducing-gpt-5-for-developers/) |

Our results align with published benchmarks. The flagship models (GPT-5.2, GPT-5-mini, Qwen3-235B) all achieve ~96% accuracy on this subset.

## Cost Comparison

### Model Pricing (per 1M tokens)

| Provider | Model | Input | Output | Batch Discount | Source |
|----------|-------|-------|--------|----------------|--------|
| OpenAI | GPT-5.2 | $1.75 | $14.00 | 50% | [OpenAI](https://platform.openai.com/docs/pricing) |
| OpenAI | GPT-5-mini | $0.25 | $2.00 | 50% | [OpenAI](https://platform.openai.com/docs/pricing) |
| Anthropic | Claude Sonnet 4 | $3.00 | $15.00 | 50% | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 | 50% | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |
| Doubleword | Qwen3-235B (realtime) | $0.80 | $2.40 | — | [Doubleword](https://docs.doubleword.ai/batches/model-pricing) |
| Doubleword | Qwen3-235B (24h batch) | $0.20 | $0.60 | — | [Doubleword](https://docs.doubleword.ai/batches/model-pricing) |
| Doubleword | Qwen3-30B (realtime) | $0.16 | $0.80 | — | [Doubleword](https://docs.doubleword.ai/batches/model-pricing) |
| Doubleword | Qwen3-30B (24h batch) | $0.05 | $0.20 | — | [Doubleword](https://docs.doubleword.ai/batches/model-pricing) |

### Actual Cost for This Evaluation (500 questions)

| Model | Mode | Input Tokens | Output Tokens | Cost |
|-------|------|--------------|---------------|------|
| GPT-5.2 | Realtime | 51,900 | 50,113 | **$0.79** |
| GPT-5.2 | Batch (50% off) | 51,900 | 50,113 | **$0.40** |
| GPT-5-mini | Realtime | 51,900 | 180,316 | $0.37 |
| GPT-5-mini | Batch (50% off) | 51,900 | 180,316 | **$0.19** |
| Qwen3-235B | Realtime | 54,828 | 111,155 | $0.31 |
| Qwen3-235B | 24h Batch | 54,828 | 111,155 | **$0.08** |
| Qwen3-30B | Realtime | 54,828 | 124,629 | $0.11 |
| Qwen3-30B | 24h Batch | 54,828 | 124,629 | **$0.03** |

### Cost Comparison Summary (Batch Mode)

| Provider | Model | Accuracy | This Eval (500q) | Full GSM8K (1,319q) |
|----------|-------|----------|------------------|---------------------|
| OpenAI | GPT-5.2 | 96.4% | $0.40 | $1.05 |
| OpenAI | GPT-5-mini | 96.4% | $0.19 | $0.50 |
| **Doubleword** | **Qwen3-235B** | **96.2%** | **$0.08** | **$0.21** |
| Doubleword | Qwen3-30B | 94.6% | $0.03 | $0.08 |

## Key Findings

1. **Flagship models are tied on accuracy**: GPT-5.2, GPT-5-mini, and Qwen3-235B all achieve ~96% on GSM8K. The 0.2% difference between them is within noise.

2. **Qwen3-235B matches GPT-5 at 5x lower cost**:
   - GPT-5.2 batch: $0.40 for 96.4% accuracy
   - Qwen3-235B batch: $0.08 for 96.2% accuracy
   - Same accuracy tier, 5x cheaper

3. **GPT-5-mini is surprisingly strong**: Matches the flagship GPT-5.2 on this benchmark while being 7x cheaper ($0.19 vs $0.40 in batch mode).

4. **Qwen3-30B offers best value for cost-sensitive workloads**: At $0.03 per 500 questions, it achieves 94.6% accuracy—only 1.8 points below the flagship models at 13x lower cost.

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
