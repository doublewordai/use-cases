# Model Evaluation at Scale: Qwen3-235B Matches GPT-5.2 at 5x Lower Cost

When model evaluation is cheap enough, you can run comprehensive benchmarks instead of spot-checking. We evaluated four models on the full GSM8K test set (1,319 questions) and found that Doubleword's Qwen3-235B matches GPT-5.2's 96.9% accuracy at \$0.21 versus \$1.06—the kind of comparison that becomes routine when batch pricing makes exhaustive testing economical.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

The standard approach to model selection is vibes: run a few examples, eyeball the outputs, pick what feels right. This works until it doesn't—you ship a model that fails on edge cases nobody tested, or you pay 5x more than necessary for equivalent performance.

The problem isn't that people don't want to run rigorous evaluations. It's that running GPT-5.2 on a full benchmark costs real money. At batch pricing (\$0.875/M input, \$7/M output), a 1,319-question GSM8K run costs \$1.06. That's not expensive in absolute terms, but it adds up when you're comparing five models across three benchmarks weekly. So teams don't, and model selection remains more art than science.

Batch inference changes this. At 50% off realtime pricing, comprehensive evaluation becomes a routine part of the development cycle rather than a quarterly event. We ran exactly this experiment to see what you learn when cost isn't the limiting factor.

## The Experiment

GSM8K (Grade School Math 8K) contains 8,500 word problems requiring 2-8 reasoning steps. We ran the full test split of 1,319 questions, enough to get statistically meaningful accuracy while keeping costs under \$2. Each problem looks something like:

> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

The model must reason through the steps and produce a numerical answer. We compared four models: OpenAI's GPT-5.2 and GPT-5-mini (via OpenAI's Batch API), and Qwen3-235B and Qwen3-30B (via Doubleword's batch API).

## Results

The flagship models have converged. GPT-5.2 leads with 96.9% accuracy, with Qwen3-235B just 0.9 percentage points behind—well within the range that prompt engineering or problem selection could close.

| Model | Accuracy | Cost (batch) |
|-------|----------|--------------|
| GPT-5.2 | 96.9% (1278/1319) | \$1.06 |
| Qwen3-235B | 96.0% (1266/1319) | \$0.21 |
| GPT-5-mini | 95.5% (1260/1319) | \$0.51 |
| Qwen3-30B | 94.7% (1249/1319) | \$0.08 |

These numbers align with published benchmarks. OpenAI reports GPT-5-mini at ~93% on GSM8K; we measured slightly higher, likely due to prompt engineering. All four models exceed 94% accuracy, confirming that modern LLMs have largely solved grade-school math.

The interesting finding isn't that big models are accurate—that's expected. It's that Qwen3-235B delivers equivalent accuracy to GPT-5.2 at one-fifth the cost. If you're running model evaluations at scale, this is the relevant comparison: 96.0% versus 96.9% accuracy, \$0.21 versus \$1.06.

## What the Models Got Wrong

The 53 questions that Qwen3-235B missed (versus 41 for GPT-5.2) reveal common failure modes. Looking at the errors, two patterns emerge.

First, multi-step problems with intermediate calculations that need to be tracked. The models occasionally drop a value or substitute the wrong number from an earlier step. This is a known weakness in chain-of-thought reasoning—working memory degrades over long inference chains.

Second, problems requiring unit conversions or rate calculations. "If a car travels 60 miles in 45 minutes, what's its speed in miles per hour?" The models sometimes fail to convert minutes to hours before dividing. These errors suggest that explicit unit handling in prompts might improve accuracy, though we didn't test this.

The failure sets overlap substantially across models, suggesting these are genuinely hard problems rather than model-specific blindspots. GPT-5.2's 12-question advantage over Qwen3-235B (41 vs 53 errors) represents the cost of the 5x savings.

## Cost Breakdown

Here's the actual token usage and costs from our full GSM8K run:

| Model | Input Tokens | Output Tokens | Input Cost | Output Cost | Total |
|-------|--------------|---------------|------------|-------------|-------|
| GPT-5.2 | 137,783 | 134,393 | \$0.12 | \$0.94 | **\$1.06** |
| GPT-5-mini | 137,783 | 486,642 | \$0.02 | \$0.49 | **\$0.51** |
| Qwen3-235B | 145,496 | 295,365 | \$0.03 | \$0.18 | **\$0.21** |
| Qwen3-30B | 145,496 | 332,913 | \$0.01 | \$0.07 | **\$0.08** |

Pricing: OpenAI Batch API at 50% off realtime ([source](https://platform.openai.com/docs/guides/batch)). Doubleword pricing at [doubleword.ai/pricing](https://doubleword.ai/pricing).

GPT-5-mini generates 3.6x more output tokens than GPT-5.2 for the same problems—it's more verbose in its reasoning. This explains why GPT-5-mini costs nearly half as much per token but ends up costing 48% of GPT-5.2 instead of the 14% you'd expect from the per-token pricing alone.

For equivalent accuracy, Qwen3-235B costs \$0.21 versus GPT-5.2's \$1.06—5x savings. Scale this to a proper evaluation suite and the numbers get interesting.

## Scaling to a Full Benchmark Suite

GSM8K is one benchmark. A rigorous evaluation pipeline runs multiple:

| Benchmark | Questions | What it measures |
|-----------|-----------|------------------|
| GSM8K | 1,319 | Math reasoning |
| MMLU | 14,042 | Broad knowledge |
| HumanEval | 164 | Code generation |
| MATH | 5,000 | Competition math |
| ARC-Challenge | 1,172 | Science reasoning |
| **Total** | **~22,000** | |

At our measured cost-per-question, running this full suite weekly for a year:

| Provider | Per Run | Annual (52 weeks) |
|----------|---------|-------------------|
| GPT-5.2 (OpenAI Batch) | ~\$18 | **~\$940** |
| Qwen3-235B (Doubleword Batch) | ~\$3.50 | **~\$180** |

That's **\$760/year** back in your pocket—enough to matter, not enough to compromise on evaluation rigor. The 5x cost advantage holds across benchmarks because it's driven by per-token pricing, not problem complexity.

## Running Your Own Evaluation

The evaluation runs in three steps: submit the batch, wait for completion, analyze results.

First, set up your environment and API keys:

```bash
cd model-evals && uv sync

export DOUBLEWORD_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # optional, for GPT models
```

Submit an evaluation batch. The `-n` flag limits the number of questions (e.g., `-n 100` for a quick test):

```bash
uv run model-evals run --model 235b
```

The `--model` flag accepts aliases (`235b`, `30b`, `gpt5.2`, `gpt5-mini`, `gpt5-nano`) or full model names. Doubleword models use 24-hour batch windows; OpenAI models use their Batch API with similar latency.

Check batch status while you wait:

```bash
uv run model-evals status
```

Once complete, analyze results:

```bash
uv run model-evals analyze
```

This produces accuracy metrics, error breakdowns, and cost summaries. The `results/` directory contains raw outputs for deeper analysis.

## Limitations

**Single benchmark.** GSM8K measures math reasoning; your use case may require different capabilities (code generation, multilingual, domain knowledge). The 5x cost ratio holds, but accuracy gaps may differ.

**Prompt sensitivity.** We used a simple prompt format. More sophisticated prompting (chain-of-thought, few-shot) could shift the accuracy rankings.

**Snapshot in time.** Model capabilities and pricing change. These results reflect February 2026; verify current pricing before production decisions.

## Notes

Batch inference trades latency for cost. The 24-hour processing window is ideal for evaluation pipelines, CI/CD, and any workflow where you're measuring rather than shipping. For interactive applications, realtime pricing applies.

The 5x cost ratio holds across benchmarks because it's driven by per-token pricing, not problem type. We ran GSM8K end-to-end; cost projections for the full suite use the same per-token rates.

## Conclusion

The flagship models have largely converged on GSM8K—GPT-5.2 achieves 96.9%, Qwen3-235B hits 96.0%, and even the smaller models exceed 94%. The differentiator is cost: Qwen3-235B delivers near-flagship performance at \$0.21 for the full benchmark versus \$1.06 for GPT-5.2.

For teams running regular model evaluations, the practical implication is clear: you can now afford to be thorough. A comprehensive benchmark suite (GSM8K, MMLU, HumanEval, MATH, ARC) that costs ~\$18/week on GPT-5.2 drops to ~\$3.50/week on Qwen3-235B. Over a year, that's ~\$940 versus ~\$180—enough savings to run evaluations on every model update, every prompt change, every dataset shift, while keeping \$760 in your budget for other things. The accuracy is there; the price is finally right.
