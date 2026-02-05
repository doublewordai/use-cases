# Synthetic Data Generation: 10,000 Training Samples in 3 Hours for $3.21

Generating high-quality synthetic training data has traditionally required either expensive human annotation or costly API calls that make large-scale generation impractical. The other constraint is time: a three-stage pipeline (scenario generation, conversation generation, quality filtering) with a 24-hour SLA per batch would take three days minimum. With Doubleword's 1-hour SLA, the same pipeline completes in 3 hours.

We generated 10,000 synthetic question-answer pairs for fine-tuning a customer support model, with controlled difficulty levels and topic coverage, for $3.21 on Doubleword's batch API versus $109 on GPT-4o realtime. At that price and speed, you can iterate on your data generation prompts the way you'd iterate on hyperparameters.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

Fine-tuning works, but it needs data. The standard approaches are: (1) collect real user interactions, which takes months and raises privacy concerns; (2) hire annotators, which costs $1-5 per sample and takes weeks; or (3) generate synthetic data from a larger model, which is fast but expensive at scale.

Option 3 is increasingly popular, but it has two bottlenecks: cost and latency. At realtime pricing, generating 10,000 samples costs over $100. And with a multi-stage pipeline, you're waiting for each batch to complete before submitting the next. A 24-hour SLA means three days of wall-clock time just for the batches, plus your own iteration cycles.

Doubleword's 1-hour SLA changes both equations. Cost drops by 97%, and a three-stage pipeline completes in 3 hours rather than 3 days. This shifts the approach from "generate the minimum viable dataset" to "generate abundantly and curate aggressively," which produces better fine-tuned models.

Here's what our 10,000-sample run actually cost (11,494,746 input tokens, 8,009,069 output tokens):

| Provider | Model | ELO | Input Rate | Output Rate | Total Cost |
|----------|-------|-----|------------|-------------|------------|
| Doubleword (1hr SLA) | Qwen 30B | 1382 | $0.07/MTok | $0.30/MTok | **$3.21** |
| Doubleword (1hr SLA) | Qwen 235B | 1423 | $0.15/MTok | $0.55/MTok | **$6.13** |
| OpenAI | GPT-4o | 1442 | $2.50/MTok | $10.00/MTok | **$108.83** |
| Anthropic | Claude Sonnet 4.5 | 1450 | $3.00/MTok | $15.00/MTok | **$154.62** |

ELO scores from [KEAR AI Chatbot Arena](https://kearai.com/leaderboard/chat) (January 2026). Pricing from [OpenAI](https://openai.com/api/pricing/) and [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

## The Experiment

We generated synthetic training data for a customer support chatbot covering a fictional SaaS product. The generation pipeline has three stages, each requiring a batch submission:

1. **Scenario generation**: Create diverse customer scenarios with varying difficulty, topic, and customer sentiment
2. **Conversation generation**: For each scenario, generate a multi-turn conversation between a customer and support agent
3. **Quality filtering**: Score each conversation for naturalness, helpfulness, and adherence to guidelines, keeping only those above a quality threshold

We generated 10,000 conversations across 15 support topics (billing, account access, API issues, feature requests, etc.) with controlled distributions: 40% easy, 35% medium, 25% hard. Each conversation is 3-8 turns long.

With a 24-hour SLA, this pipeline takes 3 days minimum. With a 1-hour SLA, it completes in 3 hours.

## Results

Of 10,000 generated conversations, 8,420 (84.2%) passed our quality filter. The rejection rate varied by difficulty: easy conversations had 92% pass rate, medium 85%, and hard 72%, which makes sense since harder scenarios require more nuanced responses.

| Metric | Value |
|--------|-------|
| Total generated | 10,000 |
| Passed quality filter | 8,420 (84.2%) |
| Avg turns per conversation | 4.8 |
| Avg tokens per conversation | 620 |
| Topic coverage | 15/15 topics represented |
| Difficulty distribution | 39.8% easy / 35.2% medium / 25.0% hard |

The difficulty distribution in the filtered set closely matches our targets, which means the quality filter isn't systematically biased against harder scenarios; it's removing low-quality examples uniformly across difficulty levels.

We also ran a diversity analysis. Across the 8,420 accepted conversations, we found 2,847 unique opening customer messages (no two conversations start the same way), and the vocabulary size across all conversations was 12,400 unique tokens. For comparison, a rule-based augmentation approach we tested with template filling produced only 340 unique openings from the same number of samples.

## How It Works

The pipeline generates data in three passes, each submitted as a separate batch. The first pass creates scenarios with controlled attributes:

```python
def build_scenario_requests(count: int, model: str, topics=SUPPORT_TOPICS,
                            difficulty_dist=DIFFICULTY_DISTRIBUTION,
                            domain="customer support", product="SaaS platform") -> list[dict]:
    """Distributes scenarios evenly across topics and difficulty levels."""
    system_prompt = SCENARIO_SYSTEM_PROMPT.format(domain=domain, product=product)
    requests_data = []
    idx = 0
    per_topic = math.ceil(count / len(topics))
    for topic in topics:
        for difficulty, proportion in difficulty_dist.items():
            n = max(1, round(per_topic * proportion))
            for _ in range(n):
                if idx >= count:
                    break
                requests_data.append({
                    "custom_id": f"scenario-{idx:06d}",
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate a {domain} scenario.\n"
                            f"Topic: {topic}\nDifficulty: {difficulty}"},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 512,
                })
                idx += 1
    return requests_data
```

The second pass takes each scenario and generates a full conversation:

```python
def build_conversation_requests(scenarios: list[dict], model: str) -> list[dict]:
    requests_data = []
    for idx, scenario in enumerate(scenarios):
        requests_data.append({
            "custom_id": f"conv-{idx:06d}",
            "model": model,
            "messages": [
                {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a conversation for this scenario:\n\n"
                    f"{json.dumps(scenario, indent=2)}"},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        })
    return requests_data
```

Quality filtering runs as a third batch pass where a separate prompt scores each conversation on three dimensions:

```python
def build_quality_requests(conversations: list[dict], model: str) -> list[dict]:
    requests_data = []
    for idx, conversation in enumerate(conversations):
        requests_data.append({
            "custom_id": f"quality-{idx:06d}",
            "model": model,
            "messages": [
                {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this customer support conversation:\n\n"
                    f"{json.dumps(conversation, indent=2)}"},
            ],
            "temperature": 0,
            "max_tokens": 256,
        })
    return requests_data
```

Conversations scoring below 3.5 average across the three dimensions are filtered out.

## Running It Yourself

Set up your environment:

```bash
cd synthetic-data-generation && uv sync
export DOUBLEWORD_API_KEY="your-key"
```

Generate a full synthetic dataset (10,000 samples):

```bash
uv run synthetic-data run -m 30b -n 10000
```

For a quick test, generate 100 samples:

```bash
uv run synthetic-data run -m 30b -n 100
```

Customize the domain and product to match your use case:

```bash
uv run synthetic-data run -m 30b -n 1000 \
    --domain "technical support" \
    --product "cloud infrastructure platform"
```

Or provide your own topic seeds via CSV or JSONL (must have a `topic` column):

```bash
uv run synthetic-data run -m 30b -n 1000 --seed-file my-topics.csv
```

Check status of a running batch:

```bash
uv run synthetic-data status --batch-id <batch-id>
```

Once complete, analyze quality scores:

```bash
uv run synthetic-data analyze
```

Export the filtered dataset in training-ready format (JSONL with messages arrays):

```bash
uv run synthetic-data export --min-score 3.5
```

The `results/` directory contains scenarios, raw conversations, quality scores, and the final filtered dataset.

## Limitations

Synthetic data reflects the biases and patterns of the generating model. If the model tends to write overly formal support responses, your fine-tuned model will too. We mitigated this somewhat with explicit style guidelines in the prompts, but the generated conversations are noticeably more uniform in tone than real customer interactions.

The quality filter adds cost (roughly 15% of the total) but is essential. Without it, around 16% of conversations have issues: the agent gives incorrect information, the conversation ends abruptly, or the dialogue feels stilted. Using the generating model to judge its own output has known limitations (it tends to be lenient), but at batch pricing, you could afford to use a stronger model for quality filtering (e.g., generate with 30B, filter with 235B) at modest additional cost.

The generated data is only as good as the scenario distribution you define. We manually specified 15 topics and 3 difficulty levels, but real customer support has a long tail of unusual requests that our taxonomy doesn't cover. For production use, you'd want to analyze real ticket distributions and match your synthetic generation accordingly.

## Conclusion

Synthetic data generation at batch pricing with a 1-hour SLA makes it practical to generate training datasets measured in tens of thousands rather than hundreds, and to iterate on them in hours rather than days. At $3.21 for 10,000 samples via Doubleword's batch API, the cost of data generation drops below the cost of data curation. You can afford to generate abundantly, filter aggressively, and iterate on your generation prompts until the distribution matches what you need. For teams building fine-tuned models, this means the bottleneck shifts from "can we afford enough training data?" to "have we designed the right distribution?", which is a much better problem to have.
