# Migrate an agent from a flagship to an open-weight model

Token spend is currently a hot topic, and it is a popular reason to look at open-weight models, alongside the benefits of selecting models which fit different use cases. You might be weighing a full switch from a closed model, or blending open and closed across your workloads. Either way, a model needs to fit the work before you can rely on it and evaluations are the way to do this.

When considering switching a closed model to an open model, you can test by running an eval you have already run, or set up a new one that puts an open and a closed model side by side. Point it at a model on Doubleword and see how they compare.

Comparing models is three steps:
- Baseline your current model on real tasks.
- Replay those tasks against open-weight candidates on Doubleword's batch tier (or flex tier).
- Score the candidates.

Based on the results, you can shadow the winner on live traffic (this means you test it with real traffic to see how it would compare against the current model in production). If the open model performs the same or better than the closed model then make the switch and save!

Doubleword runs two of those steps cheaply. The batch tier can replay a whole test set for cents via an OpenAI-compatible (and Anthropic compatible) gateway which serves whichever model you pick. Kimi K3, DeepSeek, Qwen, GLM, and gpt-oss are all in the [catalog](https://docs.doubleword.ai/inference-api/models) with new models being added every week. One key gives you the best of open source. To evaluate several models of a similar class, you can run evals in parallel, and pick winners, or shortlist models that could substitute with a tweak (for example, some models have more jumpy agent behaviour 'verbosity' and so minor changes to the system prompt can accommodate the difference to your base model).

## Run it

Clone the example and run the eval with the Doubleword CLI. You need two keys, a Doubleword key for the open candidates and an OpenAI key for the closed baseline (see below).

```bash
dw examples clone agent-migration
cd agent-migration
dw project setup      # installs deps with uv
dw project run-all    # baseline, evaluate, score, shadow
```

`dw project info` lists every step.

## What you need

- A Doubleword API key for the open candidates.
- An OpenAI API key for the closed baseline.
- The Doubleword [`dw` CLI](https://github.com/doublewordai/dw) for batch runs.
- [Arize Phoenix](https://docs.doubleword.ai/inference-api/integrations/arize-phoenix), self-hosted, for tracing and scoring. Run it with `phoenix serve`.

Sign in to the [Doubleword Console](https://app.doubleword.ai), create a key under [API Keys](https://app.doubleword.ai/api-keys), and copy it, since it is only shown once.

![Doubleword console login](https://cdn.sanity.io/images/g1zo7y59/production/d6aa7c02234ff963c5bb87a6cc2826829f5e08b0-2940x1912.png)

![Generating a Doubleword API key in the Doubleword console](https://cdn.sanity.io/images/g1zo7y59/production/8e8d763da5e558756d57e5d74ebbe859f321a1d0-2940x1912.png)

```bash
export DOUBLEWORD_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Step 1: Baseline the flagship

Run your agent on its current model over a set of real tasks. Save the answers, the cost per task, and the latency. A new model has to match them.

Freeze those tasks into a golden set. Include the easy wins, the edge cases, and the multi-step jobs. Every candidate model will run against this same set. If you trace the agent with Phoenix, then you can pull the tasks straight from its logs. The workbook ships a 150-ticket support-triage set as a worked example, which you can switch out or amend to suit your use case. Each ticket is pre-labelled, so you can score answers without checking them by hand.

In the workbook the baseline runs a closed model on OpenAI in realtime:

```bash
dw project run baseline -- --model gpt-5.6-sol
```

## Step 2: Evaluate candidates offline

Run the golden set against each candidate on the batch tier, away from production. Prepare one batch file, then stamp each model onto its own copy with `dw files prepare --model`.

```bash
dw files prepare batches/batch.jsonl --model moonshotai/kimi-k3 --output-file batches/kimi.jsonl
dw batches run batches/kimi.jsonl --watch --output-id .kimi-id
dw batches results --from-file .kimi-id -o results/kimi.jsonl
dw batches analytics --from-file .kimi-id
```

Score each candidate two ways. Against the golden labels, for category accuracy, escalation accuracy, and valid JSON. Against the baseline answers, for how closely it matches, judged by a cheap Doubleword model in Phoenix. `response_format` pins the schema and Doubleword enforces it, so every candidate returns valid categories and valid JSON.

Here is a real run of the workbook's 150 tickets. GPT-5.6 Sol runs realtime, which is how you serve a closed flagship. Kimi K3 runs on Doubleword's batch tier, the right home for a high-volume background job like triage. Both run at low reasoning effort, since triage needs none. Intelligence scores are the [Artificial Analysis](https://artificialanalysis.ai/) index; the Kimi cost comes from `dw batches analytics`.

| Model | AA intelligence | Category accuracy | Escalation accuracy | Valid outputs | Cost, 150 tickets |
| --- | --- | --- | --- | --- | --- |
| GPT-5.6 Sol (closed, realtime) | 60.9 | 92.7% | 94.0% | 150/150 | $0.44 |
| Kimi K3 (open, batch) | 59.7 | 92.7% | 97.3% | 150/150 | $0.21 |

GPT-5.6 Sol and Kimi K3 sit about a point apart on the intelligence index, so this is a fair test. They tied on category accuracy at 92.7%, Kimi K3 edged escalation, and both returned valid JSON on every ticket. The open model held the quality and ran for $0.21 on Doubleword's batch tier against the flagship's $0.44 realtime, less than half the cost. Match the model to the job. You need the numbers to do that.

GPT-5.6 Sol runs realtime through OpenAI, standing for the closed flagship you would migrate from. Kimi K3 runs on Doubleword's batch tier, which is where the saving comes from for high-volume background work.

## Step 3: Shadow the pick

An offline score is strong. Shadow mode confirms it on live traffic before any user sees the new model. Serve the user the flagship's answer. In the background, send the same prompt to the candidate on the async (flex) tier, which costs less than realtime and starts within about a minute. Log both and compare them like you scored the golden set. You write the fan-out, and Doubleword serves both calls. The shadow call runs off the user's path, so they feel no delay.

## Make the call

Read the numbers. If a candidate matches your accuracy, stays inside your latency budget, and costs less, switch to it. Everything already runs through the Doubleword gateway on an OpenAI-compatible endpoint, so switching is a one-line model change. Keep the flagship one config value away in case you need to roll back.

Keep the harness. Re-run the golden set whenever a new model ships or you change a prompt. You get the same cost and quality check every time.

## Next steps

- **Run it on your own tasks.** Replace the sample dataset with your own and point the eval at your models.
- **Turn on prompt caching.** A repeated prefix makes [cached reads](https://docs.doubleword.ai/inference-api/prompt-caching) about 90% cheaper.
- **Right-size per step.** Send heavy reasoning to a bigger model and simple extraction to a smaller one, on one key.

Grab a [Doubleword API key](https://app.doubleword.ai) and point the workbook at your agent.
