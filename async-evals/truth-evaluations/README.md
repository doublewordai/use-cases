# LLM-as-Judge Evaluations Observed in Arize Phoenix - 817 Answers and Judgements for $0.50

**Judge every output with a frontier model on batch inference, and track the
scores in [Arize](https://arize.com) Phoenix. comprehensive evaluation at a fraction of realtime cost.**

[LLM-as-judge](https://doubleword.ai/glossary#llm-as-a-judge) is a standard way to measure answer quality at scale but it's
expensive, because you run a strong model over *every* output. The trick is that
evaluation is a **background** workload: you're measuring, not serving, so you
don't need realtime latency. That's exactly what batch inference is for.

This project takes a heavy eval workload — **generate answers to all 817
[TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) questions,
then judge every one of them** — runs both halves on
[Doubleword's batch tier](https://docs.doubleword.ai/inference-api/intro-to-doubleword-inference),
and lands the results in [Arize Phoenix](https://arize.com/phoenix) as a
first-class **Dataset + Experiment** you can browse, filter, and compare.

To run this yourself, install the [`dw` CLI](https://github.com/doublewordai/dw)
and `dw login`, or sign up at [app.doubleword.ai](https://app.doubleword.ai/).

<p align="center">
  <img src="images/demo-hero.png" height="auto" width="auto" alt="Doubleword × Arize Phoenix" />
</p>

> One Doubleword API key covers realtime, async, and batch. The batch tier runs on
> Doubleword's high-throughput backend — see **[doubleword.ai/pricing](https://doubleword.ai/pricing/)**.

## Why this matters

Running an LLM judge at realtime prices doesn't scale: an agentic app makes many
calls per request, and judging each one at full price turns quality monitoring
into a line item nobody wants to pay. So teams sample a handful of outputs, eyeball
them, and hope.

Batch changes the math. Because evaluation tolerates latency, you move it off the
hot path: enqueue the whole workload and let Doubleword's high-throughput backend
chew through it at batch rates. Suddenly judging *every* output — not a 1%
sample — is affordable.

The missing half is **observability**. A cheap score you can't see isn't useful.
So we push the eval set and every judge score into Phoenix as a Dataset +
Experiment: per-example inputs, generated answers, and four scores (relevance,
truthfulness, tone, overall), with run-to-run comparison built in.

The result: comprehensive, observable evals at a fraction of realtime cost.

## How it works

Two batches, one dataset, one experiment:

```
TruthfulQA (817 q)
   │
   ├─▶ Batch 1: GENERATE answers           (Doubleword batch)
   │
   ├─▶ Batch 2: JUDGE each answer          (Doubleword batch — the expensive part)
   │            relevance · truthfulness · tone
   │
   └─▶ Arize Phoenix
         Dataset:    the 817 questions + reference answers
         Experiment: generated answer + 4 judge scores per example
```

The judge is a single prompt returning a typed score — see
[`src/judge.py`](src/judge.py). Everything else is plumbing: load the data,
emit OpenAI-format batch JSONL, submit it via the `dw` CLI, and record the
results in Phoenix. There is **no agent and no retrieval** — just the eval
workload and the integration, which is the whole point.

### The Phoenix bridge

The expensive work happens out-of-process on the batch tier. By the time results
return, every answer and score is already computed — so the Phoenix Experiment's
`task` and `evaluators` are pure lookups (no live LLM calls). See
[`src/phoenix_io.py`](src/phoenix_io.py). Phoenix still renders per-example
outputs, all four scores, and lets you compare experiments across models or
prompts.

---

## Quick start

### 1. Prerequisites

- A **Doubleword API key** — sign in at [app.doubleword.ai](https://app.doubleword.ai/).
- The **[`dw` CLI](https://github.com/doublewordai/dw)**:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/doublewordai/dw/main/install.sh | sh
  dw login --api-key dw-...
  ```
- **[uv](https://docs.astral.sh/uv/)** and **[Docker](https://www.docker.com/get-started/)**.

### 2. Configure

```bash
cp .env.example .env     # then paste your DOUBLEWORD_API_KEY
uv sync --extra dev
```

Every other setting has a working default in [`src/config.py`](src/config.py).

### 3. Start Phoenix (local, Docker)

```bash
docker compose up -d
# Phoenix UI at http://localhost:6006
```

A local Phoenix is the only standing dependency — fair, since visualising the
results in Arize is the point of the integration.

### 4. Run the pipeline with the `dw` CLI

The full workflow lives in [`dw.toml`](dw.toml). Run it end-to-end:

```bash
dw project run-all
```

…or step through it for control. Generate answers (one batch):

```bash
dw project run prepare -- -n 817                         # TruthfulQA → generate.jsonl + Phoenix Dataset
dw files prepare batches/generate.jsonl --model deepseek-ai/DeepSeek-V4-Pro
dw batches run batches/generate.jsonl --watch --output-id .gen-id
dw batches results --from-file .gen-id -o results/answers.jsonl
```

Judge every answer (the expensive batch):

```bash
dw project run prepare-judge -- -a results/answers.jsonl  # answers → judge.jsonl
dw files prepare batches/judge.jsonl --model deepseek-ai/DeepSeek-V4-Pro
dw batches run batches/judge.jsonl --watch --output-id .judge-id
dw batches results --from-file .judge-id -o results/scores.jsonl
```

Score, record the Phoenix Experiment, and check the cost:

```bash
dw project run analyze -- -a results/answers.jsonl -s results/scores.jsonl
dw batches analytics --from-file .gen-id
dw batches analytics --from-file .judge-id
```

Open [http://localhost:6006](http://localhost:6006) → **Datasets → truthfulqa-eval
→ Experiments** to browse per-example answers and scores.

### 5. Or run it all in one script

If you'd rather not orchestrate the CLI, [`eval.py`](eval.py) does the whole loop
in-process (both batches via `autobatcher.BatchOpenAI`, then the Phoenix
Dataset + Experiment):

```bash
uv run python eval.py -n 50      # quick run
uv run python eval.py            # full 817
```

Save this as `eval.py`, export your API keys, and run it. You've just built a
scalable, cost-managed, rate-limit-proof eval pipeline.

---

## Results

> _Generation is measured (817/817). Judge is extrapolated to the full 817 from the
> measured per-request cost of the judge batch; mean scores are from the graded
> sample. Batch cost comes from `dw batches analytics`._

| Stage | Requests | Input tokens | Output tokens | Batch cost |
|-------|---------:|-------------:|--------------:|-----------:|
| Generate | 817 | 34,175 | 70,012 | $0.15 |
| Judge | 817 | ~231,500 | ~82,400 | ~$0.34 |
| **Total** | **1,634** | **~265,700** | **~152,400** | **~$0.50** |

Mean judge scores (0–1): relevance **0.92** · truthfulness **0.93** · tone **0.95** (overall **0.93**).

Batch cost is reported by `dw batches analytics` (authoritative, always current).
See [doubleword.ai/pricing](https://doubleword.ai/pricing) for live rates.

---

## Tests

```bash
uv run pytest        # offline: JSONL shape, judge prompt, Score model, result parsing
```

The suite is hermetic — no network, no Phoenix, no API key required.

## Project layout

```
eval.py              one-file pipeline (both batches → Phoenix)
dw.toml              dw CLI workflow (prepare → run → results → analyze)
src/
  config.py          env → validated settings
  judge.py           the judge: one prompt + a typed Score
  data.py            TruthfulQA load, batch JSONL emit, result parsing
  phoenix_io.py      Doubleword-batch ↔ Phoenix Dataset/Experiment bridge
  cli.py             prepare / prepare-judge / analyze
tests/               offline unit tests
```
