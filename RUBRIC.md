# Use Case Rubric

Each use case demonstrates a "more is different" capability of the Doubleword
Batch API—something that becomes possible when LLM inference is cheap and
scalable.

## Deliverables

Each use case produces:

1. **Code**: Walkthrough implementation. Users sign up for Doubleword API, then step through. CLI is the default entrypoint.
2. **README**: The published artifact—combines brief, results, and replication instructions (see Report Format below).
3. **Live demo** (stretch goal): User inputs their API key, we visually illustrate the example.

## Report Format

The README should be concise and include:

- **What we did**: Brief description of the experiment
- **Data**: What input data, how much
- **Baseline**: What we compared against
- **Results**: Key findings with numbers
- **Cost comparison**: Requests made/tokens used, cost on OpenAI vs Doubleword
- **Conclusion**: One paragraph takeaway

## Evaluation Criteria

### Thesis

Does it demonstrate a "more is different" insight? Why does batch enable something qualitatively new?

### Baseline

Is there a meaningful comparison? (single-shot, embeddings, human effort, traditional ML)

### Metrics

Are improvements measurable? (accuracy, coverage, cost, time)

### Success

Is the result compelling? Frame as:

- "X% improvement over baseline"
- "Surfaces Y findings that baseline missed"
- "Achieves Z at 1/N the cost"

### Cost

Is there a clear cost comparison showing batch advantage?

## Quality Criteria

Each report section has minimum quality standards:

| Element | Quality Criteria |
|---------|------------------|
| **What we did** | States the thesis in one sentence. Reader knows the "so what" immediately. |
| **Data** | Real-world data with clear provenance. Not synthetic or toy. Size is meaningful (not 10 samples). |
| **Baseline** | Fair comparison—something a reader would actually consider. Not a strawman. |
| **Results** | Headline number is clear. Tables are scannable. Includes failure analysis, not just wins. |
| **Cost comparison** | Apples-to-apples (same tokens, same task). Shows actual $ spent. Cites pricing sources. |
| **Conclusion** | Answers "should I use this?" not just "what did we learn?" |

## Style Guide

### Voice

Conversational, technical, direct. Explaining something interesting to a smart colleague.

- Show reasoning, not just conclusions. Walk through what doesn't work before what does.
- Be concrete and definitive. Express genuine uncertainty when appropriate.
- First person is fine: "We ran...", "I think..."
- Casual asides land well: "painful.", "fair enough"
- Dry humor works: "If you speed something up by 20%, you did a good job. If you speed it up by 1000%, somebody did a bad job."
- Prefer longer, carefully constructed sentences over short punchy ones. Vary rhythm, but err longer.

### Avoid

- Corporate voice ("we're excited to announce...")
- Cliches: "the heavy lifting," "deep dive," "secret sauce"
- "Simply do X" (nothing is simple if you don't know how)
- "Obviously," "easy," "just" (dismissive)
- Em dashes, emojis
- Bulleted lists without analysis—have a high threshold for using them
- Avoid LLM-isms - "It's not X-It's Y", "It's not just something - it's something else", em dashes, etc.

### Structure

Use cases serve twin goals:

1. **Results**: Here's what you can achieve with batch inference
2. **Replication**: Here's how to do it yourself for similar tasks

These can be separate sections. 

### Code Examples

Code should be threaded through explanation, not dumped in blocks.

- Guide readers toward understanding how to use batch for the task
- Clear and concise—not exhaustive
- Real code, not pseudocode
- Focus on the concept being illustrated; cut boilerplate

### CLI Uniformity

CLI structure should be reasonably uniform across projects:

- Similar number of steps to run a complete example
- Aligned step names where tasks overlap (e.g., `status`, `analyze`)
- Consistent flags (`-m` for model, `-n` for count, `-o` for output)

### Call to Action

Every use case should open with a clear path to trying it:

> To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

Place this early—before the reader gets deep into results. The goal is conversion, not just education.

## Pre-Publish Checklist

Before publishing, verify:

- [ ] Thesis stated in first paragraph
- [ ] CTA to sign up and get API key near the top
- [ ] Data is real-world with clear source
- [ ] Baseline is fair (not strawman)
- [ ] Cost comparison uses actual $ with sources cited
- [ ] Key finding is in the title or subtitle
- [ ] Limitations acknowledged
- [ ] Code runs with `uv sync && uv run <command>`
- [ ] README is skimmable in 2 minutes

## Folder Structure

```
use-case-name/
├── README.md          # Combined brief + results (the published artifact)
├── pyproject.toml     # Project config (standardized)
├── src/               # Implementation code
│   ├── __init__.py
│   ├── cli.py         # CLI entrypoint
│   ├── batch.py       # Batch API utilities
│   └── ...
├── data/              # Sample data or scripts to fetch it
└── demo/              # Live demo (stretch goal)
```

## Project Alignment

All use cases must follow these conventions for consistency:

### Required Configuration

| Setting | Value |
|---------|-------|
| Python version | `>=3.11` |
| Build system | hatchling |
| Entry point | `[project.scripts]` matching folder name |

### Required Models

All projects comparing models must include these with consistent aliases:

| Alias | Model |
|-------|-------|
| `30b` | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| `235b` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` |
| `gpt5-nano` | `gpt-5-nano` |
| `gpt5-mini` | `gpt-5-mini` |
| `gpt5.2` | `gpt-5.2` |

### CLI Options

Standard options across all projects:

- `--model`, `-m`: Model alias or full name (default: `30b`)
- `--input`, `-i`: Input file or directory
- `--output`, `-o`: Output directory (default: `results/`)
- `--dry-run`: Prepare but don't submit

See `BATCH_API.md` for full implementation details.
