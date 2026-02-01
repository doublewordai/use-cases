# Use Case Rubric

Each use case demonstrates a "more is different" capability of the Doubleword Batch API—something that becomes possible when LLM inference is cheap and scalable.

## Deliverables

Each use case produces:

1. **Code**: Walkthrough implementation. Users sign up for Doubleword API, then step through. CLI is the default entrypoint.
2. **Report**: A concise markdown file summarizing results (see Report Format below).
3. **Live demo** (stretch goal): User inputs their API key, we visually illustrate the example.

## Report Format

The report (`report.md`) should be concise and include:

- **What we did**: Brief description of the experiment
- **Data**: What input data, how much
- **Baseline**: What we compared against
- **Results**: Key findings with numbers
- **Cost comparison**: Requests made, cost on OpenAI vs Doubleword
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

## Folder Structure

```
use-case-name/
├── INSTRUCTIONS.md    # Brief describing the use case
├── pyproject.toml     # Project config (standardized)
├── src/               # Implementation code
│   ├── __init__.py
│   ├── cli.py         # CLI entrypoint
│   ├── batch.py       # Batch API utilities
│   └── ...
├── data/              # Sample data or scripts to fetch it
├── demo/              # Live demo (stretch goal)
└── report.md          # Results report
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

See `CLAUDE.md` for full implementation details.
