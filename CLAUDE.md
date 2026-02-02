# Use Case Editing Instructions

You are editing use case articles to prepare them for publication. Each use case has a README.md that needs to be improved to match the standards in `RUBRIC.md`.

## Your Task

1. Read `RUBRIC.md` thoroughly—it defines quality criteria, style guide, and the pre-publish checklist
2. Read the README.md in the use case folder you've been assigned
3. Edit the README to meet the rubric standards
4. Verify against the pre-publish checklist

## Use Cases

| Folder | Topic |
|--------|-------|
| `bug-detection-ensemble/` | Vulnerability classification with LLM ensemble |
| `dataset-compilation/` | Company dataset compilation via LLM + search |
| `model-evals/` | Model evaluation on GSM8K benchmark |
| `structured-extraction/` | Receipt data extraction with vision models |

## Key Improvements to Make

### Structure
- Thesis in first paragraph (the "more is different" insight)
- CTA early: "To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key."
- Twin sections: Results (what you get) and Replication (how to do it yourself)
- Key finding in title or subtitle

### Style
- Conversational, technical, direct voice
- Show reasoning, not just conclusions
- Remove corporate voice, cliches, "simply", "just", "easy"
- Thread code through explanation (not code dumps)
- Prefer longer sentences; vary rhythm

### Content
- Ensure baseline comparison is fair
- Cost comparison with actual $ and cited sources
- Include failure analysis, not just wins
- Acknowledge limitations
- Verify code examples actually run

## Pre-Publish Checklist

Before you're done, verify:

- [ ] Thesis stated in first paragraph
- [ ] CTA to sign up and get API key near the top
- [ ] Data is real-world with clear source
- [ ] Baseline is fair (not strawman)
- [ ] Cost comparison uses actual $ with sources cited
- [ ] Key finding is in the title or subtitle
- [ ] Limitations acknowledged
- [ ] Code runs with `uv sync && uv run <command>`
- [ ] README is skimmable in 2 minutes

## Output

Edit the README.md directly. Make substantive improvements—don't just add a CTA and call it done. The goal is publication-ready articles that make someone want to try the Doubleword Batch API.
