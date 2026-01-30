# Bug Detection Ensemble

Ask "is there a bug?" 20 different ways, aggregate responses. Catch more bugs than single-shot review, and use vote splits to identify code needing human attention.

## The idea

Different prompts catch different bugs. An ensemble of review prompts:
- Increases recall (fewer missed bugs)
- Provides confidence signal (unanimous = clear, split = ambiguous)
- Covers different bug categories (security, logic, style)

## Suggested approach

- Collect code samples with known bugs (CWE examples, synthetic bugs)
- Create varied review prompts (security focus, logic focus, different personas)
- Run ensemble, aggregate bug findings
- Measure precision/recall vs single-shot
- Analyze whether vote splits correlate with bug difficulty

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
