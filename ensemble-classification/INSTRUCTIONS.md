# Ensemble Classification

Run the same classification N times with varied prompts, majority-vote the results. Demonstrate that you can buy accuracy with cheap inference.

## The idea

A single LLM response has variance. Run it many times, aggregate, and you get:
- Higher accuracy (majority vote beats single shot)
- Confidence calibration (unanimous = high confidence, split = uncertain)
- A signal for what needs human review

## Suggested approach

- Use a labelled sentiment dataset (SST-2, IMDB)
- Run classification at N=1, 5, 10, 20, 50
- Plot accuracy vs ensemble size
- Show confidence calibration (vote agreement vs actual accuracy)

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
