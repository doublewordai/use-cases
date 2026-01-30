# Model Evals

Run large-scale model evaluations affordably. Thousands of test cases across multiple models to understand capabilities and track regression.

## The idea

Thorough model evaluation requires many test cases. Batch makes it economical to:
- Run full benchmark suites (MMLU, HumanEval, GSM8K)
- Compare models head-to-head
- Find category-specific weaknesses
- Track performance over time

## Suggested approach

- Pick an eval suite (or define custom evals)
- Run against target models via batch
- Compute scores per category
- Identify weaknesses not visible in aggregate scores
- Compare cost to real-time evaluation

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
