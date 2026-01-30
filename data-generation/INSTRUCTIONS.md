# Data Generation

Generate large synthetic datasets (training data, test fixtures, simulation inputs) at scales that would be prohibitive with real-time pricing.

## The idea

Many tasks need synthetic data: training ML models, populating test environments, running simulations. Batch makes generating 10,000+ records economical.

## Suggested approach

- Define a data schema (user profiles, transactions, products, etc.)
- Generate at scale with diversity instructions
- Validate against schema
- Measure: compliance rate, diversity metrics, cost per record
- Compare to rule-based generators (Faker) on realism/diversity

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
