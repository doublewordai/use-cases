# Dataset Compilation

Use LLMs with web search to compile exhaustive datasets of real-world information. "Find all X that match Y" becomes a batch job.

## The idea

Compiling datasets manually takes days. An LLM with search can:
- Systematically explore a search space
- Extract structured data from results
- Find entities that commercial databases miss

Example: "All SaaS companies in the UK with >$1M ARR"

## Suggested approach

- Define target entities and criteria
- Generate search queries for coverage
- Extract structured data from search results
- Deduplicate and merge
- Validate a sample, measure accuracy and completeness

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
