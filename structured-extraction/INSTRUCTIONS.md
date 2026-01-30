# Structured Extraction

Extract structured data from documents (receipts, invoices, contracts) using ensemble voting. Show that consensus extraction beats single-shot accuracy.

## The idea

Single-shot extraction makes errors. Run extraction multiple times, take the consensus per field, and you get:
- Higher field-level accuracy
- Confidence scores (consensus % indicates reliability)
- Clear signal for which extractions need review

## Suggested approach

- Use a labelled dataset (SROIE receipts, CORD, or similar)
- Define extraction schema (vendor, date, total, line items)
- Run extraction at N=1, 5, 10
- Compare field accuracy across ensemble sizes
- Identify which field types benefit most

## See also

Refer to [RUBRIC.md](../RUBRIC.md) for evaluation criteria and report format.
