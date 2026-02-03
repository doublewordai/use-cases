"""Prompts and configuration for data processing pipelines."""

INDUSTRY_SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Energy",
    "Manufacturing",
    "Retail",
    "Real Estate",
    "Telecommunications",
    "Transportation",
    "Education",
    "Agriculture",
    "Construction",
    "Entertainment",
    "Hospitality",
    "Legal Services",
    "Consulting",
    "Nonprofit",
    "Government",
    "Insurance",
    "Pharmaceuticals",
]

NORMALIZE_PROMPT = (
    "You are a data normalization specialist. Given a company record from SEC "
    "EDGAR filings, standardize the company name into a clean, canonical format.\n\n"
    "Rules:\n"
    "- Expand abbreviations: Corp -> Corporation, Inc -> Incorporated, "
    "Intl -> International, Ltd -> Limited, Hldgs -> Holdings\n"
    "- Fix casing to proper title case (e.g. 'NVIDIA CORP' -> 'Nvidia Corporation')\n"
    "- Keep legal suffixes (Inc., Corp., Ltd.) but standardize their format\n"
    "- Remove trailing slashes, extra whitespace, and stray punctuation\n"
    "- If an address is provided, parse it into structured components\n\n"
    "Return valid JSON with these fields:\n"
    '{"normalized_name": "...", "street": "...", "city": "...", '
    '"state": "...", "zip_code": "...", "country": "..."}\n\n'
    "If a field is unknown or not provided, use an empty string."
)

ENRICH_PROMPT = (
    "You are a business classification specialist. Given a company name, "
    "stock ticker, and exchange, classify the company into one of the "
    "following industry sectors:\n\n"
    + "\n".join(f"- {s}" for s in INDUSTRY_SECTORS)
    + "\n\n"
    "Also estimate the company size category based on what you know about it.\n\n"
    "Return valid JSON with:\n"
    '{"industry": "...", "sub_industry": "...", "size": "large|medium|small", '
    '"confidence": "high|medium|low", "reasoning": "..."}'
)

DEDUP_PROMPT = (
    "You are a data deduplication specialist. Given two company records, "
    "determine whether they refer to the same real-world entity.\n\n"
    "Consider:\n"
    "- Name variations (abbreviations, different legal suffixes, typos)\n"
    "- Ticker symbols (same ticker = same company)\n"
    "- Parent/subsidiary relationships (Alphabet Inc vs Google LLC are related "
    "but may be listed separately)\n"
    "- Different entities can have similar names "
    "(e.g., 'First National Bank of Chicago' vs 'First National Bank of Charlotte')\n\n"
    "Return valid JSON with:\n"
    '{"is_duplicate": true|false, "confidence": "high|medium|low", '
    '"relationship": "same_entity|parent_subsidiary|unrelated", "reasoning": "..."}'
)
