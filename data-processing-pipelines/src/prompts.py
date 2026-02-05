"""Prompts and JSON schemas for data processing pipelines.

Uses OpenAI-compatible structured outputs to guarantee valid JSON responses.
See: https://platform.openai.com/docs/guides/structured-outputs
"""

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

# --- Normalize Stage ---

NORMALIZE_PROMPT = (
    "You are a data normalization specialist. Given a company record from SEC "
    "EDGAR filings, standardize the company name into a clean, canonical format.\n\n"
    "Rules:\n"
    "- Expand abbreviations: Corp -> Corporation, Inc -> Incorporated, "
    "Intl -> International, Ltd -> Limited, Hldgs -> Holdings\n"
    "- Fix casing to proper title case (e.g. 'NVIDIA CORP' -> 'Nvidia Corporation')\n"
    "- Keep legal suffixes (Inc., Corp., Ltd.) but standardize their format\n"
    "- Remove trailing slashes, extra whitespace, and stray punctuation\n"
    "- If an address is provided, parse it into structured components\n"
    "- If a field is unknown or not provided, use an empty string."
)

NORMALIZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalized_company",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "normalized_name": {
                    "type": "string",
                    "description": "The standardized company name",
                },
                "street": {
                    "type": "string",
                    "description": "Street address, or empty string if unknown",
                },
                "city": {
                    "type": "string",
                    "description": "City name, or empty string if unknown",
                },
                "state": {
                    "type": "string",
                    "description": "State/province code, or empty string if unknown",
                },
                "zip_code": {
                    "type": "string",
                    "description": "Postal/ZIP code, or empty string if unknown",
                },
                "country": {
                    "type": "string",
                    "description": "Country name, or empty string if unknown",
                },
            },
            "required": [
                "normalized_name",
                "street",
                "city",
                "state",
                "zip_code",
                "country",
            ],
            "additionalProperties": False,
        },
    },
}

# --- Enrich Stage ---

ENRICH_PROMPT = (
    "You are a business classification specialist. Given a company name, "
    "stock ticker, and exchange, classify the company into one of the "
    "following industry sectors:\n\n"
    + "\n".join(f"- {s}" for s in INDUSTRY_SECTORS)
    + "\n\n"
    "Also estimate the company size category based on what you know about it."
)

ENRICH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "enriched_company",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "The primary industry sector",
                    "enum": INDUSTRY_SECTORS,
                },
                "sub_industry": {
                    "type": "string",
                    "description": "More specific sub-industry classification",
                },
                "size": {
                    "type": "string",
                    "description": "Estimated company size",
                    "enum": ["large", "medium", "small"],
                },
                "confidence": {
                    "type": "string",
                    "description": "Confidence level in the classification",
                    "enum": ["high", "medium", "low"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the classification",
                },
            },
            "required": [
                "industry",
                "sub_industry",
                "size",
                "confidence",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}

# --- Dedup Stage ---

DEDUP_PROMPT = (
    "You are a data deduplication specialist. Given two company records, "
    "determine whether they refer to the same real-world entity.\n\n"
    "Consider:\n"
    "- Name variations (abbreviations, different legal suffixes, typos)\n"
    "- Ticker symbols (same ticker = same company)\n"
    "- Parent/subsidiary relationships (Alphabet Inc vs Google LLC are related "
    "but may be listed separately)\n"
    "- Different entities can have similar names "
    "(e.g., 'First National Bank of Chicago' vs 'First National Bank of Charlotte')"
)

DEDUP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "dedup_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_duplicate": {
                    "type": "boolean",
                    "description": "Whether the two records refer to the same entity",
                },
                "confidence": {
                    "type": "string",
                    "description": "Confidence level in the determination",
                    "enum": ["high", "medium", "low"],
                },
                "relationship": {
                    "type": "string",
                    "description": "The relationship between the two records",
                    "enum": ["same_entity", "parent_subsidiary", "unrelated"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the determination",
                },
            },
            "required": ["is_duplicate", "confidence", "relationship", "reasoning"],
            "additionalProperties": False,
        },
    },
}
