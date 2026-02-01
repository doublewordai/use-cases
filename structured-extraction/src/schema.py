"""Extraction schema for receipt data."""

from pydantic import BaseModel, Field
from typing import Optional


class ReceiptExtraction(BaseModel):
    """Structured extraction from a receipt image."""

    vendor_name: Optional[str] = Field(default=None, description="Name of the store or vendor")
    vendor_address: Optional[str] = Field(default=None, description="Address of the vendor")
    date: Optional[str] = Field(default=None, description="Transaction date")
    total: Optional[float] = Field(default=None, description="Total amount paid")


# SROIE dataset fields
EVAL_FIELDS = ["vendor_name", "date", "total"]


def get_image_extraction_prompt() -> str:
    """Generate the extraction prompt for a receipt image."""
    return """Extract structured information from this receipt image.

Extract the following fields:
- vendor_name: The name of the store or business (usually at the top)
- vendor_address: The full address of the store
- date: Transaction date (format as YYYY-MM-DD if possible, otherwise as shown)
- total: Final total amount paid (the main total, not subtotals)

Return your response as valid JSON:
{
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "date": "string or null",
  "total": number or null
}

Only include fields you can confidently extract. Use null for missing/unclear values.
Respond with ONLY the JSON, no other text."""


def get_extraction_prompt(ocr_text: str) -> str:
    """Generate the extraction prompt for OCR text (legacy)."""
    return f"""Extract structured information from this receipt text.

Receipt text:
---
{ocr_text}
---

Extract the following fields:
- vendor_name: The name of the store or business
- vendor_address: The full address
- date: Transaction date (format as YYYY-MM-DD if possible)
- total: Final total paid

Return your response as valid JSON:
{{
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "date": "string or null",
  "total": number or null
}}

Only include fields you can confidently extract. Use null for missing/unclear values.
Respond with ONLY the JSON, no other text."""
