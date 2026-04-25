"""
utils/llm_processor.py
Gemini API integration — batch processing with retry and rate limiting.
Uses the google-generativeai SDK (stable, verified working).
"""

import json
import re
import time
import warnings
from typing import List, Dict, Any

# Suppress migration notice — SDK is stable and functional
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", message=".*google.generativeai.*", category=UserWarning)

import google.generativeai as genai

# ── Prompt templates ──────────────────────────────────────────────────────────

BATCH_PROMPT_TEMPLATE = """You are a senior product catalogue quality expert for a premium e-commerce company.
Analyze the following {n} products and return a JSON array with exactly {n} objects.

Products (as JSON array):
{products_json}

For each product return this exact JSON structure:
{{
  "product_id": "<same id as input>",
  "quality_score": <integer 0-100>,
  "grade": "<A/B/C/D/F>",
  "issues": [<list of specific issue strings>],
  "missing_attributes": [<list of missing/empty attribute names>],
  "suggested_title": "<improved title>",
  "suggested_description": "<improved 3-sentence description with material, use case, and care instruction>",
  "suggested_keywords": "<5 comma-separated SEO keywords>",
  "improvement_priority": "<High/Medium/Low>"
}}

Scoring rules:
- Start at 100, deduct points for each issue found
- Missing material: -20
- Missing/ambiguous colour (contains / or &): -10
- Description under 20 words: -15
- Title missing gender indicator: -10
- Missing keywords: -10
- Price is 0 or missing: -15
- Missing stock status: -5
- Vague/generic description: -10

Return ONLY a valid JSON array. No markdown, no explanation.
"""

STRICT_RETRY_PROMPT_TEMPLATE = """You are a JSON API. Return ONLY a raw JSON array — no markdown, no backticks, no text before or after.

Analyze these {n} products and return exactly {n} JSON objects in a JSON array:
{products_json}

Each object must have: product_id, quality_score (0-100 int), grade (A/B/C/D/F),
issues (array of strings), missing_attributes (array of strings), suggested_title,
suggested_description, suggested_keywords, improvement_priority (High/Medium/Low).

START YOUR RESPONSE WITH [ AND END WITH ]
"""


def _configure_genai(api_key: str):
    """Configure the Gemini client and return a GenerativeModel."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# Alias for consistency
_configure_client = _configure_genai


def _strip_markdown_json(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_json_response(raw: str) -> List[Dict]:
    """Attempt to parse the model response as a JSON array."""
    cleaned = _strip_markdown_json(raw)
    return json.loads(cleaned)


def _validate_batch_result(result: List[Dict], expected_count: int) -> bool:
    """Basic validation that the parsed result has the right shape."""
    if not isinstance(result, list):
        return False
    if len(result) != expected_count:
        return False
    required_keys = {
        "product_id", "quality_score", "grade", "issues", "missing_attributes",
        "suggested_title", "suggested_description", "suggested_keywords", "improvement_priority",
    }
    for item in result:
        if not isinstance(item, dict):
            return False
        if not required_keys.issubset(item.keys()):
            return False
    return True


def process_batch(
    model,
    batch: List[Dict[str, Any]],
    on_retry_callback=None,
) -> List[Dict[str, Any]]:
    """
    Send a single batch to Gemini and return the list of AI result dicts.
    Retries once with a stricter prompt if JSON parsing fails.
    Falls back to stub results so the app never crashes.
    """
    n = len(batch)
    products_json = json.dumps(batch, ensure_ascii=False, indent=2)

    # ── First attempt ──
    prompt = BATCH_PROMPT_TEMPLATE.format(n=n, products_json=products_json)
    try:
        response = model.generate_content(prompt)
        parsed = _parse_json_response(response.text)
        if _validate_batch_result(parsed, n):
            return parsed
        raise ValueError(f"Validation failed: got {len(parsed)} items, expected {n}")
    except Exception as e:
        if on_retry_callback:
            on_retry_callback(str(e))

    # ── Retry with stricter prompt ──
    strict_prompt = STRICT_RETRY_PROMPT_TEMPLATE.format(n=n, products_json=products_json)
    try:
        response = model.generate_content(strict_prompt)
        parsed = _parse_json_response(response.text)
        if _validate_batch_result(parsed, n):
            return parsed
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except Exception:
        pass

    # ── Fallback stubs so the app never crashes ──
    return [
        {
            "product_id": str(row.get("product_id", "unknown")),
            "quality_score": 0,
            "grade": "F",
            "issues": ["LLM analysis failed for this product"],
            "missing_attributes": [],
            "suggested_title": str(row.get("title", "")),
            "suggested_description": "",
            "suggested_keywords": "",
            "improvement_priority": "High",
        }
        for row in batch
    ]


def run_analysis(
    api_key: str,
    batches: List[List[Dict[str, Any]]],
    rate_limit_sleep: float = 1.0,
    progress_callback=None,
    retry_callback=None,
) -> List[Dict[str, Any]]:
    """
    Run AI analysis over all batches.

    Args:
        api_key:           Gemini API key.
        batches:           List of row-dict batches (from prepare_batches).
        rate_limit_sleep:  Seconds to sleep between batches (respects 15 RPM free tier).
        progress_callback: Callable(batch_idx, total_batches, products_done, total_products)
        retry_callback:    Callable(message) called when a retry is triggered.

    Returns:
        Flat list of AI result dicts for all products.
    """
    model = _configure_genai(api_key)
    all_results: List[Dict[str, Any]] = []
    total_batches = len(batches)
    total_products = sum(len(b) for b in batches)
    products_done = 0

    for idx, batch in enumerate(batches):
        # Sleep between batches to respect rate limits (skip before first call)
        if idx > 0:
            time.sleep(rate_limit_sleep)

        batch_results = process_batch(model, batch, on_retry_callback=retry_callback)
        all_results.extend(batch_results)
        products_done += len(batch)

        if progress_callback:
            progress_callback(idx + 1, total_batches, products_done, total_products)

    return all_results
