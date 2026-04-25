"""
utils/quality_scorer.py
Local pre-scorer — runs BEFORE the LLM call to give instant preview scores.
"""

import re
import pandas as pd

GENDER_KEYWORDS = ["women's", "men's", "womens", "mens", "unisex", "boys'", "girls'"]
AMBIGUOUS_COLOUR_PATTERN = re.compile(r"[/&]", re.IGNORECASE)


def _check_missing(value) -> bool:
    """Return True if value is empty / NaN."""
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    return str(value).strip() == ""


def _word_count(text) -> int:
    if _check_missing(text):
        return 0
    return len(str(text).split())


def pre_score_row(row: dict) -> dict:
    """
    Compute a local quality pre-score for a single product row.
    Returns a dict:
      {
        'pre_score': int,
        'grade': str,
        'issues': list[str],
        'missing_attributes': list[str]
      }
    """
    score = 100
    issues = []
    missing_attributes = []

    # --- Missing material ---
    if _check_missing(row.get("material")):
        score -= 20
        issues.append("Missing material")
        missing_attributes.append("material")

    # --- Ambiguous colour ---
    colour = str(row.get("colour", "")).strip()
    if _check_missing(row.get("colour")):
        score -= 10
        issues.append("Missing colour")
        missing_attributes.append("colour")
    elif AMBIGUOUS_COLOUR_PATTERN.search(colour):
        score -= 10
        issues.append(f"Ambiguous colour value: '{colour}'")

    # --- Short description ---
    desc_words = _word_count(row.get("description"))
    if desc_words == 0:
        score -= 15
        issues.append("Missing description")
        missing_attributes.append("description")
    elif desc_words < 20:
        score -= 15
        issues.append(f"Description too short ({desc_words} words, need ≥20)")

    # --- Title missing gender ---
    title = str(row.get("title", "")).lower()
    if not any(kw in title for kw in GENDER_KEYWORDS):
        score -= 10
        issues.append("Title missing gender indicator")

    # --- Missing / empty keywords ---
    if _check_missing(row.get("keywords")):
        score -= 10
        issues.append("Missing keywords")
        missing_attributes.append("keywords")

    # --- Price is 0 or missing ---
    price_raw = row.get("price", None)
    try:
        price_val = float(price_raw)
        if price_val <= 0:
            score -= 15
            issues.append("Price is zero or negative")
    except (TypeError, ValueError):
        score -= 15
        issues.append("Missing or invalid price")
        missing_attributes.append("price")

    # --- Missing stock_status ---
    if _check_missing(row.get("stock_status")):
        score -= 5
        issues.append("Missing stock_status")
        missing_attributes.append("stock_status")

    score = max(0, score)
    grade = _score_to_grade(score)

    return {
        "pre_score": score,
        "grade": grade,
        "issues": issues,
        "missing_attributes": missing_attributes,
    }


def _score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 45:
        return "D"
    return "F"


def pre_score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply pre_score_row to every row in the DataFrame.
    Adds columns: pre_score, pre_grade, pre_issues, pre_missing_attributes
    """
    results = df.apply(lambda row: pre_score_row(row.to_dict()), axis=1)
    df = df.copy()
    df["pre_score"] = results.apply(lambda r: r["pre_score"])
    df["pre_grade"] = results.apply(lambda r: r["grade"])
    df["pre_issues"] = results.apply(lambda r: "; ".join(r["issues"]) if r["issues"] else "None")
    df["pre_missing_attributes"] = results.apply(
        lambda r: "; ".join(r["missing_attributes"]) if r["missing_attributes"] else "None"
    )
    return df
