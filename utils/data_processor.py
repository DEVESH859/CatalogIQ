"""
utils/data_processor.py
CSV parsing, batching, merging, and summary statistics.
"""

import io
import pandas as pd
from typing import List, Dict, Any

REQUIRED_COLUMNS = [
    "product_id",
    "title",
    "category",
    "material",
    "colour",
    "description",
    "price",
    "gender",
    "size_range",
    "season",
    "keywords",
    "stock_status",
]

AI_OUTPUT_COLUMNS = [
    "quality_score",
    "grade",
    "issues",
    "suggested_title",
    "suggested_description",
    "suggested_keywords",
    "missing_attributes",
    "improvement_priority",
]


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV from an uploaded Streamlit file object or a file path string.
    Validates that all required columns are present.
    Raises ValueError with missing column names on failure.
    """
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file, dtype=str)
    else:
        df = pd.read_csv(uploaded_file, dtype=str)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Normalise whitespace but keep raw values for display
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def prepare_batches(df: pd.DataFrame, batch_size: int = 8) -> List[List[Dict[str, Any]]]:
    """
    Split the DataFrame into batches of `batch_size` rows.
    Each row is converted to a plain dict.
    Returns a list of batches (each batch = list of row dicts).
    """
    records = df.to_dict(orient="records")
    batches = []
    for i in range(0, len(records), batch_size):
        batches.append(records[i : i + batch_size])
    return batches


def merge_results(original_df: pd.DataFrame, ai_results_list: List[Dict]) -> pd.DataFrame:
    """
    Merge AI output dicts back into the original DataFrame by product_id.
    Missing AI results for a product_id are filled with defaults.
    """
    if not ai_results_list:
        return original_df.copy()

    ai_df = pd.DataFrame(ai_results_list)

    # Ensure product_id is str in both
    original_df = original_df.copy()
    original_df["product_id"] = original_df["product_id"].astype(str)
    ai_df["product_id"] = ai_df["product_id"].astype(str)

    # Convert list columns to semicolon-separated strings for storage
    for col in ["issues", "missing_attributes"]:
        if col in ai_df.columns:
            ai_df[col] = ai_df[col].apply(
                lambda v: "; ".join(v) if isinstance(v, list) else str(v) if pd.notna(v) else ""
            )

    merged = original_df.merge(ai_df, on="product_id", how="left")

    # Fill defaults where merge didn't find a match
    defaults = {
        "quality_score": 0,
        "grade": "F",
        "issues": "Analysis failed",
        "suggested_title": "",
        "suggested_description": "",
        "suggested_keywords": "",
        "missing_attributes": "",
        "improvement_priority": "High",
    }
    for col, default in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
        else:
            merged[col] = default

    # Ensure quality_score is numeric
    merged["quality_score"] = pd.to_numeric(merged["quality_score"], errors="coerce").fillna(0).astype(int)

    return merged


def generate_enriched_csv(df: pd.DataFrame) -> bytes:
    """
    Return the enriched DataFrame as UTF-8 encoded CSV bytes for download.
    Ensures all original columns come first, followed by AI output columns.
    """
    original_cols = [c for c in df.columns if c not in AI_OUTPUT_COLUMNS]
    ai_cols = [c for c in AI_OUTPUT_COLUMNS if c in df.columns]
    ordered_cols = original_cols + ai_cols
    out = df[ordered_cols]
    return out.to_csv(index=False).encode("utf-8")


def calculate_summary_stats(enriched_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate aggregate quality metrics from the enriched DataFrame.
    """
    df = enriched_df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0)

    total = len(df)
    avg_score = round(float(df["quality_score"].mean()), 1)
    needs_review = int((df["quality_score"] < 60).sum())

    # Grade distribution
    grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    if "grade" in df.columns:
        counts = df["grade"].value_counts().to_dict()
        for g in grade_dist:
            grade_dist[g] = counts.get(g, 0)

    # Top 5 worst products
    worst_cols = ["product_id", "title", "quality_score", "grade", "issues"]
    available_worst = [c for c in worst_cols if c in df.columns]
    worst_5 = (
        df[available_worst]
        .sort_values("quality_score", ascending=True)
        .head(5)
        .to_dict(orient="records")
    )
    # Add top_issue field
    for row in worst_5:
        issues_str = row.get("issues", "")
        row["top_issue"] = issues_str.split(";")[0].strip() if issues_str else "N/A"

    # Missing attribute frequency
    attr_freq: Dict[str, int] = {}
    if "missing_attributes" in df.columns:
        for val in df["missing_attributes"].dropna():
            for attr in str(val).split(";"):
                attr = attr.strip()
                if attr and attr.lower() not in ("none", ""):
                    attr_freq[attr] = attr_freq.get(attr, 0) + 1

    return {
        "average_quality_score": avg_score,
        "grade_distribution": grade_dist,
        "top_5_worst_products": worst_5,
        "missing_attribute_frequency": attr_freq,
        "total_products": total,
        "products_needing_review": needs_review,
    }
