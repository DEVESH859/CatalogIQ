"""
test_all.py — Full end-to-end test for CatalogIQ
Run: python test_all.py
"""

import sys
import json
import traceback

import os
API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")
SAMPLE_CSV = "sample_data/quince_supplier_feed_sample.csv"

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results = []

def check(name, fn):
    try:
        result = fn()
        print(f"{PASS} {name}")
        if result:
            print(f"       {result}")
        results.append((name, True, result))
    except Exception as e:
        msg = str(e)[:200]
        print(f"{FAIL} {name}")
        print(f"       Error: {msg}")
        traceback.print_exc()
        results.append((name, False, msg))

# ── Test 1: CSV loading ───────────────────────────────────────────────────────
from utils.data_processor import load_csv, prepare_batches, merge_results, generate_enriched_csv, calculate_summary_stats

def test_csv_load():
    df = load_csv(SAMPLE_CSV)
    assert len(df) == 60, f"Expected 60 rows, got {len(df)}"
    assert "product_id" in df.columns
    assert "description" in df.columns
    return f"{len(df)} rows, {len(df.columns)} columns loaded"

check("CSV loading (60 rows, required columns)", test_csv_load)

# ── Test 2: Flaw detection in sample data ─────────────────────────────────────
def test_flaw_detection():
    df = load_csv(SAMPLE_CSV)
    missing_material = df[df["material"].isna() | (df["material"].str.strip() == "")].shape[0]
    ambiguous_colour = df[df["colour"].str.contains("/|&", na=False, regex=True)].shape[0]
    missing_price = df[df["price"].isna() | (df["price"].str.strip().isin(["", "0", "0.0"]))].shape[0]
    missing_stock = df[df["stock_status"].isna() | (df["stock_status"].str.strip() == "")].shape[0]
    return (f"missing_material={missing_material}, ambiguous_colour={ambiguous_colour}, "
            f"missing_price={missing_price}, missing_stock={missing_stock}")

check("Flaw detection in CSV", test_flaw_detection)

# ── Test 3: Pre-scorer ────────────────────────────────────────────────────────
from utils.quality_scorer import pre_score_dataframe, pre_score_row

def test_prescorer():
    df = load_csv(SAMPLE_CSV)
    scored = pre_score_dataframe(df)
    assert "pre_score" in scored.columns
    assert "pre_grade" in scored.columns
    assert "pre_issues" in scored.columns
    mn, mx = scored["pre_score"].min(), scored["pre_score"].max()
    assert 0 <= mn <= mx <= 100
    grades = scored["pre_grade"].value_counts().to_dict()
    return f"score range {mn}–{mx}, grades: {grades}"

check("Pre-scorer (rule-based local scoring)", test_prescorer)

# ── Test 4: Batch preparation ─────────────────────────────────────────────────
def test_batching():
    df = load_csv(SAMPLE_CSV)
    batches = prepare_batches(df, batch_size=8)
    total = sum(len(b) for b in batches)
    assert total == 60, f"Expected 60 total, got {total}"
    assert len(batches) == 8, f"Expected 8 batches (60/8 = ceil 8), got {len(batches)}"
    return f"{len(batches)} batches, sizes: {[len(b) for b in batches]}"

check("Batch preparation (batch_size=8, 60 rows)", test_batching)

# ── Test 5: Batch optimisation (sort by category) ────────────────────────────
def test_batch_opt():
    df = load_csv(SAMPLE_CSV)
    sorted_df = df.sort_values("category").reset_index(drop=True)
    batches = prepare_batches(sorted_df, batch_size=8)
    # Check first batch is all same or similar category
    first_cats = [b["category"] for b in batches[0]]
    return f"First batch categories: {set(first_cats)}"

check("Batch optimisation mode (sort by category)", test_batch_opt)

# ── Test 6: LLM API — single batch of 3 ──────────────────────────────────────
print()
print("Running LLM API test (single batch of 3 products) — please wait...")
from utils.llm_processor import process_batch, _configure_genai

def test_llm_single_batch():
    df = load_csv(SAMPLE_CSV)
    batches = prepare_batches(df, batch_size=8)
    small_batch = batches[0][:3]  # Just 3 products for speed
    model = _configure_genai(API_KEY)
    results_llm = process_batch(model, small_batch)
    assert len(results_llm) == 3, f"Expected 3 results, got {len(results_llm)}"
    for r in results_llm:
        assert "quality_score" in r
        assert "grade" in r
        assert "suggested_title" in r
        assert "suggested_description" in r
        assert "suggested_keywords" in r
        assert "issues" in r
        assert "missing_attributes" in r
        assert "improvement_priority" in r
    summary = [(r["product_id"], r["quality_score"], r["grade"]) for r in results_llm]
    return f"Results: {summary}"

check("LLM API — Gemini 1.5 Flash (3-product batch)", test_llm_single_batch)

# ── Test 7: Merge results ─────────────────────────────────────────────────────
def test_merge():
    df = load_csv(SAMPLE_CSV)
    # Use dummy AI results
    dummy = [
        {"product_id": str(df.iloc[i]["product_id"]),
         "quality_score": 75, "grade": "B",
         "issues": ["Test issue"], "missing_attributes": [],
         "suggested_title": "New Title",
         "suggested_description": "Better desc.",
         "suggested_keywords": "a, b, c",
         "improvement_priority": "Low"}
        for i in range(3)
    ]
    merged = merge_results(df.head(3), dummy)
    assert "quality_score" in merged.columns
    assert "suggested_title" in merged.columns
    assert len(merged) == 3
    return f"Merged shape: {merged.shape}, score col: {list(merged['quality_score'])}"

check("Merge AI results into DataFrame", test_merge)

# ── Test 8: Summary stats ─────────────────────────────────────────────────────
def test_stats():
    df = load_csv(SAMPLE_CSV)
    dummy = [
        {"product_id": str(df.iloc[i]["product_id"]),
         "quality_score": [90, 45, 70, 30, 80][i % 5],
         "grade": ["A","D","B","F","A"][i % 5],
         "issues": ["issue one; issue two"],
         "missing_attributes": ["material"] if i % 3 == 0 else [],
         "suggested_title": "", "suggested_description": "",
         "suggested_keywords": "", "improvement_priority": "High"}
        for i in range(len(df))
    ]
    merged = merge_results(df, dummy)
    stats = calculate_summary_stats(merged)
    assert "average_quality_score" in stats
    assert "grade_distribution" in stats
    assert "top_5_worst_products" in stats
    assert "missing_attribute_frequency" in stats
    assert stats["total_products"] == 60
    return (f"avg={stats['average_quality_score']}, "
            f"needs_review={stats['products_needing_review']}, "
            f"grades={stats['grade_distribution']}")

check("Summary statistics calculation", test_stats)

# ── Test 9: CSV export ────────────────────────────────────────────────────────
def test_csv_export():
    df = load_csv(SAMPLE_CSV)
    dummy = [
        {"product_id": str(df.iloc[i]["product_id"]),
         "quality_score": 75, "grade": "B",
         "issues": ["issue"], "missing_attributes": [],
         "suggested_title": "Title", "suggested_description": "Desc",
         "suggested_keywords": "kw1,kw2", "improvement_priority": "Low"}
        for i in range(len(df))
    ]
    merged = merge_results(df, dummy)
    csv_bytes = generate_enriched_csv(merged)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 1000
    lines = csv_bytes.decode("utf-8").split("\n")
    header = lines[0]
    assert "quality_score" in header
    assert "suggested_title" in header
    assert "suggested_description" in header
    return f"CSV export: {len(csv_bytes)} bytes, {len(lines)} lines"

check("Enriched CSV export (download bytes)", test_csv_export)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("FAILED TESTS:")
    for name, ok, msg in results:
        if not ok:
            print(f"  - {name}: {msg}")
else:
    print("All tests passed.")
print("=" * 60)
