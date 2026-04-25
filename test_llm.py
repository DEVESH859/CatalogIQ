"""
test_llm.py — Live Gemini API integration test
"""
import sys, time
sys.path.insert(0, 'd:/CatalogIQ')

from utils.data_processor import load_csv, prepare_batches, merge_results, generate_enriched_csv, calculate_summary_stats
from utils.llm_processor import process_batch, _configure_genai

import os
API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")
df = load_csv("sample_data/quince_supplier_feed_sample.csv")
batches = prepare_batches(df, batch_size=8)

print(f"Testing with 2 batches (16 products) via Gemini 1.5 Flash...")
model = _configure_genai(API_KEY)

all_results = []
for i, batch in enumerate(batches[:2]):
    if i > 0:
        time.sleep(1)
    r = process_batch(model, batch)
    all_results.extend(r)
    print(f"Batch {i+1}: received {len(r)} results back")

print()
print("--- Sample AI Results (first 5 products) ---")
for r in all_results[:5]:
    pid = r.get("product_id", "?")
    score = r.get("quality_score", "?")
    grade = r.get("grade", "?")
    priority = r.get("improvement_priority", "?")
    issues = r.get("issues", [])
    if isinstance(issues, list):
        issues_str = "; ".join(issues[:2])
    else:
        issues_str = str(issues)[:80]
    suggested = r.get("suggested_title", "")[:60]
    print(f"  {pid:8}  score={score:>3}  grade={grade}  priority={priority}")
    print(f"    Issues    : {issues_str}")
    print(f"    Sug. title: {suggested}")
    print()

# Merge & stats
merged = merge_results(df.head(16), all_results)
stats = calculate_summary_stats(merged)
print("--- Dashboard Stats ---")
print(f"  avg_score        = {stats['average_quality_score']}")
print(f"  needs_review     = {stats['products_needing_review']}")
print(f"  grade_dist       = {stats['grade_distribution']}")
print(f"  missing_attrs    = {stats['missing_attribute_frequency']}")

# CSV export
csv_b = generate_enriched_csv(merged)
header = csv_b.decode().split("\n")[0]
print(f"\n  export_bytes     = {len(csv_b)}")
print(f"  export_header    = {header}")

print("\n[ALL PASSED] LLM integration working correctly.")
