"""
app.py — CatalogIQ: AI-powered product catalogue quality inspector
Main Streamlit application.
"""

import os
import time
import warnings

# Suppress FutureWarning from google.generativeai SDK (cosmetic only, does not affect functionality)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.data_processor import (
    load_csv,
    prepare_batches,
    merge_results,
    generate_enriched_csv,
    calculate_summary_stats,
)
from utils.llm_processor import run_analysis
from utils.quality_scorer import pre_score_dataframe

# ── Env & page config ─────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="CatalogIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global reset & font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── Force Deep Slate Dark Theme ── */
.stApp {
    background-color: #0f172a !important;
}
/* Ensure default text is white */
.stApp p, .stApp span, .stApp div, .stApp label {
    color: #f8fafc;
}
/* But keep headers visible */
.stApp h1, .stApp h2, .stApp h3 {
    color: #ffffff !important;
}

/* ── Spacing & Layout ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ── Hero header ── */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.25rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 2.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #020617 !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* ── Logo text ── */
.sidebar-logo {
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}
.metric-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
}
.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.metric-delta {
    font-size: 0.8rem;
    color: #38bdf8;
    margin-top: 0.25rem;
}

/* ── Grade badges ── */
.grade-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}
.grade-A { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.grade-B { background: #422006; color: #facc15; border: 1px solid #ca8a04; }
.grade-C { background: #451a03; color: #fb923c; border: 1px solid #ea580c; }
.grade-D { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
.grade-F { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }

/* ── Score circle ── */
.score-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 auto 1rem;
    border: 4px solid;
    box-shadow: 0 0 20px rgba(255,255,255,0.05);
}
.score-high  { background: #064e3b; color: #4ade80; border-color: #22c55e; box-shadow: 0 0 15px rgba(34, 197, 94, 0.3); }
.score-mid   { background: #422006; color: #fde047; border-color: #eab308; box-shadow: 0 0 15px rgba(234, 179, 8, 0.3); }
.score-low-m { background: #451a03; color: #fdba74; border-color: #f97316; box-shadow: 0 0 15px rgba(249, 115, 22, 0.3); }
.score-low   { background: #450a0a; color: #fca5a5; border-color: #ef4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }

/* ── Before / After Layout  ── */
.raw-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
}
.raw-card * {
    color: #cbd5e1 !important;
}

.enriched-card {
    background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 4px 30px rgba(139, 92, 246, 0.15);
}

/* ── Section divider ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e293b;
    margin-top: 2rem;
}

/* ── Progress bar container ── */
.progress-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

/* ── Issue item ── */
.issue-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.35rem 0;
    color: #f87171 !important;
    font-size: 0.95rem;
}
.issue-bullet {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ef4444;
    margin-top: 0.45rem;
    flex-shrink: 0;
}

/* ── Priority badge ── */
.priority-high   { background: #450a0a; color: #fca5a5; padding: 0.2rem 0.7rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; border: 1px solid #ef4444; }
.priority-medium { background: #422006; color: #fde047; padding: 0.2rem 0.7rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; border: 1px solid #eab308; }
.priority-low    { background: #064e3b; color: #6ee7b7; padding: 0.2rem 0.7rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; border: 1px solid #10b981; }

/* ── Upload box enhancement ── */
[data-testid="stFileUploader"] {
    border-radius: 16px !important;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1rem;
    background: transparent;
    border-bottom: 1px solid #1e293b;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 1rem;
    padding-bottom: 0.5rem;
}

/* ── Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "uploaded_df": None,
    "enriched_df": None,
    "analysis_complete": False,
    "analysis_progress": 0.0,
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "status_text": "",
    "stats": None,
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper utilities ───────────────────────────────────────────────────────────

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "quince_supplier_feed_sample.csv")

GRADE_COLORS = {"A": "#22c55e", "B": "#eab308", "C": "#f97316", "D": "#ef4444", "F": "#ef4444"}


def _score_color(score: int) -> str:
    if score >= 80:
        return "#dcfce7"
    elif score >= 50:
        return "#fef9c3"
    return "#fee2e2"


def _score_text_color(score: int) -> str:
    if score >= 80:
        return "#166534"
    elif score >= 50:
        return "#854d0e"
    return "#991b1b"


def _score_class(score: int) -> str:
    if score >= 80:
        return "score-high"
    elif score >= 65:
        return "score-mid"
    elif score >= 50:
        return "score-low-m"
    return "score-low"


def _priority_class(priority: str) -> str:
    p = str(priority).lower()
    if p == "high":
        return "priority-high"
    elif p == "medium":
        return "priority-medium"
    return "priority-low"


def _color_score_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return Styler with color-coded quality_score column."""
    if "quality_score" not in df.columns:
        return df

    def highlight_score(val):
        try:
            v = int(val)
        except (TypeError, ValueError):
            return ""
        bg = _score_color(v)
        color = _score_text_color(v)
        return f"background-color: {bg}; color: {color}; font-weight: 600;"

    try:
        return df.style.map(highlight_score, subset=["quality_score"])
    except AttributeError:
        return df.style.applymap(highlight_score, subset=["quality_score"])


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">CatalogIQ</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem; margin-top:0.25rem;'>"
        "AI-powered product catalogue quality inspection</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b; font-size:0.75rem;'>"
        "Designed for the scaling challenges of multi-supplier e-commerce operations</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### Settings")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza...",
        help="Obtain a free API key at https://aistudio.google.com",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    batch_size = st.slider(
        "Batch Size",
        min_value=4,
        max_value=12,
        value=8,
        step=1,
        help="Number of products sent per API call. Larger batches reduce call volume but increase latency.",
    )

    # Batch processing inspired by Middle Mile routing logic —
    # reduce API calls by grouping similar items
    batch_opt = st.checkbox(
        "Batch Optimisation Mode",
        value=False,
        help="Sorts products by category before batching so similar items share LLM context, improving output consistency.",
    )

    show_preview = st.checkbox(
        "Show pre-analysis preview",
        value=True,
        help="Display local rule-based quality scores before submitting to the LLM.",
    )

    st.divider()
    st.markdown(
        "<p style='color:#475569; font-size:0.72rem;'>"
        "v1.0 &nbsp;·&nbsp; Gemini 2.5 Flash &nbsp;·&nbsp; 15 RPM free tier</p>",
        unsafe_allow_html=True,
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Upload & Analyse", "Quality Dashboard", "Before / After"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Upload & Analyse
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="hero-title">CatalogIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Upload a supplier product feed and surface every data quality issue in seconds.</div>',
        unsafe_allow_html=True,
    )

    # ── File upload row ──
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload your supplier product feed",
            type=["csv"],
            label_visibility="visible",
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load Sample Data", use_container_width=True):
            try:
                df = load_csv(SAMPLE_CSV_PATH)
                st.session_state.uploaded_df = df
                st.session_state.enriched_df = None
                st.session_state.analysis_complete = False
                st.session_state.analysis_progress = 0.0
                st.success(f"Sample data loaded — {len(df)} products")
            except Exception as exc:
                st.error(f"Failed to load sample data: {exc}")

    # ── Process uploaded file (only on new file) ──
    if uploaded_file is not None:
        if st.session_state.get('last_uploaded') != uploaded_file.name:
            try:
                df = load_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.enriched_df = None
                st.session_state.analysis_complete = False
                st.session_state.analysis_progress = 0.0
                st.session_state.last_uploaded = uploaded_file.name
            except ValueError as exc:
                st.error(f"CSV validation error: {exc}")
                # Use return instead of st.stop to not break the page layout
                st.stop()

    df: pd.DataFrame | None = st.session_state.uploaded_df

    if df is not None:
        st.markdown("---")

        # ── Column detection ──
        from utils.data_processor import REQUIRED_COLUMNS

        col_status = st.expander("Column Detection", expanded=False)
        with col_status:
            cols = st.columns(4)
            for i, col_name in enumerate(REQUIRED_COLUMNS):
                found = col_name in df.columns
                status = "Present" if found else "Missing"
                color = "#166534" if found else "#991b1b"
                cols[i % 4].markdown(
                    f"<span style='color:{color}; font-weight:600; font-size:0.85rem;'>{status}</span>"
                    f"&nbsp;&nbsp;<code>{col_name}</code>",
                    unsafe_allow_html=True,
                )

        # ── Raw data table ──
        st.markdown('<div class="section-header">Raw Supplier Feed</div>', unsafe_allow_html=True)
        st.dataframe(df, height=300, use_container_width=True)
        st.caption(f"{len(df)} products · {len(df.columns)} columns")

        # ── Pre-score preview ──
        if show_preview:
            st.markdown(
                '<div class="section-header">Pre-Analysis Preview — Local Rule-Based Scoring</div>',
                unsafe_allow_html=True,
            )
            preview_df = pre_score_dataframe(df)
            preview_cols = ["product_id", "title", "pre_score", "pre_grade", "pre_issues"]
            available_preview = [c for c in preview_cols if c in preview_df.columns]
            st.dataframe(
                _color_score_df(preview_df[available_preview].rename(columns={"pre_score": "quality_score"})),
                height=280,
                use_container_width=True,
            )

        # ── Analyse button ──
        st.markdown("")
        if not st.session_state.analysis_complete:
            analyse_clicked = st.button(
                "Analyse with AI",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.api_key,
            )
            if not st.session_state.api_key:
                st.warning("Enter your Gemini API key in the sidebar to enable AI analysis.")
        else:
            analyse_clicked = False

        # ── Analysis flow ──
        if analyse_clicked and st.session_state.api_key:
            st.session_state.analysis_complete = False
            st.session_state.enriched_df = None

            # Batch optimisation mode groups by category before batching
            working_df = df.copy()
            if batch_opt:
                working_df = working_df.sort_values("category").reset_index(drop=True)

            batches = prepare_batches(working_df, batch_size=batch_size)
            total_batches = len(batches)
            total_products = len(working_df)

            progress_bar = st.progress(0.0, text="Initialising analysis…")
            status_placeholder = st.empty()
            time_placeholder = st.empty()
            retry_placeholder = st.empty()

            start_time = time.time()

            def _progress_cb(batch_idx, t_batches, products_done, t_products):
                frac = products_done / t_products if t_products else 1.0
                elapsed = time.time() - start_time
                rate = products_done / elapsed if elapsed > 0 else 0
                remaining = int((t_products - products_done) / rate) if rate > 0 else 0
                progress_bar.progress(
                    min(frac, 1.0),
                    text=f"Processing batch {batch_idx} of {t_batches} — {products_done} products analysed",
                )
                status_placeholder.markdown(
                    f"**Batch {batch_idx} / {t_batches}** — {products_done} of {t_products} products processed"
                )
                time_placeholder.caption(f"Estimated time remaining: {remaining}s")

            def _retry_cb(msg):
                retry_placeholder.warning(f"Retrying batch — {msg[:100]}")

            with st.spinner("Running AI analysis…"):
                try:
                    all_results = run_analysis(
                        api_key=st.session_state.api_key,
                        batches=batches,
                        rate_limit_sleep=1.0,
                        progress_callback=_progress_cb,
                        retry_callback=_retry_cb,
                    )
                except Exception as exc:
                    error_str = str(exc)
                    if "429" in error_str or "rate" in error_str.lower():
                        st.error("Rate limit reached. Waiting 10 seconds before resuming.")
                        for i in range(10, 0, -1):
                            time.sleep(1)
                            status_placeholder.info(f"Resuming in {i}s…")
                    else:
                        st.error(f"Analysis failed: {exc}")
                        st.stop()

            progress_bar.progress(1.0, text="Analysis complete.")
            status_placeholder.empty()
            time_placeholder.empty()
            retry_placeholder.empty()

            enriched = merge_results(working_df, all_results)
            st.session_state.enriched_df = enriched
            st.session_state.analysis_complete = True
            st.session_state.stats = calculate_summary_stats(enriched)
            st.rerun()

        # ── Results table ──
        if st.session_state.analysis_complete and st.session_state.enriched_df is not None:
            enriched = st.session_state.enriched_df
            st.success("AI analysis complete.")
            st.markdown('<div class="section-header">Enriched Results</div>', unsafe_allow_html=True)

            display_cols = [
                "product_id", "title", "quality_score", "grade",
                "issues", "improvement_priority",
                "suggested_title", "suggested_keywords",
            ]
            available_display = [c for c in display_cols if c in enriched.columns]
            styled = _color_score_df(enriched[available_display])
            st.dataframe(styled, height=400, use_container_width=True)

            csv_bytes = generate_enriched_csv(enriched)
            st.download_button(
                label="Download Enriched CSV",
                data=csv_bytes,
                file_name="catalogiq_enriched_feed.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info("Upload a CSV file or click Load Sample Data to begin.")
        with st.expander("Required CSV columns"):
            from utils.data_processor import REQUIRED_COLUMNS
            st.code(", ".join(REQUIRED_COLUMNS))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Quality Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="hero-title" style="font-size:2rem;">Quality Dashboard</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analysis_complete or st.session_state.enriched_df is None:
        st.info("Complete an AI analysis in the Upload & Analyse tab to populate the dashboard.")
    else:

        stats = st.session_state.stats or calculate_summary_stats(st.session_state.enriched_df)
        enriched = st.session_state.enriched_df

        avg_score = stats["average_quality_score"]
        total = stats["total_products"]
        needs_review = stats["products_needing_review"]
        hours_saved = round(needs_review * 0.25, 1)
        grade_dist = stats["grade_distribution"]
        worst_5 = stats["top_5_worst_products"]
        attr_freq = stats["missing_attribute_frequency"]

        # ── Row 1: KPI cards ──
        st.markdown("### Key Metrics")
        k1, k2, k3, k4 = st.columns(4)

        def _kpi(col, number, label, delta=""):
            delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
            html = f'<div class="metric-card"><div class="metric-number">{number}</div><div class="metric-label">{label}</div>{delta_html}</div>'
            with col:
                st.markdown(html, unsafe_allow_html=True)

        _kpi(k1, f"{avg_score}", "Avg Quality Score", f"−{round(100 - avg_score, 1)} from perfect")
        _kpi(k2, str(total), "Total Products Analysed")
        _kpi(k3, str(needs_review), "Products Needing Review", "score < 60")
        _kpi(k4, f"{hours_saved}h", "Est. Hours Saved", f"{needs_review} × 0.25h manual review")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Grade distribution & attribute frequency ──
        col_grade, col_attr = st.columns(2)
        
        import altair as alt

        with col_grade:
            st.markdown("### Grade Distribution")
            grade_df = pd.DataFrame.from_dict(grade_dist, orient="index", columns=["Count"])
            grade_df.index.name = "Grade"
            grade_df = grade_df.reset_index()
            # Premium Altair Chart
            chart1 = alt.Chart(grade_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#6366f1").encode(
                x=alt.X('Grade:N', title=None, axis=alt.Axis(grid=False, labelAngle=0, labelColor="#94a3b8", domain=False, ticks=False)),
                y=alt.Y('Count:Q', title=None, axis=alt.Axis(grid=True, gridColor="#1e293b", gridDash=[2,2], labelColor="#94a3b8", domain=False, ticks=False))
            ).properties(height=280)
            st.altair_chart(chart1, use_container_width=True)

        with col_attr:
            st.markdown("### Missing Attribute Frequency")
            if attr_freq:
                attr_df = (
                    pd.DataFrame.from_dict(attr_freq, orient="index", columns=["Count"])
                    .sort_values("Count", ascending=False)
                )
                attr_df.index.name = "Attribute"
                attr_df = attr_df.reset_index()
                # Premium Altair Chart
                chart2 = alt.Chart(attr_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#38bdf8").encode(
                    x=alt.X('Attribute:N', title=None, sort='-y', axis=alt.Axis(grid=False, labelAngle=-45, labelColor="#94a3b8", domain=False, ticks=False)),
                    y=alt.Y('Count:Q', title=None, axis=alt.Axis(grid=True, gridColor="#1e293b", gridDash=[2,2], labelColor="#94a3b8", domain=False, ticks=False))
                ).properties(height=280)
                st.altair_chart(chart2, use_container_width=True)
            else:
                st.success("No missing attributes detected across the feed.")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Row 3: Top 5 worst products ──
        st.markdown("### Lowest-Scoring Products")
        if worst_5:
            worst_df = pd.DataFrame(worst_5)
            display_worst = [c for c in ["product_id", "title", "quality_score", "grade", "top_issue"]
                             if c in worst_df.columns]
            st.dataframe(_color_score_df(worst_df[display_worst]), use_container_width=True, hide_index=True)
        else:
            st.info("No product data available.")

        st.markdown("---")

        # ── Row 4: Title comparison ──
        st.markdown("### Original vs. AI-Suggested Titles")
        if "suggested_title" in enriched.columns:
            title_compare = enriched[["product_id", "title", "suggested_title", "quality_score"]].copy()
            title_compare.columns = ["Product ID", "Original Title", "AI Suggested Title", "Score"]
            title_compare = title_compare[title_compare["AI Suggested Title"].str.strip() != ""]
            st.dataframe(
                _color_score_df(title_compare.rename(columns={"Score": "quality_score"})),
                use_container_width=True,
                height=350,
                hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Before / After
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        '<div class="hero-title" style="font-size:2rem;">Before / After</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Select a product to compare the original supplier data with AI-enriched output.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analysis_complete or st.session_state.enriched_df is None:
        st.info("Complete an AI analysis in the Upload & Analyse tab to use this view.")
    else:

        enriched = st.session_state.enriched_df

        # Build selector options
        if "title" in enriched.columns:
            options = [f"{row['product_id']} — {row['title']}" for _, row in enriched.iterrows()]
        else:
            options = enriched["product_id"].tolist()

        selected_label = st.selectbox("Select a product", options, key="before_after_selector")
        selected_id = selected_label.split(" — ")[0].strip()

        row_series = enriched[enriched["product_id"].astype(str) == str(selected_id)]
        if row_series.empty:
            st.warning("Product not found.")
            st.stop()

        row = row_series.iloc[0].to_dict()

        # ── Two-column layout ──
        left_col, right_col = st.columns(2, gap="large")

        # ── LEFT: Raw feed card ──
        with left_col:
            st.markdown("#### Original Supplier Data")
            original_fields = [
                "product_id", "title", "category", "material", "colour",
                "description", "price", "gender", "size_range", "season",
                "keywords", "stock_status",
            ]

            html_fields = ""
            for field in original_fields:
                val = row.get(field, "")
                is_empty = not str(val).strip() or str(val).strip() in ("nan", "0", "0.0")
                badge = (
                    " <span style='color:#ef4444; font-size:0.7rem; font-weight:600; "
                    "background:#fee2e2; padding:0.1rem 0.4rem; border-radius:4px;'>MISSING</span>"
                    if is_empty else ""
                )
                display_val = str(val) if str(val) not in ("nan", "") else "<em style='color:#cbd5e1;'>—</em>"
                html_fields += f"""
                <div style='margin-bottom:0.6rem;'>
                  <span style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; color:#94a3b8; font-weight:600;'>{field}</span>{badge}<br>
                  <span style='font-size:0.92rem; color:#cbd5e1;'>{display_val}</span>
                </div>"""

            st.markdown(
                f'<div class="raw-card">{html_fields}</div>',
                unsafe_allow_html=True,
            )

        # ── RIGHT: AI enriched card ──
        with right_col:
            st.markdown("#### AI-Enriched Data")

            score = int(row.get("quality_score", 0))
            grade = str(row.get("grade", "F"))
            score_class = _score_class(score)
            priority = str(row.get("improvement_priority", "High"))
            priority_class = _priority_class(priority)

            st.markdown(
                f"""<div style='text-align:center; margin-bottom:1.25rem;'>
                      <div class='score-circle {score_class}'>{score}</div>
                      <span class='grade-badge grade-{grade}'>Grade {grade}</span>
                      &nbsp;
                      <span class='{priority_class}'>{priority} Priority</span>
                    </div>""",
                unsafe_allow_html=True,
            )

            # Issues list
            issues_raw = str(row.get("issues", ""))
            if issues_raw and issues_raw.lower() not in ("nan", "none", ""):
                issue_list = [i.strip() for i in issues_raw.split(";") if i.strip()]
                issues_html = "".join(
                    f"<div class='issue-item'><span class='issue-bullet'></span>{issue}</div>"
                    for issue in issue_list
                )
                st.markdown(
                    f"<div style='margin-bottom:1rem;'><strong>Issues Detected:</strong>{issues_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("**No issues detected.**")

            # Enriched fields
            enriched_fields = {
                "Suggested Title": row.get("suggested_title", ""),
                "Suggested Description": row.get("suggested_description", ""),
                "Suggested Keywords": row.get("suggested_keywords", ""),
                "Missing Attributes": row.get("missing_attributes", ""),
            }

            html_enriched = ""
            for label, val in enriched_fields.items():
                v = str(val).strip() if str(val) not in ("nan", "") else "—"
                html_enriched += f"""
                <div style='margin-bottom:0.9rem;'>
                  <span style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; color:#818cf8; font-weight:700;'>{label}</span><br>
                  <span style='font-size:0.9rem; color:#e2e8f0; line-height:1.5;'>{v}</span>
                </div>"""

            st.markdown(
                f'<div class="enriched-card">{html_enriched}</div>',
                unsafe_allow_html=True,
            )
