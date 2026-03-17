"""
dashboard/app.py
==================
Streamlit web dashboard for the RAG Benchmarking Framework.

Features:
  - Interactive filterable experiment results table
  - Accuracy & metric comparison bar charts (Plotly)
  - Latency comparison chart
  - Per-metric breakdown
  - Auto-refresh every 30 seconds

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Make project root importable when run from project dir ────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results_storage.database import ExperimentDatabase

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Benchmark Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  h1 { color: #4f8ef7; }
  h2, h3 { color: #2dd4bf; }
  .metric-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }
  .stDataFrame { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_data(db_path: str) -> pd.DataFrame:
    """Load experiments from SQLite with 30-second cache TTL."""
    db = ExperimentDatabase(db_path)
    return db.get_all()


def main():
    st.title("🔬 RAG Benchmarking Dashboard")
    st.caption("Automatically comparing RAG pipeline techniques | Auto-refreshes every 30s")

    # ── Database path ──────────────────────────────────────────────────────────
    default_db = str(ROOT / "results_storage" / "experiments.db")
    db_path = st.sidebar.text_input("Database path", value=default_db)

    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    # ── Load data ────────────────────────────────────────────────────────────
    df_raw = load_data(db_path)

    if df_raw.empty:
        st.warning(
            "No experiment results found. Run experiments first:\n"
            "```\npython main.py run --config config/experiment_config.yaml\n```"
        )
        _show_empty_state()
        return

    # ── Sidebar Filters ────────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")
    df = df_raw.copy()

    def multiselect_filter(label, col):
        options = sorted(df[col].dropna().unique().tolist())
        if len(options) > 1:
            sel = st.sidebar.multiselect(label, options, default=options)
            return df[df[col].isin(sel)]
        return df

    df = multiselect_filter("OCR Engine", "ocr_engine")
    df = multiselect_filter("Chunking Strategy", "chunking_strategy")
    df = df[df["chunk_size"].isin(
        st.sidebar.multiselect(
            "Chunk Size",
            sorted(df["chunk_size"].dropna().unique().tolist()),
            default=sorted(df["chunk_size"].dropna().unique().tolist()),
        )
    )] if "chunk_size" in df.columns and not df["chunk_size"].dropna().empty else df
    df = multiselect_filter("Embedding Model", "embedding_model")
    df = multiselect_filter("Vector DB", "vector_db")
    df = multiselect_filter("Retrieval Strategy", "retrieval_strategy")
    df = multiselect_filter("LLM Model", "llm_model")
    df = df[df["status"] == "completed"] if "status" in df.columns else df

    # ── KPI Row ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    def _safe_mean(series):
        return round(series.dropna().mean(), 3) if not series.dropna().empty else "N/A"

    col1.metric("Total Experiments", len(df))
    col2.metric("Avg Faithfulness",      _safe_mean(df.get("faithfulness", pd.Series(dtype=float))))
    col3.metric("Avg Answer Relevancy",  _safe_mean(df.get("answer_relevancy", pd.Series(dtype=float))))
    col4.metric("Avg Context Precision", _safe_mean(df.get("context_precision", pd.Series(dtype=float))))
    col5.metric("Avg Total Latency (s)", _safe_mean(df.get("total_latency_s", pd.Series(dtype=float))))

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Results Table",
        "🎯 Accuracy Comparison",
        "⚡ Latency Analysis",
        "📈 Metrics Deep Dive",
        "🏆 Model Rankings",
    ])

    with tab1:
        _tab_results_table(df)

    with tab2:
        _tab_accuracy(df)

    with tab3:
        _tab_latency(df)

    with tab4:
        _tab_metrics(df)

    with tab5:
        _tab_rankings(df)

    # ── Auto-refresh ───────────────────────────────────────────────────────────
    time.sleep(1)
    st.sidebar.info("Dashboard auto-refreshes every 30 seconds (cached).")


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_results_table(df: pd.DataFrame):
    st.markdown("### Experiment Results")

    display_cols = [
        "experiment_id", "timestamp",
        "ocr_engine", "chunking_strategy", "chunk_size",
        "embedding_model", "vector_db", "retrieval_strategy", "llm_model",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "precision_at_k", "recall_at_k", "total_latency_s", "status",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇ Download Full Results CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="experiment_results.csv",
                mime="text/csv",
            )


def _tab_accuracy(df: pd.DataFrame):
    st.markdown("### Accuracy Comparison")

    metric_col = st.selectbox(
        "Metric to compare",
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
         "precision_at_k", "recall_at_k"],
        key="accuracy_metric",
    )
    group_by = st.selectbox(
        "Group by",
        ["llm_model", "embedding_model", "retrieval_strategy",
         "chunking_strategy", "vector_db", "ocr_engine"],
        key="accuracy_group",
    )

    if metric_col not in df.columns or df[metric_col].dropna().empty:
        st.info(f"No data available for metric: {metric_col}")
        return

    agg = df.groupby(group_by)[metric_col].mean().reset_index().sort_values(metric_col, ascending=False)
    fig = px.bar(
        agg,
        x=group_by,
        y=metric_col,
        color=group_by,
        title=f"Mean {metric_col.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: embedding vs vector_db
    if "embedding_model" in df.columns and "vector_db" in df.columns and metric_col in df.columns:
        pivot = df.pivot_table(
            values=metric_col,
            index="embedding_model",
            columns="vector_db",
            aggfunc="mean",
        )
        if not pivot.empty:
            st.markdown("#### Embedding × Vector DB Heatmap")
            fig2 = px.imshow(
                pivot,
                text_auto=".3f",
                color_continuous_scale="Blues",
                title=f"{metric_col.replace('_', ' ').title()} Heatmap",
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)


def _tab_latency(df: pd.DataFrame):
    st.markdown("### Latency Analysis")

    latency_cols = [c for c in ["retrieval_latency_s", "generation_latency_s", "total_latency_s"]
                    if c in df.columns]

    if not latency_cols:
        st.info("No latency data available.")
        return

    group_by = st.selectbox(
        "Group by",
        ["llm_model", "embedding_model", "retrieval_strategy", "chunking_strategy", "vector_db"],
        key="lat_group",
    )

    lat_agg = df.groupby(group_by)[latency_cols].mean().reset_index()
    fig = px.bar(
        lat_agg.melt(id_vars=group_by, value_vars=latency_cols, var_name="latency_type", value_name="seconds"),
        x=group_by,
        y="seconds",
        color="latency_type",
        barmode="group",
        title=f"Mean Latency (s) by {group_by.replace('_', ' ').title()}",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Box plot for total latency distribution
    if "total_latency_s" in df.columns:
        st.markdown("#### Total Latency Distribution")
        fig2 = px.box(
            df, x=group_by, y="total_latency_s",
            color=group_by,
            title="Total Latency Distribution",
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)


def _tab_metrics(df: pd.DataFrame):
    st.markdown("### RAG Metrics Deep Dive")

    rag_metrics = [c for c in ["faithfulness", "answer_relevancy",
                                "context_precision", "context_recall"]
                   if c in df.columns]

    if not rag_metrics:
        st.info("No RAG metric data available.")
        return

    group_by = st.selectbox(
        "Compare by",
        ["llm_model", "retrieval_strategy", "embedding_model", "chunking_strategy"],
        key="metric_group",
    )

    groups = df[group_by].dropna().unique().tolist()
    if len(groups) == 0:
        return

    fig = go.Figure()
    for g in groups:
        sub = df[df[group_by] == g]
        means = [sub[m].mean() for m in rag_metrics]
        fig.add_trace(go.Scatterpolar(
            r=means,
            theta=[m.replace("_", " ").title() for m in rag_metrics],
            fill="toself",
            name=str(g),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"RAG Metrics Radar by {group_by.replace('_', ' ').title()}",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_rankings(df: pd.DataFrame):
    st.markdown("### 🏆 Model Rankings")

    score_cols = [c for c in [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "precision_at_k", "recall_at_k"
    ] if c in df.columns]

    if not score_cols:
        st.info("No scoring metrics available.")
        return

    rank_by = st.selectbox("Rank by technique", [
        "llm_model", "embedding_model", "retrieval_strategy",
        "chunking_strategy", "vector_db",
    ], key="rank_group")

    agg = df.groupby(rank_by)[score_cols].mean().reset_index()
    agg["composite_score"] = agg[score_cols].mean(axis=1)
    agg = agg.sort_values("composite_score", ascending=False).reset_index(drop=True)
    agg.index += 1  # 1-based ranking

    st.dataframe(
        agg.style.background_gradient(subset=score_cols + ["composite_score"], cmap="Greens"),
        use_container_width=True,
    )

    fig = px.bar(
        agg,
        x=rank_by,
        y="composite_score",
        color="composite_score",
        color_continuous_scale="Viridis",
        title=f"Composite Score by {rank_by.replace('_', ' ').title()}",
        text_auto=".3f",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _show_empty_state():
    st.markdown("""
    ---
    ### Getting Started

    1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    2. **Add documents** to `data/documents/` (PDF or TXT)

    3. **Run a quick experiment:**
    ```bash
    python main.py run --config config/experiment_config.yaml
    ```

    4. **Come back here** — the dashboard will automatically load results!
    """)


if __name__ == "__main__":
    main()
