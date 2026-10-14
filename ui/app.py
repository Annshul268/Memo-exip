"""
MEMO-EXI: Research-Grade Adaptive Temporal Memory Management Interface.
Minimal, modern, editorial UI built with Streamlit.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path regardless of execution entry point
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.agent_pipeline import TemporalMemoryAgentPipeline
from memory.schema import MemoryStatus, QueryTemporalIntent

# Page configuration
st.set_page_config(
    page_title="MEMO-EXI | Adaptive Temporal Memory",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined Minimal Editorial Theme CSS
st.markdown("""
<style>
    /* Global Fonts & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #191919;
    }
    
    /* Top title & headers */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: #111111;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F6F6F2;
        border-right: 1px solid #E6E6DF;
        padding-top: 1.5rem;
    }
    
    /* Subtle Cards & Containers */
    .memo-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E5DF;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .memo-metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E5DF;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .memo-metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2C4A6F;
        line-height: 1.2;
    }
    
    .memo-metric-lbl {
        font-size: 0.8rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-current { background-color: #EBF5EE; color: #2D6A4F; border: 1px solid #D8EEDF; }
    .badge-superseded { background-color: #F1F3F5; color: #495057; border: 1px solid #E2E6EA; }
    .badge-disputed { background-color: #FFF3BF; color: #D9480F; border: 1px solid #FFE066; }
    
    /* Clean inputs & buttons */
    div.stButton > button:first-child {
        background-color: #2C4A6F;
        color: #FFFFFF;
        border: 1px solid #2C4A6F;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.45rem 1.2rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E3550;
        border-color: #1E3550;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Cache the core pipeline instance across Streamlit reruns."""
    return TemporalMemoryAgentPipeline(
        sqlite_path="data/temporal_memory.db",
        chroma_path="data/chroma_db"
    )


pipeline = get_pipeline()

# Sidebar Navigation & System Status
st.sidebar.markdown("### MEMO-EXI")
st.sidebar.caption("Adaptive Temporal Memory Management for LLM Agents")

# Status indicator
status_color = "#2D6A4F" if pipeline.llm.is_mock else "#2C4A6F"
provider_label = "Mock LLM (Deterministic)" if pipeline.llm.is_mock else f"Provider: {pipeline.llm.provider_name}"
st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem; font-size:0.85rem; color:#444;">
        <span style="height:9px; width:9px; background-color:{status_color}; border-radius:50%; display:inline-block;"></span>
        <span>{provider_label}</span>
    </div>
    """,
    unsafe_allow_html=True
)

app_section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Memory Chat",
        "Memory Explorer",
        "Context Inspector",
        "Evaluation"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype • SQLite source-of-truth • ChromaDB vector index")


# ==============================================================================
# SECTION 1: Overview
# ==============================================================================
if app_section == "Overview":
    st.title("MEMO-EXI")
    st.markdown("#### Adaptive Temporal Memory Management for LLM Agents")
    st.write(
        "A minimal, research-grade external memory architecture designed to preserve version lineages, "
        "prevent stale-fact hallucinations, handle conflicting facts safely, and pack relevant memories into strict context budgets."
    )

    # Real backend data queries
    all_records = pipeline.sqlite.list_records()
    current_count = sum(1 for r in all_records if r.status == MemoryStatus.CURRENT)
    superseded_count = sum(1 for r in all_records if r.status == MemoryStatus.SUPERSEDED)
    disputed_count = sum(1 for r in all_records if r.status == MemoryStatus.DISPUTED)
    vector_count = pipeline.vector.collection.count()

    # Backend Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="memo-metric-box">
            <div class="memo-metric-val">{len(all_records)}</div>
            <div class="memo-metric-lbl">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="memo-metric-box">
            <div class="memo-metric-val">{current_count}</div>
            <div class="memo-metric-lbl">Current Facts</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="memo-metric-box">
            <div class="memo-metric-val">{superseded_count}</div>
            <div class="memo-metric-lbl">Superseded Versions</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="memo-metric-box">
            <div class="memo-metric-val">{vector_count}</div>
            <div class="memo-metric-lbl">Indexed Vectors</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="memo-metric-box">
            <div class="memo-metric-val">{pipeline.default_budget_tokens}</div>
            <div class="memo-metric-lbl">Budget (Tokens)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture and Temporal Logic Explainer
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### How Temporal Memory Works")
        st.markdown("""
        Standard Retrieval-Augmented Generation (RAG) retrieves memories based solely on semantic vector similarity.
        When a user changes a fact (e.g., *“my deadline changed from Nov 10 to Nov 20”*), vector search frequently retrieves
        both facts or mistakenly favors the older one, causing catastrophic hallucinations.
        
        **MEMO-EXI resolves this through four research contributions:**
        
        1. **Bitemporal Relational Model:** Separates *system transaction time* (when recorded) from *real-world validity time* (`valid_from` to `valid_until`).
        2. **Safe, Non-Destructive Superseding:** Updating a fact closes the previous version's validity interval and links it via `supersedes_id`. Historical versions remain permanently queryable.
        3. **Query Temporal Intent Classification:** Evaluates whether queries target *current* facts, *historical* past facts, point-in-time intervals, or chronological *timelines*.
        4. **Utility-per-Token Knapsack Selection:** Ranks candidates by combined semantic, temporal, and importance utility, penalizing redundant versions to fit within strict token budgets.
        """)

    with col_right:
        st.markdown("### System Architecture & Pipeline")
        st.markdown("""
        ```
        User Input (Conversation / Structured Fact)
                            │
                            ▼
               Temporal Update Manager
              ┌─────────────┴─────────────┐
              ▼                           ▼
        SQLite Store                 ChromaDB
        (Authoritative Truth,        (Dense Semantic
        Lineage & Validity)           Retrieval Index)
              └─────────────┬─────────────┘
                            ▼
              Temporal Intent Classifier
                            │
                            ▼
             Context-Budget Knapsack Selector
            (Utility Scoring & Redundancy Penalty)
                            │
                            ▼
            Grounded LLM Generation (or Mock)
        ```
        """)


# ==============================================================================
# SECTION 2: Memory Chat
# ==============================================================================
elif app_section == "Memory Chat":
    st.title("Memory Chat")
    st.write("Converse with the agent or ingest facts. Observe how updated facts preserve history and respond to current vs historical questions.")

    # Reset Data with confirmation
    with st.expander("Storage Management / Reset Demo Data"):
        st.warning("Resetting will wipe all stored memories from both SQLite and ChromaDB.")
        confirm_reset = st.checkbox("I confirm I want to reset all demo memories", value=False)
        if st.button("Purge Memory Storage", disabled=not confirm_reset):
            pipeline.reset()
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.success("Memory storage purged. All databases reset.")
            st.rerun()

    # Two columns: Fact Ingestion on left, Chat & Query on right
    c_ingest, c_chat = st.columns([1, 1.2])

    with c_ingest:
        st.markdown("### 📥 Ingest Information")
        ingest_mode = st.radio("Ingestion Mode", ["Natural Language Statement", "Structured Fact"], horizontal=True)

        if ingest_mode == "Natural Language Statement":
            with st.form("nl_form"):
                st.caption("Examples: 'My deadline is November 20', 'The project manager is Alice'")
                stmt = st.text_input("Natural Language Assertion", value="My deadline is November 20")
                submit_stmt = st.form_submit_button("Ingest Statement", use_container_width=True)
                if submit_stmt and stmt:
                    res = pipeline.ingest_natural_statement(stmt)
                    rec = res["record"]
                    st.success(f"Ingested! Action: `{res['action']}` | Subject: `{rec.subject}`, Attribute: `{rec.attribute}` -> `{rec.value}`")

        else:
            with st.form("struct_form"):
                subj = st.text_input("Subject Entity", value="user", placeholder="e.g. user, project, client")
                attr = st.text_input("Attribute / Relation", value="deadline", placeholder="e.g. deadline, supervisor")
                val = st.text_input("Factual Value", value="November 20", placeholder="e.g. November 20, Hall B")
                
                c_conf, c_imp = st.columns(2)
                with c_conf:
                    conf = st.slider("Confidence", 0.1, 1.0, 1.0, 0.05)
                with c_imp:
                    imp = st.slider("Importance", 0.1, 1.0, 0.8, 0.05)

                explicit_corr = st.checkbox("Explicit confirmed update / correction", value=True)
                submit_fact = st.form_submit_button("Commit Structured Fact", use_container_width=True)

                if submit_fact and subj and attr and val:
                    res = pipeline.ingest_fact(
                        subject=subj,
                        attribute=attr,
                        value=val,
                        confidence=conf,
                        importance=imp,
                        is_explicit_correction=explicit_corr
                    )
                    st.success(f"Committed! Action: `{res['action']}` | Status: `{res['status']}`")

    with c_chat:
        st.markdown("### ❓ Query the Agent")
        
        # Budget control
        chat_budget = st.slider("Context Token Budget", min_value=128, max_value=2048, value=512, step=64)

        # Predefined test questions for convenience
        st.caption("Quick sample queries:")
        q_cols = st.columns(3)
        sample_q = None
        if q_cols[0].button("Current Deadline", use_container_width=True):
            sample_q = "What is my current deadline now?"
        if q_cols[1].button("Historical Deadline", use_container_width=True):
            sample_q = "What was my previous deadline before?"
        if q_cols[2].button("Timeline History", use_container_width=True):
            sample_q = "What is the history of my deadline changes?"

        # Chat input
        query_input = st.text_input("Ask a question:", value=sample_q if sample_q else "", placeholder="e.g. What is my current deadline now?")
        submit_q = st.button("Submit Query", use_container_width=True)

        if submit_q and query_input:
            with st.spinner("Executing hybrid retrieval & temporal synthesis..."):
                ans = pipeline.query(user_query=query_input, budget_tokens=chat_budget)

            st.markdown(f"**Detected Intent:** `{ans['temporal_intent'].upper()}` | **Memories Used:** {ans['selected_count']} / {ans['candidate_count']}")
            
            # Agent response card
            st.markdown(f"""
            <div class="memo-card">
                <div style="font-size:0.8rem; color:#666; margin-bottom:0.25rem;">AGENT RESPONSE ({ans['provider'].upper()})</div>
                <div style="font-size:1.05rem; line-height:1.5;">{ans['response']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Inspection of used memories
            with st.expander("Inspect Grounded Context (Memories Injected into Prompt)"):
                sel_res = ans["selection_result"]
                st.caption(f"Used tokens: {ans['used_tokens']} / {ans['budget_tokens']} | Compliant: {ans['budget_respected']}")
                if sel_res.selected_memories:
                    for idx, cand in enumerate(sel_res.selected_memories, 1):
                        st.markdown(
                            f"**{idx}. {cand.record.subject} → {cand.record.attribute}:** `{cand.record.value}`  \n"
                            f"Status: `{cand.record.status.value.upper()}` | Sim: `{cand.semantic_similarity:.2f}` | "
                            f"Temp: `{cand.temporal_relevance:.2f}` | Utility: `{cand.utility_score:.2f}` | Tokens: `{cand.estimated_tokens}`"
                        )
                else:
                    st.info("No memories selected for this prompt.")


# ==============================================================================
# SECTION 3: Memory Explorer
# ==============================================================================
elif app_section == "Memory Explorer":
    st.title("Memory Explorer")
    st.write("Authoritative SQLite storage inspection. Review persistent records, validity intervals, and chronological version chains.")

    c_filt1, c_filt2, c_filt3 = st.columns([1, 1, 1])
    with c_filt1:
        status_filter = st.selectbox("Status Filter", ["ALL", "current", "superseded", "disputed"])
    with c_filt2:
        subject_filter = st.text_input("Filter by Subject Entity", value="")
    with c_filt3:
        if st.button("Refresh Storage View", use_container_width=True):
            st.rerun()

    status_arg = None if status_filter == "ALL" else MemoryStatus(status_filter)
    subj_arg = subject_filter.strip() if subject_filter.strip() else None

    records = pipeline.sqlite.list_records(status=status_arg, subject=subj_arg)

    if records:
        rows = []
        for r in records:
            rows.append({
                "ID": r.id,
                "Subject": r.subject,
                "Attribute": r.attribute,
                "Value": r.value,
                "Status": r.status.value,
                "Valid From": r.valid_from.strftime("%Y-%m-%d %H:%M:%S"),
                "Valid Until": r.valid_until.strftime("%Y-%m-%d %H:%M:%S") if r.valid_until else "Active (None)",
                "Supersedes ID": r.supersedes_id or "—",
                "Confidence": round(r.confidence, 2),
                "Importance": round(r.importance, 2),
                "Accesses": r.access_count
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No records match the active filter criteria.")

    st.markdown("---")
    st.markdown("### 📜 Version Lineage & Timeline Inspection")
    st.caption("Trace how a specific fact evolved chronologically across updates.")

    col_l1, col_l2, col_l3 = st.columns([1, 1, 0.8])
    with col_l1:
        trace_subj = st.text_input("Subject", value="user", key="trace_s")
    with col_l2:
        trace_attr = st.text_input("Attribute", value="deadline", key="trace_a")
    with col_l3:
        st.write("")
        st.write("")
        trace_btn = st.button("Trace Version Chain", use_container_width=True)

    if trace_btn or trace_subj:
        history = pipeline.engine.get_fact_lineage(trace_subj, trace_attr)
        if history:
            st.markdown(f"Found **{len(history)}** versions in lineage for `{trace_subj}` → `{trace_attr}`:")
            for idx, h in enumerate(history, 1):
                badge_class = "badge-current" if h.status == MemoryStatus.CURRENT else "badge-superseded"
                if h.status == MemoryStatus.DISPUTED:
                    badge_class = "badge-disputed"
                
                st.markdown(f"""
                <div class="memo-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <strong>Version {idx}: {h.subject} → {h.attribute}</strong>
                        <span class="badge {badge_class}">{h.status.value}</span>
                    </div>
                    <div style="font-size:1.1rem; font-weight:600; color:#111;">Value: "{h.value}"</div>
                    <div style="font-size:0.85rem; color:#666; margin-top:0.35rem;">
                        <span>Valid From: <code>{h.valid_from.isoformat()}</code></span> | 
                        <span>Valid Until: <code>{h.valid_until.isoformat() if h.valid_until else 'Indefinite (Active)'}</code></span> | 
                        <span>Supersedes: <code>{h.supersedes_id or 'None (Root)'}</code></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No version history recorded for `{trace_subj}` → `{trace_attr}`.")


# ==============================================================================
# SECTION 4: Context Inspector
# ==============================================================================
elif app_section == "Context Inspector":
    st.title("Context Inspector")
    st.write(
        "Inspect the utility scoring, redundancy penalty, and token-bounded knapsack selection "
        "pipeline for any query under context budget constraints."
    )

    test_q = st.text_input("Test Query for Inspection:", value="What is my current deadline now?")
    budget_setting = st.slider("Context Token Budget", min_value=100, max_value=1500, value=300, step=50)

    if st.button("Analyze Selection Process", use_container_width=True):
        candidates = pipeline.retriever.retrieve(test_q, top_k=10)
        selection = pipeline.budget_manager.select_memories(candidates, budget_tokens=budget_setting)

        # Overview Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates Retrieved", len(candidates))
        c2.metric("Selected Memories", len(selection.selected_memories))
        c3.metric("Tokens Consumed", f"{selection.used_tokens} / {selection.total_budget_tokens}")
        c4.metric("Budget Compliant", "YES" if selection.budget_respected else "NO")

        # Selected Table
        st.markdown("### Selected Memories (Injected into Prompt)")
        if selection.selected_memories:
            s_rows = [{
                "Subject": m.record.subject,
                "Attribute": m.record.attribute,
                "Value": m.record.value,
                "Status": m.record.status.value,
                "Sim Score": round(m.semantic_similarity, 3),
                "Temp Score": round(m.temporal_relevance, 3),
                "Utility Score": round(m.utility_score, 3),
                "Tokens": m.estimated_tokens
            } for m in selection.selected_memories]
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No candidates selected under current budget.")

        # Rejected Table
        st.markdown("### Rejected Candidates")
        if selection.rejected_memories:
            r_rows = [{
                "Subject": m.record.subject,
                "Attribute": m.record.attribute,
                "Value": m.record.value,
                "Status": m.record.status.value,
                "Utility Score": round(m.utility_score, 3),
                "Tokens": m.estimated_tokens,
                "Rejection Reason": m.rejection_reason
            } for m in selection.rejected_memories]
            st.dataframe(pd.DataFrame(r_rows), use_container_width=True, hide_index=True)
        else:
            st.success("All candidates fit within the token budget without rejection.")


# ==============================================================================
# SECTION 5: Evaluation
# ==============================================================================
elif app_section == "Evaluation":
    st.title("Evaluation")
    st.write(
        "Empirical evaluation results generated from actual benchmark runs comparing "
        "MEMO-EXI against standard memory baselines."
    )

    results_path = Path("results/benchmark_results.json")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("Run / Re-run Benchmark Suite", use_container_width=True):
            with st.spinner("Executing 10 scenarios across 5 baselines..."):
                from evaluation.runner import run_benchmark
                run_benchmark(output_dir="results", budget_tokens=500)
            st.success("Benchmark completed! Reloading results...")
            st.rerun()

    if not results_path.exists():
        st.info("No benchmark results found at `results/benchmark_results.json`. Click the button above to run the suite.")
    else:
        with open(results_path, "r", encoding="utf-8") as f:
            bench = json.load(f)

        meta = bench.get("metadata", {})
        summaries = bench.get("baseline_summaries", {})
        raw_runs = bench.get("raw_scenario_runs", [])

        st.caption(f"Run Timestamp: `{meta.get('timestamp')}` | Scenarios: `{meta.get('scenario_count')}` | Budget: `{meta.get('budget_tokens')} tokens`")

        # Summary Table
        sum_rows = []
        for b_name, data in summaries.items():
            sum_rows.append({
                "Baseline": b_name,
                "Overall Acc (%)": data.get("overall_accuracy_pct"),
                "Current Acc (%)": data.get("current_fact_accuracy_pct"),
                "Historical Acc (%)": data.get("historical_fact_accuracy_pct"),
                "Stale Error (%)": data.get("stale_fact_error_rate_pct"),
                "Avg Tokens": data.get("avg_tokens_per_query"),
                "Budget Compliant (%)": data.get("budget_compliance_pct"),
                "Retrieval Lat (ms)": data.get("avg_retrieval_latency_ms")
            })

        summary_df = pd.DataFrame(sum_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_acc, c_tok = st.columns(2)

        with c_acc:
            st.markdown("#### Accuracy Comparison (%)")
            chart_acc = summary_df.set_index("Baseline")[["Overall Acc (%)", "Current Acc (%)", "Historical Acc (%)"]]
            st.bar_chart(chart_acc)

        with c_tok:
            st.markdown("#### Token Consumption & Stale-Fact Error")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.caption("Avg Input Tokens (Lower is better)")
                st.bar_chart(summary_df.set_index("Baseline")["Avg Tokens"])
            with col_t2:
                st.caption("Stale-Fact Error % (Lower is better)")
                st.bar_chart(summary_df.set_index("Baseline")["Stale Error (%)"])

        # Detailed Raw Scenario Breakdown
        with st.expander("Detailed Scenario-by-Scenario Evaluation Log"):
            if raw_runs:
                raw_df = pd.DataFrame([{
                    "Baseline": r["baseline"],
                    "Scenario": r["scenario_id"],
                    "Query": r["query"],
                    "Expected": r["expected"],
                    "Correct": "✅" if r["is_correct"] else "❌",
                    "Stale Error": "⚠️ YES" if r["is_stale_error"] else "NO",
                    "Tokens": r["used_tokens"]
                } for r in raw_runs])
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
