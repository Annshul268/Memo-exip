"""
Interactive Streamlit Application for Memo-exip.
Demonstrates temporal memory management, version lineage, context inspection,
and empirical benchmark results.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from memory.agent_pipeline import TemporalMemoryAgentPipeline
from memory.schema import MemoryStatus

# Page configuration
st.set_page_config(
    page_title="Memo-exip | Adaptive Temporal Memory",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_pipeline():
    """Cache the core pipeline instance across Streamlit reruns."""
    return TemporalMemoryAgentPipeline(
        sqlite_path="data/temporal_memory.db",
        chroma_path="data/chroma_db"
    )


pipeline = get_pipeline()

# Sidebar Navigation & System Status
st.sidebar.title("⏳ Memo-exip")
st.sidebar.caption("Adaptive Temporal Memory for LLM Agents")

# Status badges
mode_badge = "🟢 MOCK LLM (Deterministic)" if pipeline.llm.is_mock else f"⚡ REAL LLM ({pipeline.llm.provider_name})"
st.sidebar.info(f"**Runtime Provider:** {mode_badge}")

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat & Fact Demo",
        "🗂️ Memory Explorer",
        "🔍 Context Inspector",
        "📊 Evaluation Dashboard",
        "⚙️ System Settings"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Research Project:**  
    Adaptive Temporal Memory Management for LLM Agents Enhancement  
    *Option A: Continual External Memory*
    """
)


# ==============================================================================
# SECTION A: Chat & Fact Demo
# ==============================================================================
if app_mode == "💬 Chat & Fact Demo":
    st.header("💬 Conversation & Temporal Memory Demonstration")
    st.write(
        "Interact with the agent or ingest facts. Watch how changing facts update the "
        "current state while strictly preserving historical versions for retroactive queries."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Ingest Fact or Natural Statement")
        with st.form("ingest_form"):
            in_subject = st.text_input("Subject Entity", value="user", placeholder="e.g. user, project, client")
            in_attribute = st.text_input("Attribute", value="deadline", placeholder="e.g. deadline, address, supervisor")
            in_value = st.text_input("Factual Value", value="November 20", placeholder="e.g. November 20, Room 204")
            
            c_a, c_b = st.columns(2)
            with c_a:
                in_conf = st.slider("Confidence", 0.1, 1.0, 1.0, 0.05)
            with c_b:
                in_imp = st.slider("Importance", 0.1, 1.0, 0.8, 0.05)
                
            in_correction = st.checkbox("Confirmed explicit update / correction", value=True)
            submit_fact = st.form_submit_button("Ingest Fact", use_container_width=True)

            if submit_fact:
                if in_subject and in_attribute and in_value:
                    res = pipeline.ingest_fact(
                        subject=in_subject,
                        attribute=in_attribute,
                        value=in_value,
                        confidence=in_conf,
                        importance=in_imp,
                        is_explicit_correction=in_correction
                    )
                    st.success(f"Fact Ingested! Action: `{res['action']}` | Status: `{res['status']}`")
                else:
                    st.error("Please fill in Subject, Attribute, and Value.")

    with col2:
        st.subheader("❓ Ask the Agent")
        with st.form("query_form"):
            user_q = st.text_input(
                "Query",
                value="What is my current deadline?",
                placeholder="e.g. What is my current deadline? or What was my previous deadline before?"
            )
            q_budget = st.slider("Token Budget", min_value=128, max_value=2048, value=512, step=64)
            submit_q = st.form_submit_button("Submit Query", use_container_width=True)

            if submit_q and user_q:
                with st.spinner("Processing temporal retrieval and synthesizing response..."):
                    agent_res = pipeline.query(user_query=user_q, budget_tokens=q_budget)

                st.markdown(f"**Temporal Intent Detected:** `{agent_res['temporal_intent'].upper()}`")
                st.markdown(f"**Agent Response:**")
                st.info(agent_res["response"])

                st.markdown(
                    f"**Context Usage:** {agent_res['used_tokens']} / {agent_res['budget_tokens']} tokens "
                    f"({'✅ Within Budget' if agent_res['budget_respected'] else '❌ Budget Exceeded'})"
                )

                # Show quick retrieved summary
                with st.expander("Retrieved Context Items"):
                    for m in agent_res["selection_result"].selected_memories:
                        st.markdown(f"- `{m.record.status.value.upper()}`: {m.record.subject} -> {m.record.attribute}: **{m.record.value}** (Sim: {m.semantic_similarity:.2f}, Temp: {m.temporal_relevance:.2f})")


# ==============================================================================
# SECTION B: Memory Explorer
# ==============================================================================
elif app_mode == "🗂️ Memory Explorer":
    st.header("🗂️ Authoritative Memory Explorer (SQLite Store)")
    st.write("Inspect all persistent memory records, historical versions, and bitemporal metadata.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        filter_status = st.selectbox("Filter Status", ["ALL", "current", "superseded", "disputed"])
    with c2:
        filter_subject = st.text_input("Filter by Subject", value="")
    with c3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    status_arg = None if filter_status == "ALL" else MemoryStatus(filter_status)
    subj_arg = filter_subject.strip() if filter_subject.strip() else None

    records = pipeline.sqlite.list_records(status=status_arg, subject=subj_arg)

    if records:
        data_rows = []
        for r in records:
            data_rows.append({
                "ID": r.id,
                "Subject": r.subject,
                "Attribute": r.attribute,
                "Value": r.value,
                "Status": r.status.value,
                "Valid From": r.valid_from.strftime("%Y-%m-%d %H:%M"),
                "Valid Until": r.valid_until.strftime("%Y-%m-%d %H:%M") if r.valid_until else "Active (None)",
                "Supersedes ID": r.supersedes_id or "-",
                "Confidence": r.confidence,
                "Importance": r.importance,
                "Access Count": r.access_count
            })
        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No memories found matching the specified filters.")

    st.subheader("📜 Fact Version Lineage")
    col_hist_s, col_hist_a = st.columns(2)
    with col_hist_s:
        hist_s = st.text_input("Entity Subject", value="user", key="hist_s")
    with col_hist_a:
        hist_a = st.text_input("Entity Attribute", value="deadline", key="hist_a")

    if st.button("Trace Lineage Chain"):
        history = pipeline.engine.get_fact_lineage(hist_s, hist_a)
        if history:
            st.markdown(f"Found **{len(history)}** versions in lineage for `{hist_s}` -> `{hist_a}`:")
            for idx, h in enumerate(history, 1):
                badge = "🟢 CURRENT" if h.status == MemoryStatus.CURRENT else "🕒 SUPERSEDED"
                st.markdown(
                    f"**Version {idx}** [{badge}]  \n"
                    f"- Value: `{h.value}`  \n"
                    f"- Valid From: `{h.valid_from.isoformat()}` | Valid Until: `{h.valid_until.isoformat() if h.valid_until else 'Active'}`  \n"
                    f"- Record ID: `{h.id}` (Parent: `{h.supersedes_id or 'None'}`)"
                )
        else:
            st.warning("No lineage history found for this entity.")


# ==============================================================================
# SECTION C: Context Inspector
# ==============================================================================
elif app_mode == "🔍 Context Inspector":
    st.header("🔍 Context Budget & Memory Selection Inspector")
    st.write(
        "Inspect the utility scoring, redundancy penalty, and greedy knapsack selection "
        "algorithm under token budget pressure."
    )

    sample_query = st.text_input("Inspect query selection for:", value="What is my current deadline now?")
    budget_limit = st.slider("Context Token Budget", min_value=100, max_value=1500, value=300, step=50)

    if st.button("Run Inspection Analysis", use_container_width=True):
        candidates = pipeline.retriever.retrieve(sample_query, top_k=10)
        selection = pipeline.budget_manager.select_memories(candidates, budget_tokens=budget_limit)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates Retrieved", len(candidates))
        c2.metric("Selected Memories", len(selection.selected_memories))
        c3.metric("Tokens Used", f"{selection.used_tokens} / {selection.total_budget_tokens}")
        c4.metric("Budget Respected", "YES ✅" if selection.budget_respected else "NO ❌")

        st.subheader("✅ Selected Memories (Injected into Prompt)")
        if selection.selected_memories:
            sel_rows = [{
                "Subject": m.record.subject,
                "Attribute": m.record.attribute,
                "Value": m.record.value,
                "Status": m.record.status.value,
                "Sim Score": round(m.semantic_similarity, 3),
                "Temp Score": round(m.temporal_relevance, 3),
                "Utility Score": round(m.utility_score, 3),
                "Est. Tokens": m.estimated_tokens
            } for m in selection.selected_memories]
            st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No memories selected.")

        st.subheader("❌ Rejected Candidates")
        if selection.rejected_memories:
            rej_rows = [{
                "Subject": m.record.subject,
                "Attribute": m.record.attribute,
                "Value": m.record.value,
                "Status": m.record.status.value,
                "Utility Score": round(m.utility_score, 3),
                "Est. Tokens": m.estimated_tokens,
                "Rejection Reason": m.rejection_reason
            } for m in selection.rejected_memories]
            st.dataframe(pd.DataFrame(rej_rows), use_container_width=True, hide_index=True)
        else:
            st.success("All candidates fit within the token budget without rejection!")


# ==============================================================================
# SECTION D: Evaluation Dashboard
# ==============================================================================
elif app_mode == "📊 Evaluation Dashboard":
    st.header("📊 Empirical Benchmark Results & Baseline Comparison")
    st.write(
        "Results populated exclusively from actual experimental runs recorded in "
        "`results/benchmark_results.json`."
    )

    results_file = Path("results/benchmark_results.json")
    if not results_file.exists():
        st.warning("No benchmark output found at `results/benchmark_results.json`. Run the benchmark suite to generate data.")
        if st.button("Run Benchmark Suite Now"):
            with st.spinner("Executing benchmark across all 5 baselines..."):
                from evaluation.runner import run_benchmark
                run_benchmark()
            st.success("Benchmark completed! Reloading dashboard...")
            st.rerun()
    else:
        with open(results_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        summaries = bench_data.get("baseline_summaries", {})
        meta = bench_data.get("metadata", {})

        st.caption(f"Benchmark executed at: `{meta.get('timestamp')}` | Total Scenarios: `{meta.get('scenario_count')}` | Budget: `{meta.get('budget_tokens')} tokens`")

        # Summary Table
        summary_rows = []
        for b_name, metrics in summaries.items():
            summary_rows.append({
                "Baseline": b_name,
                "Overall Acc (%)": metrics.get("overall_accuracy_pct"),
                "Current Acc (%)": metrics.get("current_fact_accuracy_pct"),
                "Historical Acc (%)": metrics.get("historical_fact_accuracy_pct"),
                "Stale Error (%)": metrics.get("stale_fact_error_rate_pct"),
                "Avg Tokens": metrics.get("avg_tokens_per_query"),
                "Budget Compliant (%)": metrics.get("budget_compliance_pct"),
                "Retrieval Lat (ms)": metrics.get("avg_retrieval_latency_ms")
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Charts
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.subheader("Accuracy Comparison (%)")
            chart_data = summary_df.set_index("Baseline")[["Overall Acc (%)", "Current Acc (%)", "Historical Acc (%)"]]
            st.bar_chart(chart_data)

        with c_chart2:
            st.subheader("Token Consumption & Stale Error Rate")
            col_tok, col_stale = st.columns(2)
            with col_tok:
                st.caption("Avg Input Tokens per Query (Lower is more efficient)")
                st.bar_chart(summary_df.set_index("Baseline")["Avg Tokens"])
            with col_stale:
                st.caption("Stale-Fact Error Rate % (Lower is better)")
                st.bar_chart(summary_df.set_index("Baseline")["Stale Error (%)"])


# ==============================================================================
# SECTION E: Settings
# ==============================================================================
elif app_mode == "⚙️ System Settings":
    st.header("⚙️ System Configuration & Diagnostics")
    
    st.subheader("Storage Status")
    st.markdown(f"- **SQLite DB Path:** `{pipeline.sqlite.db_path.resolve()}`")
    st.markdown(f"- **ChromaDB Path:** `{pipeline.vector.persist_dir.resolve()}`")
    st.markdown(f"- **Total Stored Records in SQLite:** `{len(pipeline.sqlite.list_records())}`")
    st.markdown(f"- **Total Indexed Records in ChromaDB:** `{pipeline.vector.collection.count()}`")

    st.subheader("Reconciliation")
    st.write("Ensure ChromaDB vector index is in sync with authoritative SQLite store:")
    if st.button("Run Storage Reconciliation"):
        stats = pipeline.engine.reconcile_storage()
        st.success(f"Reconciliation Complete: {stats}")

    st.subheader("Active Provider Configuration")
    provider = pipeline.llm.provider_name
    st.text_input("Active LLM Backend", value=provider, disabled=True)
    st.text_input("Environment LLM_PROVIDER", value=os.getenv("LLM_PROVIDER", "mock"), disabled=True)
