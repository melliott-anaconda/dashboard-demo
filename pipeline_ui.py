"""
ML Pipeline Integration for app.py

This module provides:
1. sidebar_model_status()  — shows loaded model info in the sidebar
2. render_pipeline_tab()   — full Pipeline Manager tab
3. hybrid_fraud_screening() — RF+LLM scoring for existing fraud screening tab

Integration into app.py:
  1. Add `from pipeline_ui import sidebar_model_status, render_pipeline_tab` at top
  2. Call `sidebar_model_status()` in the sidebar section (after inference config)
  3. Add "ML Pipeline" to the tabs list and call `render_pipeline_tab()` in it
  4. In the fraud screening tab, add hybrid scoring toggle

Requires: pipeline_client.py in the same directory or src/
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import timedelta

# Lazy import — pipeline_client may be in src/ or root
try:
    from pipeline_client import (
        list_training_runs, load_model, load_latest_model,
        list_data_prep_runs, list_scoring_runs, load_scoring_results,
        trigger_flow, poll_flow, hybrid_score, METAFLOW_AVAILABLE,
    )
except ImportError:
    try:
        from src.pipeline_client import (
            list_training_runs, load_model, load_latest_model,
            list_data_prep_runs, list_scoring_runs, load_scoring_results,
            trigger_flow, poll_flow, hybrid_score, METAFLOW_AVAILABLE,
        )
    except ImportError:
        METAFLOW_AVAILABLE = False


# ============================================================================
# Session state defaults
# ============================================================================

def _init_pipeline_state():
    defaults = {
        "rf_model": None,
        "rf_metrics": None,
        "rf_hparams": None,
        "rf_model_config": None,
        "rf_run_id": None,
        "rf_feature_names": None,
        "pipeline_process": None,
        "pipeline_log": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_pipeline_state()


# ============================================================================
# Sidebar: Model Status
# ============================================================================

def sidebar_model_status():
    """
    Render model status in the sidebar. Call this inside `with st.sidebar:`.
    Shows loaded model metrics and a Load button.
    """
    _init_pipeline_state()

    if not METAFLOW_AVAILABLE:
        st.caption("ML Pipeline: `metaflow` not installed")
        return

    st.markdown("---")
    st.subheader("ML Model")

    if st.session_state.rf_model is not None:
        m = st.session_state.rf_metrics
        st.success(f"✅ Model loaded (run {st.session_state.rf_run_id})")
        c1, c2 = st.columns(2)
        c1.metric("F1", f"{m['f1']:.4f}")
        c2.metric("AUC", f"{m['auc_roc']:.4f}")
        hp = st.session_state.rf_hparams
        st.caption(
            f"Random Forest · {hp['n_estimators']} trees, depth {hp['max_depth']}"
        )
    else:
        st.warning("No ML model loaded")

    col_load, col_pick = st.columns(2)
    with col_load:
        if st.button("Load Latest", key="load_latest_model"):
            with st.spinner("Loading model..."):
                result = load_latest_model()
            if "error" in result:
                st.error(result["error"])
            else:
                _store_model(result)
                st.rerun()

    with col_pick:
        runs = list_training_runs(max_runs=5)
        completed = [r for r in runs if r["status"] == "completed"]
        if completed:
            labels = [
                f"{r['run_id']} (F1={r['best_f1']:.3f})"
                for r in completed
            ]
            sel = st.selectbox(
                "Run", range(len(labels)),
                format_func=lambda i: labels[i],
                key="model_run_selector",
                label_visibility="collapsed",
            )
            if st.button("Load", key="load_selected_model"):
                with st.spinner("Loading..."):
                    result = load_model(completed[sel]["run_id"])
                if "error" in result:
                    st.error(result["error"])
                else:
                    _store_model(result)
                    st.rerun()


def _store_model(result: dict):
    """Store loaded model artifacts in session state."""
    st.session_state.rf_model = result["model"]
    st.session_state.rf_metrics = result["metrics"]
    st.session_state.rf_hparams = result["hparams"]
    st.session_state.rf_model_config = result["model_config"]
    st.session_state.rf_run_id = result["run_id"]
    if "feature_names" in result:
        st.session_state.rf_feature_names = result["feature_names"]


# ============================================================================
# Pipeline Manager Tab
# ============================================================================

def render_pipeline_tab(
    pipeline_mode: str = "local",
    flows_dir: str = "flows",
):
    """
    Render the full Pipeline Manager tab content.

    Args:
        pipeline_mode: "local" or "kubernetes"
        flows_dir: path to the flows directory
    """
    if not METAFLOW_AVAILABLE:
        st.error(
            "`metaflow` is not installed in this environment. "
            "Install with `conda install metaflow` or `pip install metaflow`."
        )
        return

    # Ensure session state is initialized (may not have run at import time)
    _init_pipeline_state()

    st.subheader("ML Pipeline Manager")
    st.caption(f"Mode: **{pipeline_mode}** | Flows: `{flows_dir}`")

    ptabs = st.tabs([
        "📊 Data Prep",
        "🎯 Training",
        "📈 Scoring",
        "📋 Run History",
    ])

    # --- Data Prep ---
    with ptabs[0]:
        _render_data_prep(pipeline_mode, flows_dir)

    # --- Training ---
    with ptabs[1]:
        _render_training(pipeline_mode, flows_dir)

    # --- Scoring ---
    with ptabs[2]:
        _render_scoring(pipeline_mode, flows_dir)

    # --- Run History ---
    with ptabs[3]:
        _render_run_history()

    # --- Active process monitor ---
    if st.session_state.get("pipeline_process") is not None:
        _render_process_monitor()


# ---- Data Prep sub-tab ----

def _render_data_prep(mode, flows_dir):
    st.markdown("**Prepare training data from creditcard.csv**")
    st.markdown(
        "Loads the Kaggle dataset, adds merchant descriptions, "
        "splits into train/test with stratified sampling."
    )

    runs = list_data_prep_runs(max_runs=5)
    if runs:
        st.markdown("**Recent runs:**")
        for r in runs:
            if r["status"] == "completed":
                st.caption(
                    f"✅ `{r['run_id']}` — "
                    f"Train: {r['train_size']:,} ({r.get('train_fraud', '?')} fraud), "
                    f"Test: {r['test_size']:,} ({r.get('test_fraud', '?')} fraud)"
                )
            else:
                st.caption(f"{'❌' if r['status'] == 'failed' else '⏳'} `{r['run_id']}` — {r['status']}")

    if st.button("▶️ Run Data Prep", key="run_data_prep"):
        result = trigger_flow("data_prep", mode=mode, flows_dir=flows_dir)
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.pipeline_process = result
            st.info(f"Started: `{result['cmd']}`")
            st.rerun()


# ---- Training sub-tab ----

def _render_training(mode, flows_dir):
    st.markdown(
        "**Hyperparameter grid search** — 4 Random Forest configurations, "
        "parallel training, selects best by F1 score."
    )

    # Select data prep run
    data_runs = list_data_prep_runs(max_runs=10)
    completed_data = [r for r in data_runs if r["status"] == "completed"]

    if not completed_data:
        st.warning("No completed data prep runs. Run Data Prep first.")
        return

    data_labels = [
        f"{r['run_id']} — Train {r['train_size']:,}"
        for r in completed_data
    ]
    sel_idx = st.selectbox(
        "Data Prep Run",
        range(len(data_labels)),
        format_func=lambda i: data_labels[i],
        key="training_data_run",
    )
    data_run_id = completed_data[sel_idx]["run_id"]

    # Show recent training runs
    training_runs = list_training_runs(max_runs=5)
    if training_runs:
        st.markdown("**Recent training runs:**")
        for r in training_runs:
            if r["status"] == "completed":
                st.caption(
                    f"✅ `{r['run_id']}` — "
                    f"F1={r['best_f1']:.4f}, AUC={r['best_auc']:.4f} "
                    f"| {r['best_hparams']}"
                )
            else:
                st.caption(f"{'❌' if r['status'] == 'failed' else '⏳'} `{r['run_id']}` — {r['status']}")

    if st.button("▶️ Run Training", key="run_training"):
        result = trigger_flow(
            "training",
            params={"data_run_id": data_run_id},
            mode=mode,
            flows_dir=flows_dir,
        )
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.pipeline_process = result
            st.info(f"Started: `{result['cmd']}`")
            st.rerun()


# ---- Scoring sub-tab ----

def _render_scoring(mode, flows_dir):
    st.markdown(
        "**Batch scoring** — loads trained model, scores 100 synthetic "
        "transactions, applies APPROVE/REVIEW/BLOCK thresholds."
    )

    training_runs = list_training_runs(max_runs=10)
    completed_training = [r for r in training_runs if r["status"] == "completed"]

    if not completed_training:
        st.warning("No completed training runs. Run Training first.")
        return

    training_labels = [
        f"{r['run_id']} — F1={r['best_f1']:.4f}"
        for r in completed_training
    ]
    sel_idx = st.selectbox(
        "Training Run",
        range(len(training_labels)),
        format_func=lambda i: training_labels[i],
        key="scoring_training_run",
    )
    training_run_id = completed_training[sel_idx]["run_id"]

    if st.button("▶️ Run Scoring", key="run_scoring"):
        result = trigger_flow(
            "scoring",
            params={"training_run_id": training_run_id},
            mode=mode,
            flows_dir=flows_dir,
        )
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.pipeline_process = result
            st.info(f"Started: `{result['cmd']}`")
            st.rerun()

    # Show past scoring results
    scoring_runs = list_scoring_runs(max_runs=5)
    completed_scoring = [r for r in scoring_runs if r["status"] == "completed"]
    if completed_scoring:
        st.markdown("**Recent scoring results:**")
        for r in completed_scoring:
            decisions = r.get("decisions", [])
            if decisions:
                from collections import Counter
                counts = Counter(decisions)
                st.caption(
                    f"✅ `{r['run_id']}` — "
                    f"APPROVE: {counts.get('APPROVE', 0)}, "
                    f"REVIEW: {counts.get('REVIEW', 0)}, "
                    f"BLOCK: {counts.get('BLOCK', 0)}"
                )


# ---- Run History sub-tab ----

def _render_run_history():
    st.markdown("**All pipeline runs across all flows**")

    data = []
    for r in list_data_prep_runs(max_runs=10):
        data.append({
            "Flow": "DataPrep",
            "Run ID": r["run_id"],
            "Status": r["status"],
            "Created": r["created_at"][:19],
            "Detail": f"Train: {r.get('train_size', '?'):,}" if r["status"] == "completed" else "",
        })
    for r in list_training_runs(max_runs=10):
        data.append({
            "Flow": "Training",
            "Run ID": r["run_id"],
            "Status": r["status"],
            "Created": r["created_at"][:19],
            "Detail": f"F1={r.get('best_f1', '?')}" if r["status"] == "completed" else "",
        })
    for r in list_scoring_runs(max_runs=10):
        data.append({
            "Flow": "Scoring",
            "Run ID": r["run_id"],
            "Status": r["status"],
            "Created": r["created_at"][:19],
            "Detail": "",
        })

    if data:
        df = pd.DataFrame(data)
        df = df.sort_values("Created", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline runs found. Run Data Prep to start.")


# ---- Process Monitor (fragment for live updates) ----

def _render_process_monitor():
    """Show running flow status. Placed at bottom of pipeline tab."""
    proc_info = st.session_state.get("pipeline_process")
    if proc_info is None:
        return

    st.markdown("---")
    st.markdown(f"**Running:** `{proc_info.get('flow_name', '?')}` ({proc_info.get('mode', '?')})")

    status = poll_flow(proc_info)

    if status.get("running"):
        st.info("⏳ Flow is running... (refresh to check status)")
        if st.button("🔄 Check Status", key="check_flow_status"):
            st.rerun()
    else:
        if status.get("success"):
            st.success("✅ Flow completed successfully")
            with st.expander("Output"):
                st.code(status.get("output", ""), language="text")
        else:
            st.error(f"❌ Flow failed (exit code {status.get('returncode')})")
            with st.expander("Output"):
                st.code(status.get("output", ""), language="text")

        # Clear process
        st.session_state.pipeline_process = None


# ============================================================================
# Hybrid Fraud Screening (for integration into existing fraud screening tab)
# ============================================================================

def render_hybrid_toggle():
    """
    Render a toggle for LLM-only vs Hybrid (RF+LLM) scoring.
    Call at top of the fraud screening tab.
    Returns True if hybrid mode is selected AND a model is loaded.
    """
    _init_pipeline_state()

    if st.session_state.get("rf_model") is None:
        load_err = st.session_state.get("_pipeline_load_error")
        if load_err:
            st.caption(f"⚠️ ML model failed to load: {load_err}")
        else:
            st.caption("💡 Load an ML model in the sidebar to enable hybrid RF+LLM scoring")
        return False

    mode = st.radio(
        "Scoring Mode",
        ["LLM Only", "Hybrid (Random Forest + LLM)"],
        key="scoring_mode_toggle",
        horizontal=True,
    )
    if mode == "Hybrid (Random Forest + LLM)":
        m = st.session_state.rf_metrics
        hp = st.session_state.rf_hparams
        st.caption(
            f"Random Forest ({hp['n_estimators']} trees, depth {hp['max_depth']}) · "
            f"F1={m['f1']:.4f}, AUC={m['auc_roc']:.4f} · "
            f"RF screens all → LLM called for borderline (score ≥0.3, <0.8)"
        )
        return True
    return False


def score_transaction_hybrid(features_array, merchant, amount, ask_model_fn):
    """
    Score a single transaction using the loaded RF model + LLM fallback.

    Args:
        features_array: numpy array of 30 features (V1-V28, Time, Amount)
        merchant: merchant description string
        amount: transaction amount
        ask_model_fn: the dashboard's ask_model() function

    Returns:
        dict with rf_score, llm_score, combined_score, decision, stage
    """
    _init_pipeline_state()

    if st.session_state.get("rf_model") is None:
        return {"error": "No model loaded"}

    thresholds = st.session_state.get("rf_model_config") or {
        "llm_threshold": 0.3,
        "block_threshold": 0.8,
        "review_threshold": 0.5,
    }

    return hybrid_score(
        model=st.session_state.rf_model,
        features=features_array,
        merchant_description=merchant,
        amount=amount,
        llm_fn=ask_model_fn,
        thresholds={
            "block": thresholds.get("block_threshold", 0.8),
            "review": thresholds.get("review_threshold", 0.5),
            "llm_trigger": thresholds.get("llm_threshold", 0.3),
        },
        weights={"rf": 0.6, "llm": 0.4},
    )