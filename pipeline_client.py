"""
Pipeline Client — Metaflow artifact loader for the Streamlit dashboard.

Provides read-only access to Metaflow flow artifacts:
- Load trained RF models from FraudTrainingFlow
- List past runs with metrics
- Load scoring results from FraudScoringFlow

Works identically with local datastore (.metaflow/) and
Outerbounds K8s (S3-backed metadata service).
"""

from datetime import datetime
from typing import Optional
import traceback


def _safe_import_metaflow():
    """Check if metaflow is importable.
    Import outerbounds first (if available) so it can configure
    the metadata service URL for deployed Outerbounds apps.
    """
    try:
        try:
            import outerbounds  # noqa: F401 — triggers Metaflow config injection
        except ImportError:
            pass
        from metaflow import Flow, Run, namespace  # noqa: F401
        # Use global namespace so we can see all runs regardless of who ran them
        namespace(None)
        return True
    except ImportError:
        return False


METAFLOW_AVAILABLE = _safe_import_metaflow()


# ============================================================================
# Training runs
# ============================================================================

def list_training_runs(max_runs: int = 20) -> list[dict]:
    """
    List completed FraudTrainingFlow runs with metrics.

    Returns list of dicts:
        run_id, created_at, best_f1, best_auc, best_hparams, status
    """
    if not METAFLOW_AVAILABLE:
        return []
    from metaflow import Flow

    results = []
    try:
        flow = Flow("FraudTrainingFlow")
    except Exception:
        return []

    for run in list(flow.runs())[:max_runs]:
        entry = {
            "run_id": run.id,
            "created_at": str(run.created_at),
            "status": "unknown",
            "best_f1": None,
            "best_auc": None,
            "best_hparams": None,
        }
        try:
            if run.successful:
                end_data = run["end"].task.data
                metrics = end_data.best_metrics
                entry["status"] = "completed"
                entry["best_f1"] = metrics.get("f1")
                entry["best_auc"] = metrics.get("auc_roc")
                entry["best_hparams"] = end_data.best_hparams
            else:
                entry["status"] = "failed"
        except Exception:
            entry["status"] = "incomplete"
        results.append(entry)
    return results


def load_model(run_id: str) -> dict:
    """
    Load a trained model from a specific FraudTrainingFlow run.

    Returns dict with keys:
        model, metrics, hparams, model_config, feature_names,
        test_data (X_test, y_test, desc_test, amt_test)
    """
    if not METAFLOW_AVAILABLE:
        return {"error": "metaflow not installed"}
    from metaflow import Flow

    try:
        run = Flow("FraudTrainingFlow")[run_id]
        end_data = run["end"].task.data

        result = {
            "model": end_data.best_model,
            "metrics": end_data.best_metrics,
            "hparams": end_data.best_hparams,
            "model_config": end_data.model_config,
            "run_id": run_id,
            "created_at": str(run.created_at),
        }

        # Also grab test data for in-dashboard evaluation
        try:
            result["X_test"] = end_data.X_test
            result["y_test"] = end_data.y_test
            result["desc_test"] = end_data.desc_test
            result["amt_test"] = end_data.amt_test
            result["feature_names"] = list(end_data.X_test.columns)
        except Exception:
            pass  # test data may not be on end step in all variants

        return result

    except Exception as e:
        return {"error": f"Failed to load run {run_id}: {e}"}


def load_latest_model() -> dict:
    """
    Load the most recent successful FraudTrainingFlow model.
    """
    runs = list_training_runs(max_runs=10)
    for run in runs:
        if run["status"] == "completed":
            return load_model(run["run_id"])
    return {"error": "No completed training runs found"}


# ============================================================================
# Data prep runs
# ============================================================================

def list_data_prep_runs(max_runs: int = 20) -> list[dict]:
    """
    List completed FraudDataPrepFlow runs.

    Returns list of dicts:
        run_id, created_at, train_size, test_size, fraud_rate, status
    """
    if not METAFLOW_AVAILABLE:
        return []
    from metaflow import Flow

    results = []
    try:
        flow = Flow("FraudDataPrepFlow")
    except Exception:
        return []

    for run in list(flow.runs())[:max_runs]:
        entry = {
            "run_id": run.id,
            "created_at": str(run.created_at),
            "status": "unknown",
            "train_size": None,
            "test_size": None,
        }
        try:
            if run.successful:
                end_data = run["end"].task.data
                entry["status"] = "completed"
                entry["train_size"] = len(end_data.X_train)
                entry["test_size"] = len(end_data.X_test)
                entry["train_fraud"] = int(end_data.y_train.sum())
                entry["test_fraud"] = int(end_data.y_test.sum())
            else:
                entry["status"] = "failed"
        except Exception:
            entry["status"] = "incomplete"
        results.append(entry)
    return results


# ============================================================================
# Scoring runs
# ============================================================================

def list_scoring_runs(max_runs: int = 20) -> list[dict]:
    """
    List completed FraudScoringFlow runs.
    """
    if not METAFLOW_AVAILABLE:
        return []
    from metaflow import Flow

    results = []
    try:
        flow = Flow("FraudScoringFlow")
    except Exception:
        return []

    for run in list(flow.runs())[:max_runs]:
        entry = {
            "run_id": run.id,
            "created_at": str(run.created_at),
            "status": "unknown",
        }
        try:
            if run.successful:
                end_data = run["summarize"].task.data
                entry["status"] = "completed"
                entry["decisions"] = end_data.decisions
                entry["results_df"] = end_data.results_df
            else:
                entry["status"] = "failed"
        except Exception:
            entry["status"] = "incomplete"
        results.append(entry)
    return results


def load_scoring_results(run_id: str) -> dict:
    """Load full scoring results from a FraudScoringFlow run."""
    if not METAFLOW_AVAILABLE:
        return {"error": "metaflow not installed"}
    from metaflow import Flow

    try:
        run = Flow("FraudScoringFlow")[run_id]
        summarize_data = run["summarize"].task.data
        return {
            "results_df": summarize_data.results_df,
            "decisions": summarize_data.decisions,
            "run_id": run_id,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Flow execution (subprocess)
# ============================================================================

import subprocess
import sys
import os


def _resolve_flows_dir(flows_dir: str = "flows") -> str:
    """Resolve flows directory relative to project root or as absolute path."""
    if os.path.isabs(flows_dir):
        return flows_dir
    # Try common locations
    candidates = [
        flows_dir,
        os.path.join(os.path.dirname(__file__), "..", flows_dir),
        os.path.expanduser(f"~/Developer/fraud-detection-ob/{flows_dir}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return flows_dir


def trigger_flow(
    flow_name: str,
    params: dict | None = None,
    mode: str = "local",
    flows_dir: str = "flows",
) -> dict:
    """
    Trigger a Metaflow flow as a subprocess.

    Args:
        flow_name: One of "data_prep", "training", "scoring"
        params: Flow parameters (e.g., {"data_run_id": "123"})
        mode: "local" or "kubernetes"
        flows_dir: Path to flows directory

    Returns:
        dict with "process" (Popen object) or "error"
    """
    flow_map = {
        "data_prep": {
            "local": "data_prep_flow_local.py",
            "kubernetes": "data_prep_flow.py",
        },
        "training": {
            "local": "training_flow_local.py",
            "kubernetes": "training_flow.py",
        },
        "scoring": {
            "local": "scoring_flow_local.py",
            "kubernetes": "scoring_flow.py",
        },
    }

    if flow_name not in flow_map:
        return {"error": f"Unknown flow: {flow_name}"}

    filename = flow_map[flow_name].get(mode, flow_map[flow_name]["local"])
    resolved_dir = _resolve_flows_dir(flows_dir)
    flow_path = os.path.join(resolved_dir, filename)

    if not os.path.isfile(flow_path):
        return {"error": f"Flow file not found: {flow_path}"}

    cmd = [sys.executable, flow_path]

    # For K8s mode, add --with kubernetes (only for non-local)
    if mode == "kubernetes":
        cmd.extend(["--with", "kubernetes"])

    cmd.append("run")

    # Add parameters
    if params:
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])

    try:
        # Set working directory to the project root (parent of flows/)
        cwd = os.path.dirname(resolved_dir)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
        )
        return {
            "process": process,
            "cmd": " ".join(cmd),
            "cwd": cwd,
            "flow_name": flow_name,
            "mode": mode,
        }
    except Exception as e:
        return {"error": f"Failed to start flow: {e}"}


def poll_flow(process_info: dict) -> dict:
    """
    Check if a running flow has completed.

    Returns dict with:
        running: bool
        returncode: int | None
        output: str (stdout so far)
    """
    proc = process_info.get("process")
    if proc is None:
        return {"running": False, "error": "No process"}

    returncode = proc.poll()
    if returncode is None:
        # Still running — read available output without blocking
        return {"running": True, "output": ""}
    else:
        stdout, _ = proc.communicate()
        return {
            "running": False,
            "returncode": returncode,
            "output": stdout or "",
            "success": returncode == 0,
        }


# ============================================================================
# Hybrid scoring helper
# ============================================================================

import numpy as np


def hybrid_score(
    model,
    features,
    merchant_description: str,
    amount: float,
    llm_fn,
    thresholds: dict | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Two-stage scoring: RF first, LLM only for borderline cases.

    Args:
        model: Fitted sklearn model with predict_proba
        features: numpy array or DataFrame row (30 features)
        merchant_description: Merchant text for LLM analysis
        amount: Transaction amount
        llm_fn: Callable(prompt, system_prompt) -> str (the dashboard's ask_model)
        thresholds: dict with block, review, llm_trigger
        weights: dict with rf, llm (must sum to 1.0)

    Returns:
        dict with rf_score, llm_score (if called), combined_score, decision, stage
    """
    import time
    import pandas as pd

    thresholds = thresholds or {"block": 0.8, "review": 0.5, "llm_trigger": 0.3}
    weights = weights or {"rf": 0.6, "llm": 0.4}

    # Reshape features and wrap in DataFrame with correct column names
    # to avoid sklearn "X does not have valid feature names" warning
    if hasattr(features, "values"):
        features = features.values
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Build DataFrame with feature names the model was trained on
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    if feature_names and features.shape[1] == len(feature_names):
        features = pd.DataFrame(features, columns=feature_names)

    # Stage 1: RF prediction
    t0 = time.perf_counter()
    rf_score = float(model.predict_proba(features)[:, 1][0])
    rf_elapsed = time.perf_counter() - t0

    result = {
        "rf_score": round(rf_score, 4),
        "rf_elapsed_ms": round(rf_elapsed * 1000, 2),
        "llm_score": None,
        "llm_elapsed_ms": None,
        "llm_error": None,
        "combined_score": round(rf_score, 4),
        "stage": "rf_only",
    }

    # Stage 2: LLM for borderline cases (>= trigger, < block)
    if rf_score >= thresholds["llm_trigger"] and rf_score < thresholds["block"]:
        try:
            t0 = time.perf_counter()
            llm_prompt = (
                f"Analyze this transaction for fraud risk. "
                f"Respond with ONLY a number between 0.0 and 1.0.\n"
                f"Merchant: {merchant_description}\n"
                f"Amount: ${amount:.2f}\n"
                f"Fraud risk score:"
            )
            llm_response = llm_fn(
                llm_prompt,
                system_prompt=(
                    "You are a fraud detection expert. Respond with ONLY "
                    "a single decimal number between 0.0 and 1.0."
                ),
            )
            llm_elapsed = time.perf_counter() - t0

            # Parse numeric score from LLM response
            import re

            numbers = re.findall(r"\d+\.?\d*", llm_response)
            llm_score = float(numbers[0]) if numbers else 0.5
            llm_score = min(max(llm_score, 0.0), 1.0)

            combined = weights["rf"] * rf_score + weights["llm"] * llm_score

            result["llm_score"] = round(llm_score, 4)
            result["llm_elapsed_ms"] = round(llm_elapsed * 1000, 2)
            result["combined_score"] = round(combined, 4)
            result["stage"] = "hybrid"

        except Exception as e:
            # LLM failed — fall back to RF-only but surface the error
            result["stage"] = "rf_only_llm_failed"
            result["llm_error"] = f"{type(e).__name__}: {e}"

    # Decision
    score = result["combined_score"]
    if score >= thresholds["block"]:
        result["decision"] = "BLOCK"
    elif score >= thresholds["review"]:
        result["decision"] = "REVIEW"
    else:
        result["decision"] = "APPROVE"

    return result