"""
AI-Powered Financial Transaction Analysis — Streamlit Dashboard
Anaconda AI Catalyst — Local & Hosted Inference Demo

Converted from Jupyter notebook. Uses Anaconda Desktop API to manage
local model availability and server lifecycle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from openai import OpenAI

# ML Pipeline integration (Metaflow)
try:
    from pipeline_ui import (
        sidebar_model_status,
        render_pipeline_tab,
        render_hybrid_toggle,
        score_transaction_hybrid,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ============================================================
# Config: load config.yml from same directory as app.py
# ============================================================
_config_path = Path(__file__).parent / "config.yml"
_config = {}
if _config_path.exists():
    import yaml  # PyYAML — included in conda base
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f) or {}

DEV_MODE = _config.get("mode", "dev").lower() != "prod"

# Pipeline config
_pipeline_cfg = _config.get("pipeline", {})
PIPELINE_MODE = _pipeline_cfg.get("mode", "local")
FLOWS_DIR = _pipeline_cfg.get("flows_dir", "flows")
AUTO_LOAD_MODEL = _pipeline_cfg.get("auto_load_model", False)

# Streaming config (persisted to config.yml)
_streaming_cfg = _config.get("streaming", {})
_STREAM_TOTAL = _streaming_cfg.get("total_transactions", 200)
_STREAM_BATCH_SIZE = _streaming_cfg.get("batch_size", 5)
_STREAM_BATCH_DELAY = _streaming_cfg.get("batch_delay", 2)


def _debug(*args, **kwargs):
    """Print only in dev mode."""
    if DEV_MODE:
        print(*args, **kwargs)

# Persisted endpoint config — written by Lock button in dev mode, read in both modes
_PERSISTED_CLOUD_URL = _config.get(
    "cloud_url",
    "https://demo.se.sb.anacondaconnect.com/api/ai/inference/serve/50a1ae4f-4db9-4b93-bed4-288a46f92ff3",
)
_PERSISTED_CLOUD_KEY = _config.get("cloud_api_key", "")


def _save_config(**overrides):
    """Merge overrides into config.yml and write back.
    Also updates the in-memory _config so subsequent saves don't clobber changes.
    """
    import yaml
    global _config
    merged = dict(_config)
    merged.update(overrides)
    with open(_config_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
    _config = merged

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Transaction Monitoring — Anaconda AI Catalyst",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Anaconda Desktop API configuration
# ============================================================
DESKTOP_API_BASE = "http://localhost:8001"
DESKTOP_API_KEY = "94397214-521a-4b59-8b58-f0deedd7c47b"
TARGET_MODEL_SEARCH_TERMS = ["llama-3.1-8b-instruct", "llama-3.1-8b", "meta-llama-3.1-8b"]
TARGET_MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-q4_k_m"
TARGET_QUANTIZATION = "q4_k_m"
LOCAL_SERVER_PORT = 8080

DESKTOP_HEADERS = {
    "Authorization": f"Bearer {DESKTOP_API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# Helper: Anaconda Desktop API calls
# ============================================================
def _handle_response(r):
    """Return parsed JSON on success, or an error dict with status + body on failure."""
    try:
        body = r.json()
    except Exception:
        body = r.text
    if r.ok:
        return body
    return {
        "error": f"HTTP {r.status_code}",
        "status_code": r.status_code,
        "detail": body,
        "url": r.url,
    }


def desktop_api_get(path, params=None):
    """GET request to Anaconda Desktop API."""
    try:
        r = requests.get(
            f"{DESKTOP_API_BASE}{path}",
            headers=DESKTOP_HEADERS,
            params=params,
            timeout=10,
        )
        return _handle_response(r)
    except requests.ConnectionError:
        return {"error": "Connection refused — is Anaconda Desktop running?"}
    except Exception as e:
        return {"error": str(e)}


def desktop_api_post(path, body):
    """POST request to Anaconda Desktop API."""
    try:
        r = requests.post(
            f"{DESKTOP_API_BASE}{path}",
            headers=DESKTOP_HEADERS,
            json=body,
            timeout=30,
        )
        return _handle_response(r)
    except requests.ConnectionError:
        return {"error": "Connection refused — is Anaconda Desktop running?"}
    except Exception as e:
        return {"error": str(e)}


def desktop_api_patch(path, body):
    """PATCH request to Anaconda Desktop API."""
    try:
        r = requests.patch(
            f"{DESKTOP_API_BASE}{path}",
            headers=DESKTOP_HEADERS,
            json=body,
            timeout=30,
        )
        return _handle_response(r)
    except requests.ConnectionError:
        return {"error": "Connection refused — is Anaconda Desktop running?"}
    except Exception as e:
        return {"error": str(e)}


def find_target_model():
    """Find the target model and file in the Anaconda Desktop catalog.
    Returns (model_id, file_info_dict, error_string).
    Stores debug info in st.session_state._desktop_debug.

    Strategy:
    1. GET /api/models — scan model names against broad search terms.
    2. Extract file list from metadata.files (embedded in the model listing).
    3. If no files in metadata.files, fall back to GET /api/models/{id}/files.
    4. Match target quantization in file name or quantization field.
    """
    debug = {}
    resp = desktop_api_get("/api/models")
    if "error" in resp:
        debug["list_models_error"] = resp
        st.session_state._desktop_debug = debug
        return None, None, resp["error"]

    models = resp.get("data", [])
    debug["model_count"] = len(models)

    # Show all model names containing "llama" for diagnostics
    all_names = [m.get("name", "?") for m in models]
    debug["llama_matches"] = [n for n in all_names if "llama" in n.lower()]

    # Broad search: try each search term against each model name
    matched_model = None
    for model in models:
        name_lower = model.get("name", "").lower()
        for term in TARGET_MODEL_SEARCH_TERMS:
            if term in name_lower:
                matched_model = model
                break
        if matched_model:
            break

    if not matched_model:
        debug["search_terms"] = TARGET_MODEL_SEARCH_TERMS
        st.session_state._desktop_debug = debug
        return None, None, (
            f"No model matched search terms {TARGET_MODEL_SEARCH_TERMS}. "
            f"Llama-related models in catalog: {debug['llama_matches']}"
        )

    model_id = matched_model["id"]
    model_name = matched_model.get("name", "?")
    debug["matched_model_id"] = model_id
    debug["matched_model_name"] = model_name

    # --- Collect file list ---
    # Primary: files embedded in metadata.files from the model listing
    metadata = matched_model.get("metadata", {})
    files = metadata.get("files", [])
    debug["files_source"] = "metadata.files" if files else "separate endpoint"

    # Fallback: separate files endpoint
    if not files:
        files_resp = desktop_api_get(f"/api/models/{model_id}/files")
        if "error" in files_resp:
            files_resp = desktop_api_get(f"/models/{model_id}/files")
        debug["files_endpoint_response"] = files_resp
        if "error" not in files_resp:
            files = files_resp.get("data", [])

    debug["file_count"] = len(files)
    debug["file_names"] = [
        f"{f.get('name', '?')} | quant={f.get('quantization', '?')} | id={f.get('id', '?')}"
        for f in files
    ]

    # --- Match quantization ---
    for f in files:
        fname = f.get("name", "").lower()
        fquant = f.get("quantization", "").lower()
        if TARGET_QUANTIZATION in fname or TARGET_QUANTIZATION in fquant:
            debug["matched_file"] = f
            st.session_state._desktop_debug = debug
            return model_id, f, None

    st.session_state._desktop_debug = debug
    return None, None, (
        f"Found model `{model_name}` but no file with quantization `{TARGET_QUANTIZATION}`. "
        f"Available: {[f.get('quantization') for f in files]}"
    )


def check_model_downloaded(file_info):
    """Check if the model file is downloaded locally.
    metadata.files doesn't include isDownloaded — need the separate endpoint.
    """
    if file_info is None:
        return False
    if "isDownloaded" in file_info:
        return file_info["isDownloaded"]
    return None


def get_downloaded_models():
    """Fetch all downloaded model files from the Desktop API.
    Returns list of dicts: {model_name, model_id, file_name, file_id, quantization, size_gb}.
    """
    resp = desktop_api_get("/api/models")
    if "error" in resp:
        return []
    models = resp.get("data", [])
    downloaded = []
    for model in models:
        model_id = model["id"]
        model_name = model.get("name", "?")
        # Check files endpoint for download status
        files_resp = desktop_api_get(f"/api/models/{model_id}/files")
        if "error" in files_resp:
            continue
        for f in files_resp.get("data", []):
            if f.get("isDownloaded", False):
                size_bytes = f.get("sizeBytes", 0)
                downloaded.append({
                    "model_name": model_name,
                    "model_id": model_id,
                    "file_name": f.get("name", "?"),
                    "file_id": f.get("id", ""),
                    "quantization": f.get("quantization", "?"),
                    "size_gb": round(size_bytes / 1e9, 1) if size_bytes else 0,
                    "label": f"{model_name} ({f.get('quantization', '?')}, {round(size_bytes / 1e9, 1)}GB)",
                })
    return downloaded


def get_model_catalog():
    """Fetch full model catalog with download status for each file.
    Filters out files/models the current user cannot access.
    Returns list of dicts with model info + files array.
    """
    resp = desktop_api_get("/api/models")
    if "error" in resp:
        return []
    catalog = []
    for model in resp.get("data", []):
        model_id = model["id"]
        model_name = model.get("name", "?")
        trained_for = model.get("metadata", {}).get("trainedFor", "")
        description = model.get("metadata", {}).get("description", "")
        num_params = model.get("metadata", {}).get("numParameters", 0)

        # Get files with download status
        files_resp = desktop_api_get(f"/api/models/{model_id}/files")
        files = []
        if "error" not in files_resp:
            for f in files_resp.get("data", []):
                # Access check: skip files blocked by policy
                policy = f.get("policy") or {}
                if policy.get("is_blocked", False) and not f.get("isDownloaded", False):
                    continue

                size_bytes = f.get("sizeBytes", 0)
                dl_status = f.get("downloadStatus", {})
                files.append({
                    "file_id": f.get("id", ""),
                    "file_name": f.get("name", "?"),
                    "quantization": f.get("quantization", "?"),
                    "size_gb": round(size_bytes / 1e9, 1) if size_bytes else 0,
                    "is_downloaded": f.get("isDownloaded", False),
                    "download_status": dl_status.get("status", "not_started"),
                    "max_ram_gb": round(f.get("maxRamUsage", 0) / 1e9, 1),
                })

        # Skip models with no accessible files
        if not files:
            continue

        catalog.append({
            "model_id": model_id,
            "model_name": model_name,
            "trained_for": trained_for,
            "description": description[:120] + "..." if len(description) > 120 else description,
            "num_params_b": round(num_params / 1e9, 1) if num_params else 0,
            "files": files,
            "has_downloaded": any(f["is_downloaded"] for f in files),
        })
    return catalog


def start_model_download(model_id, file_id):
    """Trigger a model file download via Desktop API."""
    return desktop_api_patch(
        f"/api/models/{model_id}/files/{file_id}",
        {"action": "start"},
    )


def pause_model_download(model_id, file_id):
    """Pause an in-progress download."""
    return desktop_api_patch(
        f"/api/models/{model_id}/files/{file_id}",
        {"action": "pause"},
    )


def resume_model_download(model_id, file_id):
    """Resume a paused download."""
    return desktop_api_patch(
        f"/api/models/{model_id}/files/{file_id}",
        {"action": "resume"},
    )


def cancel_model_download(model_id, file_id):
    """Cancel a download by deleting the partial file."""
    try:
        r = requests.delete(
            f"{DESKTOP_API_BASE}/api/models/{model_id}/files/{file_id}",
            headers=DESKTOP_HEADERS, timeout=10,
        )
        if r.ok:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:500]}
    except Exception as e:
        return {"error": str(e)}


def get_download_progress(model_id, file_id):
    """Check download progress for a model file. Returns rich progress dict."""
    resp = desktop_api_get(f"/api/models/{model_id}/files/{file_id}")
    if "error" in resp:
        return resp
    data = resp.get("data", {})
    dl = data.get("downloadStatus", {})
    progress = dl.get("progress", {})
    return {
        "status": dl.get("status", "unknown"),
        "is_downloaded": data.get("isDownloaded", False),
        "ratio": progress.get("downloadedRatio", 0),
        "transferred_bytes": progress.get("transferredBytes", 0),
        "total_bytes": progress.get("totalBytes", 0),
        "speed_mbps": progress.get("averageMbps", 0),
        "time_ms": progress.get("downloadTimeTaken", 0),
        "paused": progress.get("paused", False),
    }


def _model_family(name):
    """Extract a grouping family name from a model name string."""
    import re
    # Common prefixes: "Meta-Llama-3..." → "Meta Llama", "Qwen2.5-..." → "Qwen", etc.
    mappings = [
        (r"^Meta-Llama", "Meta Llama"),
        (r"^Llama-Guard", "Meta Llama"),
        (r"^Llama-3", "Meta Llama"),
        (r"^Qwen", "Qwen"),
        (r"^DeepSeek", "DeepSeek"),
        (r"^gemma", "Google Gemma"),
        (r"^embeddinggemma", "Google Gemma"),
        (r"^Mistral", "Mistral"),
        (r"^NVIDIA", "NVIDIA"),
        (r"^granite", "IBM Granite"),
        (r"^gpt", "OpenAI"),
        (r"^phi-", "Microsoft"),
        (r"^TinyLlama", "TinyLlama"),
        (r"^bge-", "BAAI BGE"),
        (r"^Olmo", "Allen AI"),
    ]
    for pattern, family in mappings:
        if re.match(pattern, name, re.IGNORECASE):
            return family
    # Fallback: first token
    return name.split("-")[0].split("_")[0]


def find_server_on_port():
    """Find any server on the target port (any status)."""
    resp = desktop_api_get("/api/servers", params={"port": LOCAL_SERVER_PORT})
    if "error" in resp:
        return None
    servers = resp.get("data", [])
    return servers[0] if servers else None


def find_any_server_for_model():
    """Find any server (running or not) for the target model file."""
    resp = desktop_api_get("/api/servers")
    if "error" in resp:
        return None
    servers = resp.get("data", [])
    for s in servers:
        cfg = s.get("serverConfig", {})
        mf = s.get("modelFile", {})
        if TARGET_QUANTIZATION in cfg.get("modelFileName", "").lower() or TARGET_QUANTIZATION in mf.get("name", "").lower():
            return s
    return None


def create_and_start_server(model_filename):
    """Create a new server for the model and start it."""
    body = {
        "serverConfig": {
            "modelFileName": model_filename,
            "apiParams": {
                "port": LOCAL_SERVER_PORT,
            },
            "loadParams": {},
            "inferParams": {},
        },
        "startServerOnCreate": True,
    }
    return desktop_api_post("/api/servers", body)


def start_existing_server(server_id):
    """Start a stopped server."""
    return desktop_api_patch(
        f"/api/servers/{server_id}",
        {"action": "start"},
    )


def stop_server(server_id):
    """Stop a running server."""
    return desktop_api_patch(
        f"/api/servers/{server_id}",
        {"action": "stop"},
    )


# ============================================================
# Session state defaults
# ============================================================
if "inference_mode" not in st.session_state:
    st.session_state.inference_mode = "cloud"  # default to cloud
if "mode_radio" not in st.session_state:
    st.session_state.mode_radio = "cloud"
if "cloud_url" not in st.session_state:
    st.session_state.cloud_url = _PERSISTED_CLOUD_URL
if "cloud_api_key" not in st.session_state:
    st.session_state.cloud_api_key = _PERSISTED_CLOUD_KEY
if "local_url" not in st.session_state:
    st.session_state.local_url = f"http://localhost:{LOCAL_SERVER_PORT}"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 16384
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False
if "_desktop_debug" not in st.session_state:
    st.session_state._desktop_debug = {}
if "benchmark_log" not in st.session_state:
    st.session_state.benchmark_log = []

# Auto-load latest trained RF model on startup (if configured and available)
# Uses Metaflow Client API — requires outerbounds configure on the pod
if (
    PIPELINE_AVAILABLE
    and AUTO_LOAD_MODEL
    and st.session_state.get("rf_model") is None
):
    try:
        from pipeline_client import load_latest_model
        _auto = load_latest_model()
        if "error" not in _auto:
            st.session_state.rf_model = _auto["model"]
            st.session_state.rf_metrics = _auto["metrics"]
            st.session_state.rf_hparams = _auto["hparams"]
            st.session_state.rf_model_config = _auto["model_config"]
            st.session_state.rf_run_id = _auto["run_id"]
            if "feature_names" in _auto:
                st.session_state.rf_feature_names = _auto["feature_names"]
            print(f"[PIPELINE] Loaded model from Metaflow run {_auto['run_id']}")
        else:
            print(f"[PIPELINE] Metaflow load: {_auto['error']}")
            st.session_state["_pipeline_load_error"] = _auto["error"]
    except Exception as _e:
        print(f"[PIPELINE] Metaflow auto-load failed: {_e}")
        st.session_state["_pipeline_load_error"] = str(_e)


# ============================================================
# Data generation (same as notebook)
# ============================================================
@st.cache_data
def generate_transaction_data():
    np.random.seed(42)
    n_transactions = 200
    base_date = datetime(2026, 2, 20)

    merchant_categories = [
        "Grocery", "Restaurant", "Online Retail", "Travel", "Fuel",
        "Electronics", "ATM Withdrawal", "Wire Transfer", "Luxury Goods", "Subscription",
    ]
    countries = ["DE", "DE", "DE", "DE", "GB", "US", "FR", "NL", "CH", "NG", "RU", "CN", "BR"]

    data = {
        "transaction_id": [f"TXN-{i:06d}" for i in range(n_transactions)],
        "timestamp": [
            base_date + timedelta(hours=int(np.random.randint(0, 120)), minutes=int(np.random.randint(0, 60)))
            for _ in range(n_transactions)
        ],
        "card_number": [f"****-****-****-{np.random.randint(1000, 9999)}" for _ in range(n_transactions)],
        "amount_eur": np.round(np.abs(np.random.lognormal(mean=3.5, sigma=1.2, size=n_transactions)), 2),
        "merchant_category": np.random.choice(merchant_categories, n_transactions),
        "country": np.random.choice(
            countries, n_transactions,
            p=[0.30, 0.15, 0.10, 0.08, 0.08, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01],
        ),
        "is_online": np.random.choice([True, False], n_transactions, p=[0.4, 0.6]),
    }

    df = pd.DataFrame(data)
    df["hour"] = df["timestamp"].dt.hour

    fraud_indices = []

    # Pattern 1: Card testing
    card_test_idx = np.random.choice(range(n_transactions), 5, replace=False)
    df.loc[card_test_idx, "amount_eur"] = np.round(np.random.uniform(0.50, 2.00, 5), 2)
    df.loc[card_test_idx, "merchant_category"] = "Online Retail"
    df.loc[card_test_idx, "is_online"] = True
    df.loc[card_test_idx, "card_number"] = "****-****-****-7721"
    df.loc[card_test_idx, "timestamp"] = [base_date + timedelta(hours=2, minutes=i) for i in range(5)]
    fraud_indices.extend(card_test_idx.tolist())

    # Pattern 2: High-value geographic anomaly
    geo_fraud_idx = np.random.choice(
        [i for i in range(n_transactions) if i not in fraud_indices], 4, replace=False
    )
    df.loc[geo_fraud_idx, "amount_eur"] = np.round(np.random.uniform(5000, 15000, 4), 2)
    df.loc[geo_fraud_idx, "country"] = np.random.choice(["NG", "RU"], 4)
    df.loc[geo_fraud_idx, "merchant_category"] = np.random.choice(["Wire Transfer", "Luxury Goods"], 4)
    fraud_indices.extend(geo_fraud_idx.tolist())

    # Pattern 3: Late-night ATM withdrawals abroad
    atm_fraud_idx = np.random.choice(
        [i for i in range(n_transactions) if i not in fraud_indices], 3, replace=False
    )
    df.loc[atm_fraud_idx, "amount_eur"] = np.round(np.random.uniform(800, 2000, 3), 2)
    df.loc[atm_fraud_idx, "merchant_category"] = "ATM Withdrawal"
    df.loc[atm_fraud_idx, "country"] = "BR"
    atm_hours = np.random.choice([1, 2, 3, 4], 3)
    df.loc[atm_fraud_idx, "timestamp"] = [base_date + timedelta(hours=int(h)) for h in atm_hours]
    fraud_indices.extend(atm_fraud_idx.tolist())

    df["hour"] = df["timestamp"].dt.hour
    df["ground_truth_fraud"] = df.index.isin(fraud_indices)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# Inference helpers
# ============================================================
def get_active_config():
    """Return (base_url, api_key) based on current inference mode."""
    if st.session_state.inference_mode == "local":
        return st.session_state.local_url, ""
    else:
        return st.session_state.cloud_url, st.session_state.cloud_api_key


def _build_client(base_url, api_key):
    """Build OpenAI client. Adds x-api-key header for Outerbounds endpoints.
    Returns (client, model_name).
    """
    is_outerbounds = "merced.obp.outerbounds.com" in base_url
    kwargs = {"base_url": base_url, "api_key": api_key if api_key else "not-needed"}
    if is_outerbounds:
        kwargs["default_headers"] = {"x-api-key": api_key}
    model_name = "catalyst" if is_outerbounds else ""
    return OpenAI(**kwargs), model_name


def ask_model(prompt, system_prompt=None):
    """Send a prompt to the active inference endpoint."""
    base_url, api_key = get_active_config()
    client, model_name = _build_client(base_url, api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=st.session_state.max_tokens,
        temperature=st.session_state.temperature,
    )
    return response.choices[0].message.content


def ask_model_timed(prompt, system_prompt=None, max_tokens_override=None):
    """Send a prompt and return (response_text, elapsed_seconds, token_usage).
    Retries on 503 with exponential backoff (cloud endpoints with limited concurrency).
    Logs full request/response details to console for debugging.
    """
    base_url, api_key = get_active_config()
    client, model_name = _build_client(base_url, api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_tok = max_tokens_override or st.session_state.max_tokens
    max_retries = 5
    base_wait = 3.0  # seconds

    _debug(f"\n[DEBUG] ask_model_timed → {base_url}")
    _debug(f"[DEBUG]   max_tokens={max_tok}, temperature={st.session_state.temperature}")
    _debug(f"[DEBUG]   prompt length={len(prompt)} chars, system={'yes' if system_prompt else 'no'}")

    last_exception = None
    t0_total = time.perf_counter()

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tok,
                temperature=st.session_state.temperature,
            )
            elapsed = time.perf_counter() - t0

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            text = response.choices[0].message.content
            if attempt > 0:
                _debug(f"[DEBUG]   OK after {attempt} retries, {elapsed:.2f}s | usage={usage}")
            else:
                _debug(f"[DEBUG]   OK {elapsed:.2f}s | usage={usage} | response={text[:100]}...")

            return text, elapsed, usage

        except Exception as e:
            elapsed = time.perf_counter() - t0
            last_exception = e
            status_code = getattr(e, "status_code", None)

            _debug(f"[DEBUG]   Attempt {attempt + 1}/{max_retries + 1} ERROR after {elapsed:.2f}s: {type(e).__name__}: {e}")
            if hasattr(e, "response"):
                try:
                    _debug(f"[DEBUG]   Response headers: {dict(e.response.headers)}")
                    _debug(f"[DEBUG]   Response body: {e.response.text[:2000]}")
                except Exception:
                    pass
            if hasattr(e, "body"):
                _debug(f"[DEBUG]   Error body: {e.body}")

            # Retry only on 503 (server busy / no available server)
            if status_code == 503 and attempt < max_retries:
                wait = base_wait * (1.5 ** attempt)
                _debug(f"[DEBUG]   Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                raise


def format_transaction(row):
    """Format a transaction row for the model."""
    return (
        f"Transaction ID: {row['transaction_id']}\n"
        f"Timestamp: {row['timestamp']}\n"
        f"Card: {row['card_number']}\n"
        f"Amount: EUR {row['amount_eur']:,.2f}\n"
        f"Merchant Category: {row['merchant_category']}\n"
        f"Country: {row['country']}\n"
        f"Online Transaction: {row['is_online']}\n"
        f"Hour of Day: {row['hour']:02d}:00"
    )


# System prompts (from notebook)
FRAUD_SYSTEM_PROMPT = """You are a senior fraud analyst at a major European bank. 
You will be given credit card transaction details. Analyse the transaction and provide:

1. **Risk Level**: LOW, MEDIUM, or HIGH
2. **Risk Score**: 0-100 (0 = certainly legitimate, 100 = certainly fraudulent)
3. **Red Flags**: List any suspicious indicators
4. **Recommendation**: APPROVE, REVIEW, or BLOCK
5. **Reasoning**: Brief explanation of your assessment

Respond in the exact format above. Be concise but thorough."""

BATCH_SYSTEM_PROMPT = """You are a senior fraud analyst at a major European bank.
You will receive a batch of recent credit card transactions. Your task is to:

1. Identify any suspicious PATTERNS across the transactions (not just individual anomalies)
2. Flag specific transaction IDs that warrant investigation
3. Describe the type of fraud pattern detected (e.g., card testing, account takeover, bust-out)
4. Recommend immediate actions

Focus on cross-transaction patterns like: rapid-fire small amounts on the same card,
geographic impossibilities, unusual merchant category sequences, or velocity anomalies."""

RISK_SYSTEM_PROMPT = """You are a Chief Risk Officer preparing a daily transaction monitoring briefing 
for the executive committee of a major European bank. Generate a concise, professional risk summary 
based on the transaction statistics provided. Include:

1. Executive Summary (2-3 sentences)
2. Key Risk Indicators
3. Notable Patterns or Concerns
4. Recommended Actions

Use formal banking language appropriate for a board-level audience."""

COMPLIANCE_SYSTEM_PROMPT = """You are a regulatory compliance expert specialising in European banking regulation.
Provide accurate, concise answers to compliance questions. Reference specific regulations, directives,
or guidelines where relevant (e.g., Basel III/IV, MiFID II, PSD2, GDPR, 6AMLD).
Always note when information may need to be verified against the latest regulatory updates."""

EXTRACTION_SYSTEM_PROMPT = """You are a data extraction specialist at a bank. Extract structured information 
from the text provided and return it as valid JSON. Be precise and only extract information that is 
explicitly stated in the text."""

STREAMING_FRAUD_PROMPT = """You are a fraud detection system. You will receive one or more transactions separated by "---".
For EACH transaction, respond with exactly one line in this format:
[Transaction ID] RISK:[LOW|MEDIUM|HIGH] SCORE:[0-100] ACTION:[APPROVE|REVIEW|BLOCK] REASON:[one sentence]

Output one line per transaction, in the same order as the input. No other text."""


def show_timing(elapsed, usage=None):
    """Display a compact timing badge after an inference call."""
    parts = [f"⏱️ **{elapsed:.2f}s**"]
    if usage:
        comp = usage.get("completion_tokens")
        if comp and elapsed > 0:
            parts.append(f"{comp} tokens")
            parts.append(f"{comp / elapsed:.0f} tok/s")
    st.caption(" · ".join(parts))


def resolve_cloud_model_info(cloud_url, api_key):
    """Resolve model name/quant/size from a cloud serving endpoint URL.
    
    Flow:
    1. Extract server ID from URL
    2. GET /api/ai/inference/servers → match server → get name, model_uuid, file_uuid, status
    3. GET /api/ai/model/org/models/model-data → match model_uuid → get model name, quant, size
    
    Returns dict with keys: server_name, server_status, model_name, quantization, size_gb, label
    Cached in session state keyed by URL.
    """
    cache_key = f"_cloud_model_info_{cloud_url}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    info = {"server_name": None, "server_status": None, "model_name": None,
            "quantization": None, "size_gb": None, "label": None}

    try:
        # Extract server ID and base URL
        # URL format: https://host/api/ai/inference/serve/{server_id}
        parts = cloud_url.rstrip("/").split("/")
        server_id = parts[-1]
        base_url = cloud_url.split("/api/ai/inference/serve/")[0]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # Step 1: Get server info
        r = requests.get(f"{base_url}/api/ai/inference/servers", headers=headers, timeout=10)
        if r.ok:
            servers = r.json().get("data", {}).get("servers", [])
            server = next((s for s in servers if s.get("id") == server_id), None)
            if server:
                info["server_name"] = server.get("name")
                info["server_status"] = server.get("status")
                model_uuid = server.get("model_uuid")
                file_uuid = server.get("file_uuid")

                # Step 2: Get model catalog and match by model_uuid / file_uuid
                catalog_url = f"{base_url}/api/ai/model/org/models/model-data"
                cat_headers = dict(headers)
                cat_headers["x-anaconda-api-version"] = "2"
                _debug(f"[DEBUG] Cloud catalog lookup: GET {catalog_url}")
                _debug(f"[DEBUG]   Looking for model_uuid={model_uuid}, file_uuid={file_uuid}")
                r2 = requests.get(catalog_url, headers=cat_headers, timeout=15)
                _debug(f"[DEBUG]   Catalog response: HTTP {r2.status_code}")
                if r2.ok:
                    r2_json = r2.json()
                    # Try multiple response shapes
                    catalog = (
                        r2_json.get("result", {}).get("data", [])
                        or r2_json.get("data", {}).get("models", [])
                        or r2_json.get("data", [])
                        or r2_json.get("models", [])
                    )
                    _debug(f"[DEBUG]   Catalog entries: {len(catalog)}")
                    if catalog:
                        # Log first entry keys to understand structure
                        first = catalog[0]
                        _debug(f"[DEBUG]   First entry keys: {list(first.keys())[:15]}")
                        uuid_key = None
                        for k in ["model_uuid", "id", "modelId", "uuid"]:
                            if k in first:
                                uuid_key = k
                                _debug(f"[DEBUG]   UUID field: '{k}' = '{first[k]}'")
                                break
                    for model in catalog:
                        m_uuid = model.get("model_uuid") or model.get("id") or model.get("uuid", "")
                        if m_uuid == model_uuid:
                            info["model_name"] = model.get("name")
                            _debug(f"[DEBUG]   MATCHED model: {info['model_name']}")
                            # Find matching quantized file by file_uuid
                            qfiles = model.get("quantized_files", []) or model.get("quantizedFiles", [])
                            for qf in qfiles:
                                f_uuid = qf.get("file_uuid") or qf.get("id", "")
                                if f_uuid == file_uuid:
                                    info["quantization"] = qf.get("quant_method") or qf.get("quantMethod")
                                    size = qf.get("size_bytes") or qf.get("sizeBytes", 0)
                                    info["size_gb"] = round(size / 1e9, 1) if size else None
                                    _debug(f"[DEBUG]   MATCHED file: {info['quantization']} {info['size_gb']}GB")
                                    break
                            break
                else:
                    _debug(f"[DEBUG]   Catalog error: {r2.status_code} {r2.text[:500]}")

        # Build label
        parts = []
        if info["model_name"]:
            parts.append(info["model_name"])
        elif info["server_name"]:
            parts.append(info["server_name"])
        if info["quantization"]:
            parts.append(info["quantization"])
        if info["size_gb"]:
            parts.append(f"{info['size_gb']}GB")
        info["label"] = " · ".join(parts) if parts else None

    except Exception as e:
        _debug(f"[DEBUG] resolve_cloud_model_info error: {e}")

    st.session_state[cache_key] = info
    return info


def _get_model_label():
    """Return a label for the currently active model."""
    if st.session_state.inference_mode == "local":
        sel = st.session_state.get("selected_model_file")
        return sel["label"] if sel else "Local (unknown)"
    else:
        url = st.session_state.cloud_url
        api_key = st.session_state.cloud_api_key
        info = resolve_cloud_model_info(url, api_key)
        if info.get("label"):
            return f"Cloud · {info['label']}"
        # Fallback: server name from URL
        server_id = url.rstrip("/").split("/")[-1]
        return f"Cloud · {server_id[:12]}"


def log_inference(task_type, elapsed, usage=None):
    """Log an inference call to the benchmark ledger."""
    comp = 0
    if usage:
        comp = usage.get("completion_tokens", 0) or 0
    st.session_state.benchmark_log.append({
        "task": task_type,
        "model": _get_model_label(),
        "mode": st.session_state.inference_mode.upper(),
        "latency": round(elapsed, 3),
        "tokens": comp,
        "tok_s": round(comp / elapsed, 1) if elapsed > 0 and comp > 0 else 0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


# ============================================================
# Sidebar — Inference Configuration
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="padding:8px 0 4px 0;"><span style="font-size:1.5em;font-weight:700;color:#43B049;">⬢ ANACONDA</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.header("Inference Configuration")

    if DEV_MODE:
        def _on_mode_change():
            st.session_state.inference_mode = st.session_state.mode_radio

        st.radio(
            "Inference Endpoint",
            options=["cloud", "local"],
            format_func=lambda x: "☁️ Cloud (AI Catalyst)" if x == "cloud" else "💻 Local (Anaconda Desktop)",
            key="mode_radio",
            on_change=_on_mode_change,
        )
        mode = st.session_state.mode_radio
        st.session_state.inference_mode = mode
    else:
        # Prod: cloud only, no toggle
        st.session_state.inference_mode = "cloud"
        mode = "cloud"
        st.caption("☁️ Cloud (AI Catalyst)")

    if DEV_MODE:
        st.markdown("---")

    if mode == "cloud":
        if DEV_MODE:
            st.subheader("Cloud Endpoint")
            new_url = st.text_input(
                "Endpoint URL",
                value=st.session_state.cloud_url,
                key="cloud_url_input",
                help="OpenAI-compatible base URL for hosted inference",
            )
            st.session_state.cloud_url = new_url
            new_key = st.text_input(
                "API Key (optional)",
                value=st.session_state.cloud_api_key,
                type="password",
                key="cloud_key_input",
                help="Bearer token for the hosted endpoint. Leave blank if none required.",
            )
            st.session_state.cloud_api_key = new_key

            # Lock button — persists current URL + key to config.yml
            is_locked = (
                st.session_state.cloud_url == _PERSISTED_CLOUD_URL
                and st.session_state.cloud_api_key == _PERSISTED_CLOUD_KEY
                and _PERSISTED_CLOUD_URL != ""
            )
            if is_locked:
                st.success("🔒 Endpoint locked to config")
                if st.button("🔓 Unlock", key="unlock_endpoint", help="Allow editing again"):
                    pass  # Fields are already editable; this is just visual
            else:
                if st.button("🔒 Lock Endpoint", key="lock_endpoint",
                             help="Save current URL + API key to config.yml for prod use"):
                    _save_config(
                        cloud_url=st.session_state.cloud_url,
                        cloud_api_key=st.session_state.cloud_api_key,
                    )
                    st.success("Saved to config.yml — this endpoint will be used in prod mode.")
                    st.rerun()
    else:
        st.subheader("Local Server")
        if DEV_MODE:
            new_local_url = st.text_input(
                "Local URL",
                value=st.session_state.local_url,
                key="local_url_input",
                help="Defaults to localhost:8080. Change if your server runs on a different port.",
            )
            st.session_state.local_url = new_local_url

        # --- Model selector ---
        st.markdown("##### Model")
        if "downloaded_models" not in st.session_state:
            st.session_state.downloaded_models = []
        if "selected_model_file" not in st.session_state:
            st.session_state.selected_model_file = None

        model_col1, model_col2 = st.columns([3, 1])
        with model_col2:
            if st.button("🔄", key="refresh_models", help="Refresh model list from Anaconda Desktop"):
                with st.spinner("Scanning..."):
                    st.session_state.downloaded_models = get_downloaded_models()

        # Auto-fetch on first load if empty
        if not st.session_state.downloaded_models:
            health = desktop_api_get("/api/servers/health")
            if "error" not in health:
                st.session_state.downloaded_models = get_downloaded_models()

        dl_models = st.session_state.downloaded_models
        if dl_models:
            labels = [m["label"] for m in dl_models]
            # Try to pre-select the target model
            default_idx = 0
            for i, m in enumerate(dl_models):
                if TARGET_QUANTIZATION in m["file_name"].lower() and any(
                    term in m["model_name"].lower() for term in TARGET_MODEL_SEARCH_TERMS
                ):
                    default_idx = i
                    break

            with model_col1:
                selected_label = st.selectbox(
                    "Downloaded models",
                    labels,
                    index=default_idx,
                    key="model_selector",
                    label_visibility="collapsed",
                )
            selected_idx = labels.index(selected_label)
            st.session_state.selected_model_file = dl_models[selected_idx]
        else:
            with model_col1:
                st.caption("No downloaded models found.")
            st.session_state.selected_model_file = None

        # --- Model Catalog Browser ---
        if "model_catalog" not in st.session_state:
            st.session_state.model_catalog = []
        if "active_download" not in st.session_state:
            st.session_state.active_download = None
        # Handle pending catalog collapse (set by download trigger before rerun)
        if st.session_state.pop("_collapse_catalog", False):
            st.session_state.show_catalog = False
        if "show_catalog" not in st.session_state:
            st.session_state.show_catalog = not dl_models

        # Download progress — above catalog, always visible
        if st.session_state.active_download:
            @st.fragment(run_every=timedelta(seconds=2))
            def download_progress_panel():
                dl_info = st.session_state.active_download
                if not dl_info:
                    return
                mid, fid = dl_info["model_id"], dl_info["file_id"]
                prog = get_download_progress(mid, fid)
                status = prog.get("status", "unknown")

                if status == "completed" or prog.get("is_downloaded"):
                    st.success(f"✅ {dl_info['label']} — complete")
                    st.session_state.active_download = None
                    st.session_state.downloaded_models = get_downloaded_models()
                    st.session_state.model_catalog = []

                elif status == "in_progress":
                    ratio = prog.get("ratio", 0)
                    transferred = prog.get("transferred_bytes", 0)
                    total = prog.get("total_bytes", 0)
                    speed = prog.get("speed_mbps", 0)
                    transferred_gb = round(transferred / 1e9, 2)
                    total_gb = round(total / 1e9, 2) if total else "?"
                    eta_str = ""
                    if speed and total and transferred < total:
                        remaining_mb = (total - transferred) / 1e6
                        eta_s = remaining_mb / (speed / 8) if speed > 0 else 0
                        eta_str = f"~{eta_s:.0f}s" if eta_s < 60 else f"~{eta_s / 60:.0f}m"
                    st.progress(min(ratio, 1.0), text=f"⬇️ {dl_info['label']}")
                    parts = [f"{ratio * 100:.0f}%", f"{transferred_gb}/{total_gb} GB"]
                    if speed:
                        parts.append(f"{speed:.1f} Mbps")
                    if eta_str:
                        parts.append(eta_str)
                    st.caption(" · ".join(parts))
                    # Pause / Cancel controls
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("⏸️ Pause", key="dl_pause"):
                            pause_model_download(mid, fid)
                            st.rerun()
                    with bc2:
                        if st.button("🗑️ Cancel", key="dl_cancel"):
                            cancel_model_download(mid, fid)
                            st.session_state.active_download = None
                            st.session_state.model_catalog = []
                            st.rerun()

                elif status == "paused" or prog.get("paused"):
                    ratio = prog.get("ratio", 0)
                    st.progress(min(ratio, 1.0), text=f"⏸️ {dl_info['label']} — paused")
                    st.caption(f"{ratio * 100:.0f}% complete")
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("▶️ Resume", key="dl_resume"):
                            resume_model_download(mid, fid)
                            st.rerun()
                    with bc2:
                        if st.button("🗑️ Cancel", key="dl_cancel_p"):
                            cancel_model_download(mid, fid)
                            st.session_state.active_download = None
                            st.session_state.model_catalog = []
                            st.rerun()

                elif "error" in prog:
                    st.error(f"Download error: {prog['error']}")
                    st.session_state.active_download = None
                else:
                    st.info(f"⬇️ {dl_info['label']} — {status}")

            download_progress_panel()

        # Toggle catalog visibility
        st.checkbox("📦 Browse & Download Models", key="show_catalog")

        if st.session_state.show_catalog:
            cat_c1, cat_c2 = st.columns([3, 1])
            with cat_c2:
                if st.button("🔄", key="refresh_catalog", help="Refresh catalog"):
                    with st.spinner("Loading..."):
                        st.session_state.model_catalog = get_model_catalog()
            with cat_c1:
                cat_filter = st.selectbox(
                    "Task",
                    ["text-generation", "All", "sentence-similarity"],
                    key="catalog_filter",
                    label_visibility="collapsed",
                )

            # Auto-fetch if empty and Desktop API reachable
            if not st.session_state.model_catalog:
                health = desktop_api_get("/api/models/health")
                if "error" not in health:
                    st.session_state.model_catalog = get_model_catalog()

            catalog = st.session_state.model_catalog
            if not catalog:
                st.info("Could not reach Anaconda Desktop API.")
            else:
                if cat_filter != "All":
                    catalog = [m for m in catalog if m["trained_for"] == cat_filter]

                # Build family → model hierarchy for dropdowns
                from collections import OrderedDict
                families = OrderedDict()
                for model in catalog:
                    fam = _model_family(model["model_name"])
                    families.setdefault(fam, []).append(model)

                family_names = list(families.keys())
                selected_family = st.selectbox(
                    "Family", family_names, key="cat_family",
                    label_visibility="collapsed",
                    format_func=lambda f: (
                        f"{f} ({sum(1 for m in families[f] if m['has_downloaded'])}/{len(families[f])})"
                    ),
                )

                if selected_family:
                    fam_models = families[selected_family]
                    model_labels = [
                        f"{'🟢' if m['has_downloaded'] else '⬜'} {m['model_name']} · {m['num_params_b']}B"
                        for m in fam_models
                    ]
                    selected_model_idx = st.selectbox(
                        "Model", range(len(model_labels)), key="cat_model",
                        format_func=lambda i: model_labels[i],
                        label_visibility="collapsed",
                    )

                    if selected_model_idx is not None:
                        model = fam_models[selected_model_idx]
                        for f in model["files"]:
                            fcol1, fcol2 = st.columns([3, 1])
                            with fcol1:
                                if f["is_downloaded"]:
                                    st.caption(f"✅ {f['quantization']} — {f['size_gb']}GB")
                                elif f["download_status"] == "in_progress":
                                    st.caption(f"⏳ {f['quantization']} — downloading...")
                                else:
                                    st.caption(
                                        f"⬜ {f['quantization']} — {f['size_gb']}GB · "
                                        f"RAM: {f['max_ram_gb']}GB"
                                    )
                            with fcol2:
                                if not f["is_downloaded"] and f["download_status"] != "in_progress":
                                    btn_key = f"dl_{model['model_id']}_{f['file_id']}"
                                    if st.button("⬇️", key=btn_key, help=f"Download {f['quantization']}"):
                                        with st.spinner("Starting..."):
                                            result = start_model_download(
                                                model["model_id"], f["file_id"]
                                            )
                                        if "error" in result:
                                            st.error(f"Failed: {result['error']}")
                                        else:
                                            st.session_state.active_download = {
                                                "model_id": model["model_id"],
                                                "file_id": f["file_id"],
                                                "label": f"{model['model_name']} ({f['quantization']})",
                                            }
                                            st.session_state._collapse_catalog = True
                                            st.rerun()
                                elif f["download_status"] == "in_progress":
                                    st.caption("⏳")

        # --- Auto-refreshing server status fragment ---
        @st.fragment(run_every=timedelta(seconds=5))
        def server_status_panel():
            st.markdown("##### Server Status")

            health = desktop_api_get("/api/servers/health")
            if "error" in health:
                st.warning("Anaconda Desktop API not reachable.")
                return

            selected = st.session_state.selected_model_file
            selected_file_name = selected["file_name"] if selected else None

            # Check for any running server on target port
            server = find_server_on_port()

            # Also check for a server matching the selected model
            if not server and selected_file_name:
                resp = desktop_api_get("/api/servers")
                if "error" not in resp:
                    for s in resp.get("data", []):
                        mf = s.get("modelFile", {})
                        cfg = s.get("serverConfig", {})
                        if (selected_file_name.lower() in mf.get("name", "").lower()
                                or selected_file_name.lower() in cfg.get("modelFileName", "").lower()):
                            server = s
                            break

            if server:
                status = server.get("status", "unknown").upper()
                server_id = server.get("id", "")
                srv_info = server.get("server", {})
                srv_port = srv_info.get("port", "?")
                srv_model = server.get("modelFile", {}).get("name", "?")

                if status == "RUNNING":
                    st.success(f"✅ Server **running** on port {srv_port}")
                    st.caption(f"Model: `{srv_model}`")
                    if st.button("⏹️ Stop Server", key="stop_server_btn"):
                        with st.spinner("Stopping server..."):
                            resp = stop_server(server_id)
                        if "error" in resp:
                            st.error(f"Stop failed: {resp['error']}")

                elif status in ("STARTING",):
                    st.info(f"⏳ Server is starting on port {srv_port}...")
                    st.caption(f"Model: `{srv_model}`")

                elif status in ("STOPPING",):
                    st.info("⏳ Server is stopping...")

                else:
                    st.warning(f"Server status: **{status}**")
                    st.caption(f"Model: `{srv_model}`")
                    if st.button("🚀 Start Server", key="start_server_btn"):
                        with st.spinner("Starting server..."):
                            resp = start_existing_server(server_id)
                        if "error" in resp:
                            st.error(f"Start failed: {resp['error']}")

            else:
                if not selected_file_name:
                    st.warning("Select a model above to start a server.")
                else:
                    st.warning("No server found for selected model.")
                    st.caption(f"Model file: `{selected_file_name}`")
                    if st.button("🚀 Create & Start Server", key="create_server_btn"):
                        with st.spinner("Creating server and loading model..."):
                            resp = create_and_start_server(selected_file_name)
                        if "error" in resp:
                            st.error(f"Create failed: {resp['error']}")
                            detail = resp.get("detail")
                            if detail:
                                st.code(json.dumps(detail, indent=2) if isinstance(detail, dict) else str(detail), language="json")

        server_status_panel()

    if DEV_MODE:
        st.markdown("---")
        st.subheader("Model Parameters")
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
        st.session_state.max_tokens = st.number_input("Max Tokens", 256, 32768, st.session_state.max_tokens, 256)

        st.markdown("---")
    base_url, _ = get_active_config()
    mode_label = "CLOUD" if st.session_state.inference_mode == "cloud" else "LOCAL"
    st.caption(f"**Active:** `{mode_label}`")
    st.caption(f"**Endpoint:** `{base_url}`")

    # Resolve and display cloud model info
    if st.session_state.inference_mode == "cloud":
        cloud_info = resolve_cloud_model_info(
            st.session_state.cloud_url, st.session_state.cloud_api_key
        )
        if cloud_info.get("label"):
            st.caption(f"**Model:** {cloud_info['label']}")
        if cloud_info.get("server_name"):
            status_icon = "🟢" if cloud_info.get("server_status") == "running" else "🟡"
            st.caption(f"**Server:** {status_icon} {cloud_info['server_name']} ({cloud_info.get('server_status', '?')})")
        if DEV_MODE:
            if st.button("🔄 Refresh Model Info", key="refresh_cloud_info"):
                # Clear cached info to force re-fetch
                cache_key = f"_cloud_model_info_{st.session_state.cloud_url}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.rerun()

    if DEV_MODE:
        # ML Pipeline model status
        if PIPELINE_AVAILABLE:
            sidebar_model_status()

        # Connection test
        if st.button("🔌 Test Connection", key="test_conn"):
            with st.spinner("Testing..."):
                try:
                    result = ask_model("Reply with only: CONNECTION OK", system_prompt=None)
                    st.success(f"Response: {result.strip()[:100]}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        # Debug panel — shows raw Desktop API responses for troubleshooting
        if mode == "local" and st.session_state._desktop_debug:
            with st.expander("🐛 Desktop API Debug", expanded=False):
                st.json(st.session_state._desktop_debug)

        # Full raw diagnostic — bypasses all helpers
        if mode == "local":
            with st.expander("🔬 Run Full API Diagnostic", expanded=False):
                if st.button("Run Diagnostic", key="diag_btn"):
                    diag = []
    
                    def _raw(method, path, body=None):
                        url = f"{DESKTOP_API_BASE}{path}"
                        entry = {"method": method, "url": url, "body": body}
                        try:
                            if method == "GET":
                                r = requests.get(url, headers=DESKTOP_HEADERS, timeout=10)
                            elif method == "PATCH":
                                r = requests.patch(url, headers=DESKTOP_HEADERS, json=body, timeout=10)
                            elif method == "POST":
                                r = requests.post(url, headers=DESKTOP_HEADERS, json=body, timeout=10)
                            entry["status_code"] = r.status_code
                            entry["headers"] = dict(r.headers)
                            entry["raw_text"] = r.text[:4000]
                            try:
                                entry["json"] = r.json()
                            except Exception:
                                entry["json"] = None
                        except Exception as e:
                            entry["exception"] = str(e)
                        diag.append(entry)
                        return entry
    
                    st.markdown("##### Step 1: Health check")
                    h = _raw("GET", "/api/models/health")
                    st.write(f"→ HTTP {h.get('status_code')}")
    
                    st.markdown("##### Step 2: List models + find Llama")
                    m = _raw("GET", "/api/models")
                    models = (m.get("json") or {}).get("data", [])
                    st.write(f"Total models in catalog: {len(models)}")
    
                    # Show all model names containing "llama"
                    llama_models = [mdl for mdl in models if "llama" in mdl.get("name", "").lower()]
                    st.write(f"Models containing 'llama': {len(llama_models)}")
                    for mdl in llama_models:
                        st.caption(f"`{mdl.get('id')}` — **{mdl.get('name')}**")
    
                    # Broad match
                    matched_model = None
                    for mdl in models:
                        name_lower = mdl.get("name", "").lower()
                        for term in TARGET_MODEL_SEARCH_TERMS:
                            if term in name_lower:
                                matched_model = mdl
                                break
                        if matched_model:
                            break
    
                    if not matched_model:
                        st.error(f"No model matched any of: {TARGET_MODEL_SEARCH_TERMS}")
                        st.write("All model names (first 40):")
                        for mdl in models[:40]:
                            st.caption(mdl.get("name", "?"))
                    else:
                        mid = matched_model["id"]
                        st.success(f"Matched: `{matched_model['name']}` → id=`{mid}`")
    
                        # Get files from metadata.files
                        meta_files = matched_model.get("metadata", {}).get("files", [])
                        st.markdown(f"##### Step 3a: Files from metadata.files ({len(meta_files)} files)")
                        matched_file = None
                        for fi in meta_files:
                            marker = ""
                            if TARGET_QUANTIZATION in fi.get("name", "").lower() or TARGET_QUANTIZATION in fi.get("quantization", "").lower():
                                matched_file = fi
                                marker = " ✅ **MATCH**"
                            st.caption(
                                f"`{fi.get('id')}` — {fi.get('name')} | "
                                f"quant={fi.get('quantization')}{marker}"
                            )
    
                        # Also try the files endpoint
                        st.markdown("##### Step 3b: Files endpoint")
                        for path_variant in [f"/api/models/{mid}/files", f"/models/{mid}/files"]:
                            fr = _raw("GET", path_variant)
                            st.write(f"`{path_variant}` → HTTP {fr.get('status_code')}")
                            if fr.get("status_code") == 200:
                                ep_files = (fr.get("json") or {}).get("data", [])
                                for fi in ep_files:
                                    marker = ""
                                    if TARGET_QUANTIZATION in fi.get("name", "").lower() or TARGET_QUANTIZATION in fi.get("quantization", "").lower():
                                        if not matched_file:
                                            matched_file = fi
                                        marker = " ✅ **MATCH**"
                                    st.caption(
                                        f"`{fi.get('id')}` — {fi.get('name')} | "
                                        f"quant={fi.get('quantization')} | "
                                        f"downloaded={fi.get('isDownloaded')} | "
                                        f"status={fi.get('downloadStatus', {}).get('status', 'n/a')}"
                                        f"{marker}"
                                    )
                                break  # Don't try second path if first worked
                            else:
                                st.caption(f"Response: {fr.get('raw_text', '')[:500]}")
    
                        if not matched_file:
                            st.error(f"No file matched quantization `{TARGET_QUANTIZATION}`")
                        else:
                            fid = matched_file["id"]
                            fname = matched_file.get("name", "?")
                            st.success(f"Matched file: `{fname}` → id=`{fid}`")
    
                        st.markdown("##### Step 4: Server status")
                        sr = _raw("GET", "/api/servers")
                        servers = (sr.get("json") or {}).get("data", [])
                        st.write(f"Found {len(servers)} server(s)")
                        for s in servers:
                            sid = s.get("id", "?")
                            sstat = s.get("status", "?")
                            sport = s.get("server", {}).get("port", "?")
                            smodel = s.get("modelFile", {}).get("name", "?")
                            st.caption(f"`{sid}` — status={sstat} | port={sport} | model={smodel}")
    
                    st.markdown("##### Full diagnostic log")
                    st.code(json.dumps(diag, indent=2, default=str)[:8000], language="json")


# ============================================================
# Main content
# ============================================================
st.title("🏦 Transaction Monitoring Dashboard")
st.caption(f"Inference: **{mode_label}** · {_get_model_label()}")

df = generate_transaction_data()

# ============================================================
# Live Streaming — top of page, non-blocking
# ============================================================
if "stream_results" not in st.session_state:
    st.session_state.stream_results = []
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False
if "stream_batch_idx" not in st.session_state:
    st.session_state.stream_batch_idx = 0
if "stream_batch_size" not in st.session_state:
    st.session_state.stream_batch_size = _STREAM_BATCH_SIZE
if "stream_batch_delay" not in st.session_state:
    st.session_state.stream_batch_delay = _STREAM_BATCH_DELAY
if "stream_total" not in st.session_state:
    st.session_state.stream_total = _STREAM_TOTAL
if "stream_df" not in st.session_state:
    st.session_state.stream_df = df.sample(
        n=min(st.session_state.stream_total, len(df)), random_state=None
    ).reset_index(drop=True)
if "stream_last_batch_time" not in st.session_state:
    st.session_state.stream_last_batch_time = 0.0
if "stream_llm_queue" not in st.session_state:
    st.session_state.stream_llm_queue = []  # borderline txns awaiting LLM
if "stream_llm_calls" not in st.session_state:
    st.session_state.stream_llm_calls = 0
if "stream_rf_auto" not in st.session_state:
    st.session_state.stream_rf_auto = 0  # count of RF-only decisions


def _rf_score_single(row, rf_model):
    """Score a single transaction with the RF model. Returns (score, elapsed_ms)."""
    is_susp = row["country"] in ["NG", "RU", "BR"] or row["amount_eur"] > 5000
    np.random.seed(int(row["amount_eur"] * 100) % 2**31)
    feat = np.append(
        np.random.randn(28) * (3.0 if is_susp else 0.5),
        [row["hour"] * 3600, row["amount_eur"]],
    )
    if hasattr(rf_model, "feature_names_in_"):
        feat = pd.DataFrame([feat], columns=list(rf_model.feature_names_in_))
    else:
        feat = feat.reshape(1, -1)
    t0 = time.perf_counter()
    score = float(rf_model.predict_proba(feat)[:, 1][0])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return score, elapsed_ms


def _parse_batch_response(response_text, batch_rows):
    """Parse multi-line model response, matching lines back to transaction rows."""
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
    parsed = []
    for idx, (_, row) in enumerate(batch_rows.iterrows()):
        risk, action, score = "?", "?", "?"
        matched_line = ""
        txn_id = row["transaction_id"]
        for line in lines:
            if txn_id in line:
                matched_line = line
                break
        if not matched_line and idx < len(lines):
            matched_line = lines[idx]
        for part in matched_line.split():
            if part.startswith("RISK:"):
                risk = part.replace("RISK:", "")
            elif part.startswith("ACTION:"):
                action = part.replace("ACTION:", "")
            elif part.startswith("SCORE:"):
                score = part.replace("SCORE:", "")
        parsed.append({"txn_id": txn_id, "risk": risk, "action": action, "score": score, "line": matched_line[:120]})
    return parsed


def _flush_llm_queue():
    """Send accumulated borderline transactions to the LLM as a batch."""
    queue = st.session_state.stream_llm_queue
    if not queue:
        return

    # Build batch prompt from queued transactions
    txn_texts = [q["txn_text"] for q in queue]
    combined_prompt = "\n---\n".join(txn_texts)

    try:
        resp_text, elapsed, usage = ask_model_timed(
            combined_prompt, system_prompt=STREAMING_FRAUD_PROMPT,
            max_tokens_override=256 * len(queue),
        )
        comp_tokens = usage.get("completion_tokens", 0) or 0
        tok_s = comp_tokens / elapsed if elapsed > 0 and comp_tokens > 0 else 0
        log_inference("Live Streaming (LLM)", elapsed, usage)
        st.session_state.stream_llm_calls += 1

        # Parse LLM response
        lines = [l.strip() for l in resp_text.strip().split("\n") if l.strip()]

        for idx, q in enumerate(queue):
            # Match LLM response line to this transaction
            matched_line = ""
            for line in lines:
                if q["txn_id"] in line:
                    matched_line = line
                    break
            if not matched_line and idx < len(lines):
                matched_line = lines[idx]

            llm_risk, llm_action, llm_score_str = "?", "?", "?"
            for part in matched_line.split():
                if part.startswith("RISK:"):
                    llm_risk = part.replace("RISK:", "")
                elif part.startswith("ACTION:"):
                    llm_action = part.replace("ACTION:", "")
                elif part.startswith("SCORE:"):
                    llm_score_str = part.replace("SCORE:", "")

            # Update the existing result in stream_results
            for r in st.session_state.stream_results:
                if r["txn_id"] == q["txn_id"] and r["stage"] == "⏳ queued":
                    r["risk"] = llm_risk
                    r["action"] = llm_action
                    r["stage"] = "🔀 hybrid"
                    r["batch_latency"] = round(elapsed / len(queue), 3)
                    r["tok_s"] = round(tok_s, 1)
                    r["tokens"] = comp_tokens
                    r["response"] = matched_line[:120]
                    break

    except Exception as e:
        err_detail = f"{type(e).__name__}: {e}"
        for q in queue:
            for r in st.session_state.stream_results:
                if r["txn_id"] == q["txn_id"] and r["stage"] == "⏳ queued":
                    r["risk"] = "ERR"
                    r["action"] = "ERR"
                    r["stage"] = "❌ llm_err"
                    r["response"] = err_detail[:200]
                    break

    st.session_state.stream_llm_queue = []


def _process_one_batch():
    """Process a single batch. RF scores all, borderlines queue for LLM.
    Returns True if there are more batches to process.
    """
    sdf = st.session_state.stream_df
    batch_size = st.session_state.stream_batch_size
    batch_idx = st.session_state.stream_batch_idx
    total_batches = (len(sdf) + batch_size - 1) // batch_size

    if batch_idx >= total_batches:
        return False

    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(sdf))
    batch_rows = sdf.iloc[batch_start:batch_end]
    batch_num = batch_idx + 1

    rf_model = st.session_state.get("rf_model") if PIPELINE_AVAILABLE else None
    thresholds = (st.session_state.get("rf_model_config") or {})
    llm_trigger = thresholds.get("llm_threshold", 0.3)
    block_threshold = thresholds.get("block_threshold", 0.8)

    if rf_model is not None:
        # ---- Hybrid mode: RF first, LLM for borderlines ----
        for _, row in batch_rows.iterrows():
            rf_score, rf_ms = _rf_score_single(row, rf_model)

            if rf_score >= block_threshold:
                # RF confident: BLOCK
                st.session_state.stream_results.append({
                    "txn_id": row["transaction_id"], "amount": row["amount_eur"],
                    "country": row["country"], "category": row["merchant_category"],
                    "fraud": row["ground_truth_fraud"],
                    "risk": "HIGH", "action": "BLOCK",
                    "rf_score": round(rf_score, 4),
                    "batch_latency": round(rf_ms / 1000, 3), "per_txn_latency": round(rf_ms / 1000, 3),
                    "tok_s": 0, "tokens": 0,
                    "batch": batch_num, "stage": "🌲 RF",
                    "response": f"RF score {rf_score:.3f} ≥ {block_threshold} → auto-BLOCK",
                })
                st.session_state.stream_rf_auto += 1

            elif rf_score < llm_trigger:
                # RF confident: APPROVE
                st.session_state.stream_results.append({
                    "txn_id": row["transaction_id"], "amount": row["amount_eur"],
                    "country": row["country"], "category": row["merchant_category"],
                    "fraud": row["ground_truth_fraud"],
                    "risk": "LOW", "action": "APPROVE",
                    "rf_score": round(rf_score, 4),
                    "batch_latency": round(rf_ms / 1000, 3), "per_txn_latency": round(rf_ms / 1000, 3),
                    "tok_s": 0, "tokens": 0,
                    "batch": batch_num, "stage": "🌲 RF",
                    "response": f"RF score {rf_score:.3f} < {llm_trigger} → auto-APPROVE",
                })
                st.session_state.stream_rf_auto += 1

            else:
                # Borderline: queue for LLM
                st.session_state.stream_results.append({
                    "txn_id": row["transaction_id"], "amount": row["amount_eur"],
                    "country": row["country"], "category": row["merchant_category"],
                    "fraud": row["ground_truth_fraud"],
                    "risk": "PENDING", "action": "PENDING",
                    "rf_score": round(rf_score, 4),
                    "batch_latency": 0, "per_txn_latency": 0,
                    "tok_s": 0, "tokens": 0,
                    "batch": batch_num, "stage": "⏳ queued",
                    "response": f"RF score {rf_score:.3f} → borderline, queued for LLM",
                })
                st.session_state.stream_llm_queue.append({
                    "txn_id": row["transaction_id"],
                    "txn_text": format_transaction(row),
                    "rf_score": rf_score,
                })

        log_inference("Live Streaming (RF)", 0.001)

        # Flush LLM queue when it reaches batch_size or stream is ending
        is_last_batch = (batch_idx + 1) >= total_batches
        if len(st.session_state.stream_llm_queue) >= batch_size or is_last_batch:
            if st.session_state.stream_llm_queue:
                _flush_llm_queue()

    else:
        # ---- Fallback: LLM-only mode (no RF model loaded) ----
        txn_texts = [format_transaction(row) for _, row in batch_rows.iterrows()]
        combined_prompt = "\n---\n".join(txn_texts)

        try:
            resp_text, elapsed, usage = ask_model_timed(
                combined_prompt, system_prompt=STREAMING_FRAUD_PROMPT,
                max_tokens_override=256 * len(batch_rows),
            )
            comp_tokens = usage.get("completion_tokens", 0) or 0
            tok_s = comp_tokens / elapsed if elapsed > 0 and comp_tokens > 0 else 0
            per_txn = elapsed / len(batch_rows)
            log_inference("Live Streaming", elapsed, usage)
            parsed = _parse_batch_response(resp_text, batch_rows)

            for p, (_, row) in zip(parsed, batch_rows.iterrows()):
                st.session_state.stream_results.append({
                    "txn_id": p["txn_id"], "amount": row["amount_eur"],
                    "country": row["country"], "category": row["merchant_category"],
                    "fraud": row["ground_truth_fraud"],
                    "risk": p["risk"], "action": p["action"],
                    "rf_score": None,
                    "batch_latency": round(elapsed, 3), "per_txn_latency": round(per_txn, 3),
                    "tok_s": round(tok_s, 1), "tokens": comp_tokens,
                    "batch": batch_num, "stage": "🤖 LLM",
                    "response": p["line"],
                })
        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}"
            if hasattr(e, "status_code"):
                err_detail = f"HTTP {e.status_code}: {e}"
            for _, row in batch_rows.iterrows():
                st.session_state.stream_results.append({
                    "txn_id": row["transaction_id"], "amount": row["amount_eur"],
                    "country": row["country"], "category": row["merchant_category"],
                    "fraud": row["ground_truth_fraud"],
                    "risk": "ERR", "action": "ERR", "rf_score": None,
                    "batch_latency": 0, "per_txn_latency": 0,
                    "tok_s": 0, "tokens": 0,
                    "batch": batch_num, "stage": "❌ err",
                    "response": err_detail[:200],
                })

    st.session_state.stream_batch_idx = batch_idx + 1
    st.session_state.stream_last_batch_time = time.perf_counter()
    return batch_idx + 1 < total_batches


@st.fragment(run_every=timedelta(seconds=1))
def live_streaming_panel():
    """Self-contained streaming panel. Processes one batch per tick, never blocks the page."""
    results = st.session_state.stream_results
    sdf = st.session_state.stream_df
    batch_size = st.session_state.stream_batch_size
    total_batches = (len(sdf) + batch_size - 1) // batch_size
    is_active = st.session_state.stream_active
    is_complete = st.session_state.stream_batch_idx >= total_batches

    # Header row with controls
    hdr1, hdr2, hdr3 = st.columns([4, 1, 1])
    with hdr1:
        st.subheader("⚡ Live Transaction Analysis")
    with hdr2:
        if is_active and not is_complete:
            if st.button("⏹️ Stop", key="stream_stop", use_container_width=True):
                st.session_state.stream_active = False
                st.rerun(scope="fragment")
        else:
            if st.button("▶️ Start", key="stream_go", use_container_width=True):
                if is_complete:
                    st.session_state.stream_results = []
                    st.session_state.stream_batch_idx = 0
                    st.session_state.stream_llm_queue = []
                    st.session_state.stream_llm_calls = 0
                    st.session_state.stream_rf_auto = 0
                    st.session_state.stream_df = df.sample(
                        n=min(st.session_state.stream_total, len(df)), random_state=None
                    ).reset_index(drop=True)
                st.session_state.stream_active = True
                st.session_state.stream_last_batch_time = 0.0
                st.rerun(scope="fragment")
    with hdr3:
        if st.button("🗑️ Clear", key="stream_clear", use_container_width=True):
            st.session_state.stream_results = []
            st.session_state.stream_batch_idx = 0
            st.session_state.stream_active = False
            st.session_state.stream_last_batch_time = 0.0
            st.session_state.stream_llm_queue = []
            st.session_state.stream_llm_calls = 0
            st.session_state.stream_rf_auto = 0
            st.session_state.stream_df = df.sample(
                n=min(st.session_state.stream_total, len(df)), random_state=None
            ).reset_index(drop=True)
            st.rerun(scope="fragment")

    # Config row (dev mode only)
    if DEV_MODE:
        # Track last-saved values to avoid writing every tick
        if "_saved_stream_cfg" not in st.session_state:
            st.session_state._saved_stream_cfg = {
                "total": _STREAM_TOTAL,
                "bs": _STREAM_BATCH_SIZE,
                "delay": _STREAM_BATCH_DELAY,
            }

        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            new_total = st.slider("Total transactions", 10, 200, st.session_state.stream_total, 10, key="cfg_total")
            if new_total != st.session_state.stream_total:
                st.session_state.stream_total = new_total
        with cfg2:
            new_bs = st.slider("Batch size", 1, 20, st.session_state.stream_batch_size, 1, key="cfg_bs")
            if new_bs != st.session_state.stream_batch_size:
                st.session_state.stream_batch_size = new_bs
        with cfg3:
            new_delay = st.slider("Batch delay (s)", 0, 30, st.session_state.stream_batch_delay, 1, key="cfg_delay")
            if new_delay != st.session_state.stream_batch_delay:
                st.session_state.stream_batch_delay = new_delay

        # Persist to config.yml only when values differ from last save
        saved = st.session_state._saved_stream_cfg
        if (new_total != saved["total"]
                or new_bs != saved["bs"]
                or new_delay != saved["delay"]):
            _save_config(streaming={
                "total_transactions": new_total,
                "batch_size": new_bs,
                "batch_delay": new_delay,
            })
            st.session_state._saved_stream_cfg = {
                "total": new_total,
                "bs": new_bs,
                "delay": new_delay,
            }

    # Process one batch if active and delay has elapsed.
    # Skip on the very first render so the full page loads before any API call.
    if is_active and not is_complete:
        if st.session_state.stream_last_batch_time == 0.0 and st.session_state.stream_batch_idx == 0:
            # First render — just mark the start time, process on next tick
            st.session_state.stream_last_batch_time = time.perf_counter()
        else:
            elapsed_since_last = time.perf_counter() - st.session_state.stream_last_batch_time
            if elapsed_since_last >= st.session_state.stream_batch_delay:
                has_more = _process_one_batch()
                if not has_more:
                    st.session_state.stream_active = False
                results = st.session_state.stream_results  # refresh

    # Recalculate with potentially updated batch_size
    batch_size = st.session_state.stream_batch_size
    total_batches = (len(sdf) + batch_size - 1) // batch_size

    # --- Render current state ---
    if not results:
        if is_active:
            st.caption("Starting — first batch will process in a moment...")
        else:
            st.caption("Press **Start** to begin processing transactions.")
        return

    # KPIs
    rf_decided = [r for r in results if r.get("stage", "").startswith("🌲")]
    llm_decided = [r for r in results if r.get("stage", "").startswith("🔀")]
    pending = [r for r in results if r.get("stage") == "⏳ queued"]
    errored = [r for r in results if "err" in r.get("stage", "").lower()]
    has_rf = any(r.get("rf_score") is not None for r in results)

    unique_batch_latencies = []
    seen = set()
    for r in results:
        if r["batch"] not in seen and r["batch_latency"] > 0:
            unique_batch_latencies.append(r["batch_latency"])
            seen.add(r["batch"])
    total_inf = sum(unique_batch_latencies)

    if has_rf:
        # Hybrid mode KPIs
        mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
        mk1.metric("Processed", f"{len(results)} / {len(sdf)}")
        mk2.metric("🌲 RF Auto", f"{len(rf_decided)}")
        mk3.metric("🔀 LLM Hybrid", f"{len(llm_decided)}")
        mk4.metric("⏳ Pending", f"{len(pending)}")
        mk5.metric("LLM Calls", f"{st.session_state.stream_llm_calls}")
        llm_pct = len(llm_decided) / max(len(results), 1) * 100
        mk6.metric("LLM Rate", f"{llm_pct:.0f}%")
    else:
        # LLM-only mode KPIs
        per_txn_latencies = [r["per_txn_latency"] for r in results if r["per_txn_latency"] > 0]
        tok_rates = [r["tok_s"] for r in results if r["tok_s"] > 0]
        mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
        mk1.metric("Processed", f"{len(results)} / {len(sdf)}")
        mk2.metric("API Calls", f"{len(seen)} / {total_batches}")
        mk3.metric("Avg Batch", f"{np.mean(unique_batch_latencies):.2f}s" if unique_batch_latencies else "—")
        mk4.metric("Avg Per-Txn", f"{np.mean(per_txn_latencies):.2f}s" if per_txn_latencies else "—")
        mk5.metric("Avg tok/s", f"{np.mean(tok_rates):.0f}" if tok_rates else "—")
        mk6.metric("Throughput", f"{len(results) / total_inf * 60:.0f} txn/min" if total_inf > 0 else "—")

    # Progress
    st.progress(len(results) / len(sdf))

    # Chart + log side by side
    ch_col, log_col = st.columns([2, 1])
    with ch_col:
        if unique_batch_latencies:
            fig_live = go.Figure()
            fig_live.add_trace(go.Bar(
                x=[f"B{i+1}" for i in range(len(unique_batch_latencies))],
                y=unique_batch_latencies, marker_color="#1565c0",
                text=[f"{v:.1f}s" for v in unique_batch_latencies], textposition="outside",
            ))
            fig_live.add_hline(y=np.mean(unique_batch_latencies), line_dash="dash",
                               line_color="#d32f2f",
                               annotation_text=f"avg {np.mean(unique_batch_latencies):.2f}s")
            fig_live.update_layout(
                title=f"Batch Latency — {batch_size} txns/call", height=250,
                xaxis_title="Batch", yaxis_title="Seconds", margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_live, use_container_width=True)

    with log_col:
        # Flagged transactions pinned at top (persist throughout the run)
        flagged = [r for r in results if r.get("action") in ("BLOCK", "REVIEW")
                   or r.get("risk") in ("HIGH", "MEDIUM")]
        if flagged:
            st.markdown("**🚨 Flagged**")
            for r in flagged:
                stage = r.get("stage", "")
                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡"}.get(r["risk"], "🔴")
                rf_tag = f" RF:{r['rf_score']:.2f}" if r.get("rf_score") is not None else ""
                st.caption(
                    f"{risk_color} `{r['txn_id']}` €{r['amount']:,.2f} "
                    f"{r['country']} → **{r['action']}**{rf_tag} {stage}"
                )
            st.markdown("---")

        # Recent activity feed (last 8, excluding already-shown flagged)
        flagged_ids = {r["txn_id"] for r in flagged}
        recent = [r for r in results if r["txn_id"] not in flagged_ids][-8:]
        if recent:
            st.markdown("**Recent**")
            for r in reversed(recent):
                stage = r.get("stage", "")
                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "PENDING": "⏳"}.get(r["risk"], "⚪")
                rf_tag = f" RF:{r['rf_score']:.2f}" if r.get("rf_score") is not None else ""
                st.caption(
                    f"{risk_color} `{r['txn_id']}` €{r['amount']:,.2f} "
                    f"{r['country']} → **{r['action']}**{rf_tag}"
                )

    if is_complete:
        has_rf = any(r.get("rf_score") is not None for r in results)
        if has_rf:
            rf_count = sum(1 for r in results if r.get("stage", "").startswith("🌲"))
            llm_count = sum(1 for r in results if r.get("stage", "").startswith("🔀"))
            llm_calls = st.session_state.stream_llm_calls
            st.success(
                f"Complete: {len(results)} transactions · "
                f"🌲 RF auto-decided: {rf_count} · "
                f"🔀 LLM reviewed: {llm_count} in {llm_calls} API call(s) · "
                f"LLM savings: {(1 - llm_count / max(len(results), 1)) * 100:.0f}%"
            )
        else:
            st.success(
                f"Complete: {len(results)} transactions in {len(seen)} API calls "
                f"({total_inf:.1f}s inference)"
            )


live_streaming_panel()

st.divider()

# ---- KPI Row ----
total_volume = df["amount_eur"].sum()
total_count = len(df)
fraud_count = int(df["ground_truth_fraud"].sum())
fraud_rate = fraud_count / total_count * 100
high_risk_count = int((df["amount_eur"] > 5000).sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Volume", f"€{total_volume:,.0f}")
k2.metric("Transactions", f"{total_count:,}")
k3.metric("Flagged Fraud", fraud_count)
k4.metric("Fraud Rate", f"{fraud_rate:.1f}%")
k5.metric("High Value (>€5k)", high_risk_count)

st.divider()

# ---- Filters ----
with st.expander("🔍 Filters", expanded=False):
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sel_countries = st.multiselect("Country", sorted(df["country"].unique()), key="filter_country")
    with fc2:
        sel_categories = st.multiselect("Merchant Category", sorted(df["merchant_category"].unique()), key="filter_cat")
    with fc3:
        fraud_opts = {"All": None, "Fraudulent Only": True, "Legitimate Only": False}
        sel_fraud = st.selectbox("Fraud Status", list(fraud_opts.keys()), key="filter_fraud")
    with fc4:
        amount_range = st.slider(
            "Amount Range (EUR)", 0.0, float(df["amount_eur"].max()) + 100,
            (0.0, float(df["amount_eur"].max()) + 100), key="filter_amount",
        )

# Apply filters
filtered = df.copy()
if sel_countries:
    filtered = filtered[filtered["country"].isin(sel_countries)]
if sel_categories:
    filtered = filtered[filtered["merchant_category"].isin(sel_categories)]
fv = fraud_opts[sel_fraud]
if fv is not None:
    filtered = filtered[filtered["ground_truth_fraud"] == fv]
filtered = filtered[
    (filtered["amount_eur"] >= amount_range[0]) & (filtered["amount_eur"] <= amount_range[1])
]

# ---- Charts ----
ch1, ch2 = st.columns(2)

with ch1:
    fig_scatter = px.scatter(
        filtered, x="timestamp", y="amount_eur",
        color="ground_truth_fraud",
        color_discrete_map={True: "#d32f2f", False: "#1565c0"},
        hover_data=["transaction_id", "merchant_category", "country"],
        labels={"amount_eur": "Amount (EUR)", "timestamp": "Time", "ground_truth_fraud": "Fraud"},
        title="Transaction Amounts Over Time",
    )
    fig_scatter.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with ch2:
    country_counts = filtered["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]
    high_risk = ["NG", "RU", "BR"]
    country_counts["risk"] = country_counts["country"].apply(lambda c: "High-Risk" if c in high_risk else "Normal")
    fig_bar = px.bar(
        country_counts, x="country", y="count", color="risk",
        color_discrete_map={"High-Risk": "#d32f2f", "Normal": "#1565c0"},
        title="Transaction Count by Country",
        labels={"count": "Count", "country": "Country"},
    )
    fig_bar.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

ch3, ch4 = st.columns(2)

with ch3:
    cat_vol = filtered.groupby("merchant_category")["amount_eur"].sum().sort_values().reset_index()
    fig_cat = px.bar(
        cat_vol, y="merchant_category", x="amount_eur", orientation="h",
        title="Volume by Merchant Category (EUR)",
        labels={"amount_eur": "Total EUR", "merchant_category": ""},
        color_discrete_sequence=["#1565c0"],
    )
    fig_cat.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_cat, use_container_width=True)

with ch4:
    fraud_data = filtered[filtered["ground_truth_fraud"]]["amount_eur"]
    legit_data = filtered[~filtered["ground_truth_fraud"]]["amount_eur"]
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=legit_data, name="Legitimate", marker_color="#1565c0"))
    fig_box.add_trace(go.Box(y=fraud_data, name="Fraudulent", marker_color="#d32f2f"))
    fig_box.update_layout(title="Amount Distribution: Legit vs Fraud", height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ---- Transaction Table ----
st.subheader("Transaction Details")
display_cols = [
    "transaction_id", "timestamp", "card_number", "amount_eur",
    "merchant_category", "country", "is_online", "ground_truth_fraud",
]

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    height=400,
    column_config={
        "amount_eur": st.column_config.NumberColumn("Amount (EUR)", format="€%.2f"),
        "ground_truth_fraud": st.column_config.CheckboxColumn("Fraud?"),
        "is_online": st.column_config.CheckboxColumn("Online?"),
    },
)

st.divider()

# ============================================================
# Tabbed analysis sections
# ============================================================
_tab_names = [
    "🔍 AI Fraud Screening",
    "📊 Batch Pattern Analysis",
    "📋 Risk Report",
    "⚖️ Compliance Q&A",
    "📑 Data Extraction",
    "💬 Analyst Chat",
    "🏁 Benchmark Log",
]
if PIPELINE_AVAILABLE and DEV_MODE:
    _tab_names.append("🔧 ML Pipeline")

tabs = st.tabs(_tab_names)

# --- Tab 0: AI Fraud Screening ---
with tabs[0]:
    st.subheader("AI-Powered Fraud Screening")
    st.markdown("Select a transaction ID to send to the model for a structured risk assessment.")

    # Hybrid mode toggle (only if pipeline is available and model loaded)
    _use_hybrid = False
    if PIPELINE_AVAILABLE:
        _use_hybrid = render_hybrid_toggle()

    txn_id = st.selectbox(
        "Transaction ID",
        filtered["transaction_id"].tolist(),
        key="fraud_screen_txn",
    )

    if st.button("Analyse Transaction", key="fraud_screen_btn"):
        row = filtered[filtered["transaction_id"] == txn_id].iloc[0]
        ground_truth = "FRAUD" if row["ground_truth_fraud"] else "LEGITIMATE"
        st.info(f"Ground truth: **{ground_truth}**")

        if _use_hybrid:
            # Hybrid path: RF screens first, LLM only for borderline
            # Build synthetic feature vector matching the RF model's training schema
            # (V1-V28 PCA features + Time + Amount)
            is_susp = row["country"] in ["NG", "RU", "BR"] or row["amount_eur"] > 5000
            np.random.seed(int(row["amount_eur"] * 100) % 2**31)
            feature_values = np.append(
                np.random.randn(28) * (3.0 if is_susp else 0.5),
                [row["hour"] * 3600, row["amount_eur"]],
            )

            with st.spinner("Hybrid scoring (RF + LLM)..."):
                try:
                    h_result = score_transaction_hybrid(
                        features_array=feature_values,
                        merchant=row["merchant_category"],
                        amount=row["amount_eur"],
                        ask_model_fn=ask_model,
                    )
                except Exception as e:
                    st.error(f"Hybrid scoring error: {e}")
                    h_result = None

            if h_result and "error" not in h_result:
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("RF Score", f"{h_result['rf_score']:.4f}")
                if h_result["llm_score"] is not None:
                    hc2.metric("LLM Score", f"{h_result['llm_score']:.4f}")
                elif h_result.get("llm_error"):
                    hc2.metric("LLM Score", "⚠️ failed")
                else:
                    hc2.metric("LLM Score", "— (below threshold)")
                hc3.metric("Combined", f"{h_result['combined_score']:.4f}")
                hc4.metric("Decision", h_result["decision"])

                stage_label = h_result["stage"]
                caption_parts = [
                    f"Stage: **{stage_label}**",
                    f"RF: {h_result['rf_elapsed_ms']:.1f}ms",
                ]
                if h_result.get("llm_elapsed_ms"):
                    caption_parts.append(f"LLM: {h_result['llm_elapsed_ms']:.1f}ms")
                st.caption(" · ".join(caption_parts))

                # Show LLM error detail if the call was attempted but failed
                if h_result.get("llm_error"):
                    st.warning(f"LLM was triggered but failed: {h_result['llm_error']}")

                # Log combined timing
                total_ms = h_result["rf_elapsed_ms"] + (h_result.get("llm_elapsed_ms") or 0)
                log_inference("Fraud Screening (Hybrid)", total_ms / 1000)
            elif h_result:
                st.error(h_result["error"])
        else:
            # LLM-only path (original behaviour)
            txn_text = format_transaction(row)
            with st.spinner("Querying model..."):
                try:
                    result, elapsed, usage = ask_model_timed(txn_text, system_prompt=FRAUD_SYSTEM_PROMPT)
                    show_timing(elapsed, usage)
                    log_inference("Fraud Screening", elapsed, usage)
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Inference error: {e}")

# --- Tab 1: Batch Pattern Analysis ---
with tabs[1]:
    st.subheader("Batch Transaction Analysis — Pattern Detection")
    st.markdown(
        "Sends a batch of 20 transactions (including the card-testing cluster) "
        "to the model to detect cross-transaction fraud patterns."
    )

    if st.button("Run Batch Analysis", key="batch_btn"):
        card_test_txns = df[df["card_number"] == "****-****-****-7721"]
        other_txns = df[df["card_number"] != "****-****-****-7721"].sample(15, random_state=42)
        batch = pd.concat([card_test_txns, other_txns]).sort_values("timestamp")

        # RF pre-screening if model is loaded
        if PIPELINE_AVAILABLE and st.session_state.get("rf_model") is not None:
            st.markdown("##### Stage 1: Random Forest Pre-Screening")
            rf_model = st.session_state.rf_model
            rf_rows = []
            for _, row in batch.iterrows():
                is_susp = row["country"] in ["NG", "RU", "BR"] or row["amount_eur"] > 5000
                np.random.seed(int(row["amount_eur"] * 100) % 2**31)
                feat = np.append(
                    np.random.randn(28) * (3.0 if is_susp else 0.5),
                    [row["hour"] * 3600, row["amount_eur"]],
                )
                # Wrap in DataFrame with correct feature names
                if hasattr(rf_model, "feature_names_in_"):
                    import pandas as _pd
                    feat = _pd.DataFrame([feat], columns=list(rf_model.feature_names_in_))
                else:
                    feat = feat.reshape(1, -1)
                score = float(rf_model.predict_proba(feat)[:, 1][0])
                decision = "🔴 BLOCK" if score >= 0.8 else "🟡 REVIEW" if score >= 0.5 else "🟢 APPROVE"
                rf_rows.append({
                    "Txn ID": row["transaction_id"],
                    "Amount": f"€{row['amount_eur']:,.2f}",
                    "Country": row["country"],
                    "Category": row["merchant_category"],
                    "RF Score": round(score, 4),
                    "RF Decision": decision,
                    "Fraud?": "🔴" if row["ground_truth_fraud"] else "",
                })
            rf_df = pd.DataFrame(rf_rows)
            st.dataframe(rf_df, use_container_width=True, hide_index=True)
            flagged = sum(1 for r in rf_rows if r["RF Score"] >= 0.5)
            st.caption(f"RF flagged {flagged}/{len(rf_rows)} transactions in <1ms")
            st.markdown("##### Stage 2: LLM Cross-Transaction Pattern Analysis")

        batch_text = "\n\n".join([format_transaction(row) for _, row in batch.iterrows()])

        with st.spinner(f"Sending {len(batch)} transactions for pattern analysis..."):
            try:
                result, elapsed, usage = ask_model_timed(batch_text, system_prompt=BATCH_SYSTEM_PROMPT)
                show_timing(elapsed, usage)
                log_inference("Batch Analysis", elapsed, usage)
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 2: Risk Report ---
with tabs[2]:
    st.subheader("Portfolio Risk Summarisation")
    st.markdown("Generates an executive-level daily risk briefing from aggregated transaction statistics.")

    if st.button("Generate Risk Report", key="risk_btn"):
        base_date = datetime(2026, 2, 20)
        summary_stats = {
            "total_transactions": len(df),
            "total_volume_eur": f"{df['amount_eur'].sum():,.2f}",
            "avg_transaction_eur": f"{df['amount_eur'].mean():,.2f}",
            "max_transaction_eur": f"{df['amount_eur'].max():,.2f}",
            "online_pct": f"{df['is_online'].mean()*100:.1f}%",
            "top_countries": df["country"].value_counts().head(5).to_dict(),
            "top_categories": df["merchant_category"].value_counts().head(5).to_dict(),
            "high_value_txns_over_5000": int((df["amount_eur"] > 5000).sum()),
            "late_night_txns_midnight_to_5am": int(df["hour"].between(0, 5).sum()),
            "cross_border_pct": f"{(df['country'] != 'DE').mean()*100:.1f}%",
        }
        data_prompt = (
            f"Daily Transaction Monitoring Data — {base_date.strftime('%d %B %Y')}:\n\n"
            f"{json.dumps(summary_stats, indent=2)}"
        )

        with st.spinner("Generating executive risk briefing..."):
            try:
                result, elapsed, usage = ask_model_timed(data_prompt, system_prompt=RISK_SYSTEM_PROMPT)
                show_timing(elapsed, usage)
                log_inference("Risk Report", elapsed, usage)
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 3: Compliance Q&A ---
with tabs[3]:
    st.subheader("Regulatory Compliance Q&A")

    default_questions = [
        "What are the key requirements under the EU's 6th Anti-Money Laundering Directive (6AMLD) that a bank must implement for transaction monitoring?",
        "Under PSD2, what are a bank's obligations when a customer reports an unauthorized transaction? What are the liability rules and timeframes?",
    ]

    question = st.selectbox("Select a question", default_questions, key="compliance_q")
    custom_q = st.text_area("Or type a custom question", key="compliance_custom")

    if st.button("Ask Compliance Question", key="compliance_btn"):
        q = custom_q.strip() if custom_q.strip() else question
        with st.spinner("Querying model..."):
            try:
                result, elapsed, usage = ask_model_timed(q, system_prompt=COMPLIANCE_SYSTEM_PROMPT)
                show_timing(elapsed, usage)
                log_inference("Compliance Q&A", elapsed, usage)
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 4: Data Extraction ---
with tabs[4]:
    st.subheader("Structured Data Extraction")
    st.markdown("Extract structured JSON from unstructured text (e.g., Suspicious Activity Reports).")

    sample_email = """Subject: Suspicious Activity Report — Priority

Dear Compliance Team,

I am writing to flag a series of transactions on account DE89370400440532013000 belonging to 
Müller GmbH (client ID: CL-2024-88123). Between 18 February and 22 February 2026, we observed 
17 outbound wire transfers totalling EUR 2.3 million to three separate beneficiary accounts in 
the Cayman Islands (KY) and British Virgin Islands (VG). 

The average transaction size was approximately EUR 135,000, which is significantly above the 
client's normal operating pattern of EUR 15,000-25,000 monthly outflows. The transfers were 
initiated outside of normal business hours (between 23:00 and 04:00 CET).

The client's KYC profile was last updated on 15 March 2025 and lists the business as a 
small import/export firm with annual revenue of EUR 800,000.

Please advise on next steps.

Regards,
Anna Schmidt
Transaction Monitoring, Frankfurt"""

    email_text = st.text_area("Input text", value=sample_email, height=250, key="extract_input")

    if st.button("Extract Data", key="extract_btn"):
        extraction_prompt = f"""Extract the following fields from the email below and return as JSON:
- account_number
- client_name
- client_id
- date_range
- num_transactions
- total_amount_eur
- avg_transaction_eur
- destination_jurisdictions
- normal_monthly_outflow_range_eur
- annual_revenue_eur
- kyc_last_updated
- reporting_analyst
- risk_indicators (list)

Email:
{email_text}"""

        with st.spinner("Extracting structured data..."):
            try:
                result, elapsed, usage = ask_model_timed(extraction_prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
                show_timing(elapsed, usage)
                log_inference("Data Extraction", elapsed, usage)
                st.code(result, language="json")
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 5: Analyst Chat ---
with tabs[5]:
    st.subheader("Interactive Analyst Chat")

    system_role = st.text_input(
        "System prompt",
        value="You are a knowledgeable AI strategy advisor for the financial services industry. Provide balanced, practical advice.",
        key="chat_system",
    )
    user_question = st.text_area(
        "Your question",
        value="What are the key differences between running AI inference locally versus in the cloud for a regulated financial institution? Consider data sovereignty, latency, compliance, and cost.",
        key="chat_question",
    )

    if st.button("Send", key="chat_send"):
        with st.spinner("Thinking..."):
            try:
                result, elapsed, usage = ask_model_timed(user_question, system_prompt=system_role)
                show_timing(elapsed, usage)
                log_inference("Analyst Chat", elapsed, usage)
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference error: {e}")

# --- Tab 6: Benchmark Log ---
with tabs[6]:
    st.subheader("🏁 Model Benchmark Log")
    st.markdown(
        "Average response time by task type, model, and inference location. "
        "Every inference call across all tabs is logged automatically."
    )

    @st.fragment(run_every=timedelta(seconds=3))
    def benchmark_panel():
        blog = st.session_state.benchmark_log

        if not blog:
            st.info("No inference calls recorded yet. Use any analysis tab or the live streaming panel.")
            return

        blog_df = pd.DataFrame(blog)

        # --- Aggregate table: avg latency by task × model × mode ---
        agg = blog_df.groupby(["task", "model", "mode"]).agg(
            calls=("latency", "count"),
            avg_latency=("latency", "mean"),
            p95_latency=("latency", lambda x: np.percentile(x, 95) if len(x) >= 2 else x.iloc[0]),
            avg_tok_s=("tok_s", "mean"),
            total_tokens=("tokens", "sum"),
        ).reset_index().sort_values(["task", "avg_latency"])

        agg["avg_latency"] = agg["avg_latency"].round(2)
        agg["p95_latency"] = agg["p95_latency"].round(2)
        agg["avg_tok_s"] = agg["avg_tok_s"].round(0)

        st.markdown("##### Average Response Time by Task")
        st.dataframe(
            agg,
            use_container_width=True,
            column_config={
                "task": st.column_config.TextColumn("Task"),
                "model": st.column_config.TextColumn("Model"),
                "mode": st.column_config.TextColumn("Mode"),
                "calls": st.column_config.NumberColumn("Calls", format="%d"),
                "avg_latency": st.column_config.NumberColumn("Avg (s)", format="%.2f"),
                "p95_latency": st.column_config.NumberColumn("P95 (s)", format="%.2f"),
                "avg_tok_s": st.column_config.NumberColumn("Avg tok/s", format="%.0f"),
                "total_tokens": st.column_config.NumberColumn("Total Tokens", format="%d"),
            },
            hide_index=True,
        )

        # --- Chart: avg latency by task, grouped by model ---
        if len(agg) >= 1:
            fig_compare = px.bar(
                agg, x="task", y="avg_latency", color="model",
                barmode="group",
                text=agg["avg_latency"].apply(lambda v: f"{v:.2f}s"),
                labels={"avg_latency": "Avg Latency (s)", "task": "", "model": "Model"},
                color_discrete_sequence=["#1565c0", "#43B049", "#d32f2f", "#f57c00", "#7b1fa2"],
            )
            fig_compare.update_layout(
                title="Average Response Time by Task & Model", height=400,
                margin=dict(t=40, b=20),
            )
            fig_compare.update_traces(textposition="outside")
            st.plotly_chart(fig_compare, use_container_width=True)

        # --- Chart: avg tok/s by model (across all tasks) ---
        model_agg = blog_df.groupby(["model", "mode"]).agg(
            total_calls=("latency", "count"),
            avg_latency=("latency", "mean"),
            avg_tok_s=("tok_s", "mean"),
        ).reset_index()

        if len(model_agg) >= 2:
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_toks = px.bar(
                    model_agg, x="model", y="avg_tok_s",
                    color="mode", color_discrete_map={"LOCAL": "#43B049", "CLOUD": "#1565c0"},
                    text=model_agg["avg_tok_s"].round(0).astype(int).astype(str),
                    labels={"avg_tok_s": "Avg tok/s", "model": ""},
                )
                fig_toks.update_layout(title="Avg Tokens/Second by Model", height=350, margin=dict(t=40, b=80))
                fig_toks.update_traces(textposition="outside")
                st.plotly_chart(fig_toks, use_container_width=True)

            with ch2:
                fig_calls = px.bar(
                    model_agg, x="model", y="total_calls",
                    color="mode", color_discrete_map={"LOCAL": "#43B049", "CLOUD": "#1565c0"},
                    text=model_agg["total_calls"].astype(str),
                    labels={"total_calls": "Total Calls", "model": ""},
                )
                fig_calls.update_layout(title="Total Inference Calls by Model", height=350, margin=dict(t=40, b=80))
                fig_calls.update_traces(textposition="outside")
                st.plotly_chart(fig_calls, use_container_width=True)

        # --- Raw log (collapsed) ---
        with st.expander("📋 Raw Inference Log", expanded=False):
            st.dataframe(
                blog_df[["timestamp", "task", "model", "mode", "latency", "tokens", "tok_s"]],
                use_container_width=True, height=300,
                column_config={
                    "latency": st.column_config.NumberColumn("Latency (s)", format="%.2f"),
                    "tok_s": st.column_config.NumberColumn("tok/s", format="%.0f"),
                    "tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                },
            )

        if st.button("🗑️ Clear Benchmark Log", key="clear_bench"):
            st.session_state.benchmark_log = []
            st.rerun()

    benchmark_panel()

# --- Tab 7: ML Pipeline (dev mode only) ---
if PIPELINE_AVAILABLE and DEV_MODE:
    with tabs[7]:
        render_pipeline_tab(
            pipeline_mode=PIPELINE_MODE,
            flows_dir=FLOWS_DIR,
        )