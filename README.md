# Credit Risk Fraud Detection Dashboard

**Anaconda AI Catalyst + Metaflow ML Pipeline Demo**

An SE demo asset showing the full lifecycle from data science notebook to production-deployed application, using Anaconda's platform and Outerbounds infrastructure.

---

## What This Demonstrates

This project exists to show prospects how Anaconda's toolchain supports the entire ML workflow — not just package management, but model development, governance, deployment, and monitoring. The demo narrative walks through three phases that mirror a real enterprise adoption path.

### Phase 1: Notebook (Data Science)

The starting point is `credit_risk_fraud_analysis_llm.ipynb` — a Jupyter notebook that analyses 200 synthetic credit card transactions using LLM inference via Anaconda AI Catalyst. It covers single-transaction fraud screening, batch pattern detection, risk reporting, compliance Q&A, and structured data extraction. This is what a data scientist produces in an ungoverned environment.

### Phase 2: Streamlit Dashboard (Application)

The notebook was converted into `app.py` — a full Streamlit dashboard (~2,500 lines) with:

- **Live streaming panel** — RF model scores each transaction individually (<1ms), borderline cases queue up and batch to the LLM. Flagged transactions pin to the top of the feed.
- **AI Fraud Screening** — single transaction analysis with a hybrid toggle (LLM Only / Random Forest + LLM).
- **Batch Pattern Analysis** — RF pre-screening table + LLM cross-transaction pattern detection.
- **Risk Report, Compliance Q&A, Data Extraction, Analyst Chat** — LLM-powered analysis tabs.
- **Benchmark Log** — auto-records every inference call with latency, tokens/sec, model, and mode.
- **ML Pipeline Manager** (dev mode) — trigger Metaflow flows, view run history, load trained models.

Inference switches between Cloud (AI Catalyst hosted endpoint) and Local (Anaconda Desktop running llama.cpp on localhost:8080). The Anaconda Desktop API integration includes a full model catalog browser with download/pause/resume/cancel, server lifecycle management, and auto-detection of running servers.

### Phase 3: Deployed Application (Production)

`deploy.sh` automates the full pipeline:
1. Trains a Random Forest model on Outerbounds K8s via Metaflow
2. Deploys the Streamlit dashboard to Outerbounds
3. The deployed app loads the trained model via the Metaflow Client API

In prod mode, config panels, debug tools, API keys, and the Pipeline Manager tab are hidden. The hybrid scoring toggle remains visible.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│                                                             │
│  Live Streaming ──► RF scores all ──► borderlines to LLM   │
│  Fraud Screening ──► LLM Only / Hybrid toggle              │
│  Batch Analysis ──► RF pre-screen + LLM patterns           │
│                                                             │
│  Inference:  Cloud (AI Catalyst)  /  Local (Desktop)        │
│  ML Model:   Loaded from Metaflow artifacts on startup      │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
    ┌──────────▼──────────┐       ┌───────────▼────────────┐
    │  Anaconda AI Catalyst│       │  Metaflow Pipeline     │
    │  (hosted LLM endpoint)│      │  FraudDataPrepFlow     │
    │                      │       │  FraudTrainingFlow     │
    │  Anaconda Desktop    │       │  FraudScoringFlow      │
    │  (local llama.cpp)   │       │  (local or K8s)        │
    └──────────────────────┘       └────────────────────────┘
```

---

## Project Structure

```
dashboard-demo/
├── app.py                  # Main Streamlit dashboard
├── pipeline_client.py      # Metaflow Client API wrapper (load models, list runs)
├── pipeline_ui.py          # Pipeline Manager tab + hybrid scoring UI components
├── config.yml              # Runtime config (gitignored — contains API keys)
├── config.yml.example      # Config template (committed)
├── deploy.sh               # Train on K8s + deploy to Outerbounds
├── requirements.txt        # Pip deps for Outerbounds deploy
├── dashboard-demo.yml      # Conda environment definition
├── credit_risk_fraud_analysis_llm.ipynb  # Source notebook
├── data/
│   ├── creditcard.csv      # Kaggle dataset (gitignored, ~150MB)
│   └── README.md           # Download instructions
├── flows/                  # Metaflow flows
│   ├── data_prep_flow.py / _local.py
│   ├── training_flow.py / _local.py
│   └── scoring_flow.py / _local.py
└── .metaflow/              # Metaflow local datastore 
```

---

## Running Locally

### Prerequisites

- Anaconda or Miniconda
- Metaflow (`pip install metaflow`)
- Anaconda Desktop (optional — for local LLM inference)
- The Kaggle credit card fraud dataset in `data/creditcard.csv`

### Setup

```bash
# Clone the repo
git clone git@github.com:melliott-anaconda/dashboard-demo.git
cd dashboard-demo

# Create conda environment
conda env create -f dashboard-demo.yml
conda activate dashboard-demo

# Copy and edit config
cp config.yml.example config.yml
# Fill in cloud_url and cloud_api_key (get from AI Catalyst)

# Download the dataset directly
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud → data/creditcard.csv
```

### Train a model locally

```bash
cd ~/Developer/dashboard-demo

# Step 1: Data prep
python flows/data_prep_flow_local.py run
# Note the run ID printed at the end

# Step 2: Training (4-config hyperparameter search)
python flows/training_flow_local.py run --data_run_id <data_run_id>
```

The dashboard auto-loads the latest trained model on startup (`auto_load_model: true` in config).

### Run the dashboard

```bash
cd ~/Developer/dashboard-demo
streamlit run app.py
```

In dev mode (`mode: dev` in config.yml), you get:
- Cloud/local inference toggle
- Endpoint URL and API key fields
- Model parameter sliders
- ML Pipeline Manager tab
- Debug panels and connection test

### Config

`config.yml` controls everything. Key settings:

| Key | Values | Effect |
|---|---|---|
| `mode` | `dev` / `prod` | Shows/hides config UI, debug tools, Pipeline tab |
| `cloud_url` | URL | AI Catalyst endpoint |
| `pipeline.auto_load_model` | `true` / `false` | Load RF model from Metaflow on startup |
| `pipeline.mode` | `local` / `kubernetes` | Which flow variants the Pipeline tab triggers |
| `streaming.*` | integers | Live streaming defaults (auto-saved on slider change) |

---

## Deploying to Outerbounds

### Prerequisites

- `outerbounds` CLI installed and authenticated (`outerbounds configure <KEY>`)
- A locked cloud endpoint (use the Lock button in dev mode)
- Metaflow flows updated with `@pypi_base(python='3.12', packages=DEPS)` for K8s

### Deploy

```bash
cd ~/Developer/dashboard-demo

# Full pipeline: train on K8s → deploy dashboard
export OBP_PERIMETER_KEY="<your-outerbounds-perimeter-key>"
./deploy.sh

# Or just deploy (skip training, use latest model)
./deploy.sh --deploy-only

# Or just train (no app deploy)
./deploy.sh --train-only

# Or train locally then deploy
./deploy.sh --local-train
```

The deploy script:
1. Runs `FraudDataPrepFlow` and `FraudTrainingFlow` on K8s (chained via `--run-id-file`)
2. Swaps `config.yml` to prod mode (auto-restores on exit via trap)
3. Deploys to Outerbounds with `outerbounds app deploy`
4. The entrypoint runs `outerbounds configure` on the pod so the app can access Metaflow artifacts

### Prod mode behaviour

- Inference locked to Cloud (AI Catalyst) — no toggle
- Pipeline Manager tab hidden
- Config panels, debug tools, API keys hidden
- Hybrid scoring toggle visible (if model loaded)
- Sidebar shows model name, server status, active endpoint

---

## Metaflow Pipeline

Three flows, each with a local (`_local.py`, no decorators) and K8s variant (`@pypi_base`, `@resources`):

| Flow | Input | Output | Purpose |
|---|---|---|---|
| `FraudDataPrepFlow` | `creditcard.csv` | Train/test split with merchant descriptions | Data preparation |
| `FraudTrainingFlow` | `data_run_id` | Best RF model (4-config grid search by F1) | Model training |
| `FraudScoringFlow` | `training_run_id` | Batch predictions on 100 synthetic transactions | Validation |

Flows chain via run IDs. The dashboard loads the trained model from `FraudTrainingFlow` artifacts using the Metaflow Client API (`pipeline_client.py`).

### Hybrid scoring architecture

When a model is loaded, the live streaming panel operates in two stages:

1. **RF individually** (~0.01ms per transaction) — scores below 0.3 auto-approve, scores ≥ 0.8 auto-block
2. **LLM batch** — borderline scores (0.3–0.8) accumulate in a queue, sent to the LLM when the queue reaches batch size or the stream ends

This typically cuts LLM API calls by 70–85% while maintaining detection quality on edge cases.

---

## Key Implementation Details

**Inference client**: `_build_client()` returns an OpenAI-compatible client. For Outerbounds endpoints (`merced.obp.outerbounds.com` in URL), adds `x-api-key` header and sets `model="catalyst"`.

**503 retry**: Cloud endpoints return 503 after ~2 concurrent requests. `ask_model_timed()` retries up to 5× with exponential backoff starting at 3s.

**Dev/prod config**: Single `config.yml` with `mode: dev|prod`. The deploy script swaps to prod before deploy and restores on exit via a bash trap.

**Streaming config persistence**: Slider values auto-save to `config.yml` when changed. Session state tracks last-saved values to avoid writing every fragment tick.

**Feature names**: The RF model was trained on a DataFrame with column names (`V1`–`V28`, `Time`, `Amount`). The dashboard wraps feature arrays in a DataFrame with `model.feature_names_in_` before calling `predict_proba` to avoid sklearn warnings.

---

## Troubleshooting

**"metaflow not installed"** — Check that `pipeline_client.py` is the correct file (472 lines, starts with "Pipeline Client"), not an accidental copy of `app.py`.

**`monotonic_cst` AttributeError** — sklearn version mismatch between training and dashboard. Pin both to the same version.

**"No completed training runs found"** — The Metaflow datastore isn't accessible. Locally: check `.metaflow/` has a current training run. On deploy: ensure `outerbounds configure` runs in the pod entrypoint.

**LLM not triggering on borderline scores** — The threshold uses `>=` (inclusive). Scores exactly at 0.3 trigger the LLM. Check the `pipeline.scoring.llm_trigger` value in `config.yml`.

---

## Contact

Michael Elliott — Solutions Engineer, EMEA
