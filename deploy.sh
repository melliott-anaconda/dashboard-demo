#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# deploy.sh — Promote best model + deploy dashboard to Outerbounds
#
# Usage:
#   ./deploy.sh                  # promote best local config to K8s → deploy
#   ./deploy.sh --deploy-only    # skip training, deploy with latest model
#   ./deploy.sh --train-only     # run pipeline on K8s, don't deploy app
#   ./deploy.sh --local-train    # full grid search locally, then deploy app
#
# Training behaviour:
#   By default, the script reads the best hyperparameters from your latest
#   local FraudTrainingFlow run and promotes that SINGLE config to K8s.
#   No grid search is re-run — the same model spec trained locally is
#   reproduced on production infrastructure. If no local run exists,
#   falls back to full grid search on K8s.
#
#   --local-train always runs the full grid search locally.
#
# Prerequisites:
#   - outerbounds CLI installed and authenticated
#   - metaflow configured for Outerbounds (metadata service + S3 datastore)
#   - conda env with scikit-learn, metaflow, streamlit
#   - At least one local FraudTrainingFlow run (for promote mode)
# ============================================================================

APP_NAME_BASE="credit-fraud"
USERNAME="${USER:-$(whoami)}"
# Strip domain if it's an email-style username, keep short form
USERNAME=$(echo "${USERNAME}" | sed 's/@.*//' | tr '.' '-' | tr '[:upper:]' '[:lower:]')
APP_NAME="${APP_NAME_BASE}-${USERNAME}"
PORT=8501
PYTHON_VERSION="3.12"
CONFIG="config.yml"
CONFIG_BAK="config.yml.bak"
FLOWS_DIR="flows"
RUN_ID_DIR=".run_ids"

# Outerbounds perimeter key — required for the deployed app to access
# Metaflow metadata service and load trained model artifacts.
# Set via env var or edit this default.
OBP_PERIMETER_KEY="${OBP_PERIMETER_KEY:-}"

# Parse flags
SKIP_TRAINING=false
SKIP_DEPLOY=false
TRAIN_LOCAL=false

for arg in "$@"; do
    case $arg in
        --deploy-only)  SKIP_TRAINING=true ;;
        --train-only)   SKIP_DEPLOY=true ;;
        --local-train)  TRAIN_LOCAL=true ;;
        *)              echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo " Fraud Dashboard — Outerbounds Deploy"
echo "============================================"
echo ""
echo " Training:  $(${SKIP_TRAINING} && echo 'SKIP' || (${TRAIN_LOCAL} && echo 'LOCAL' || echo 'KUBERNETES'))"
echo " Deploy:    $(${SKIP_DEPLOY} && echo 'SKIP' || echo 'YES')"
echo " App name:  ${APP_NAME}"
echo ""

# ----------------------------------------------------------------------------
# 0. Verify prerequisites
# ----------------------------------------------------------------------------

if [[ ! -f "app.py" ]]; then
    echo "ERROR: app.py not found. Run from the project root."
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: ${CONFIG} not found."
    exit 1
fi

if [[ ! -d "${FLOWS_DIR}" ]]; then
    echo "ERROR: ${FLOWS_DIR}/ directory not found."
    exit 1
fi

mkdir -p "${RUN_ID_DIR}"

echo "Project directory: $(pwd)"
echo ""

# ----------------------------------------------------------------------------
# 1. Train model (unless --deploy-only)
# ----------------------------------------------------------------------------

if [[ "${SKIP_TRAINING}" == "false" ]]; then

    if [[ "${TRAIN_LOCAL}" == "true" ]]; then
        DATA_FLOW="${FLOWS_DIR}/data_prep_flow_local.py"
        TRAIN_FLOW="${FLOWS_DIR}/training_flow_local.py"
        K8S_FLAG=""
        echo "============================================"
        echo " Phase 1: Training (LOCAL)"
        echo "============================================"
    else
        DATA_FLOW="${FLOWS_DIR}/data_prep_flow.py"
        TRAIN_FLOW="${FLOWS_DIR}/training_flow.py"
        K8S_FLAG="--environment=fast-bakery --with kubernetes"
        echo "============================================"
        echo " Phase 1: Training (KUBERNETES — promote)"
        echo "============================================"
    fi
    echo ""

    # --- Step 1a: Data Prep ---
    echo "--- Step 1/2: Data Preparation ---"
    echo "Running: python ${DATA_FLOW} ${K8S_FLAG} run"
    echo ""

    # --run-id-file captures the run ID for chaining
    python "${DATA_FLOW}" ${K8S_FLAG} run \
        --run-id-file "${RUN_ID_DIR}/data_prep_run_id"

    DATA_RUN_ID=$(cat "${RUN_ID_DIR}/data_prep_run_id")
    echo ""
    echo "✅ Data prep complete — run ID: ${DATA_RUN_ID}"
    echo ""

    # --- Step 1b: Training ---
    # Extract best hyperparameters from the latest LOCAL training run.
    # The deploy promotes this exact config to K8s — no grid search,
    # same model specification, production infrastructure.
    # Extract best hyperparameters from the latest LOCAL training run.
    # The deploy promotes this exact config to K8s — no grid search,
    # same model specification, production infrastructure.
    PROMOTE_HPARAMS=""
    if [[ "${TRAIN_LOCAL}" == "false" ]]; then
        echo "--- Extracting best hyperparameters from latest local training run ---"
        PROMOTE_HPARAMS=$(python3 << 'PYEOF'
import json, sys
try:
    from metaflow import Flow
    run = Flow("FraudTrainingFlow").latest_successful_run
    hp = run["end"].task.data.best_hparams
    m = run["end"].task.data.best_metrics
    print(json.dumps(hp))
    print(f"   Promoting from local run {run.id}: {hp}", file=sys.stderr)
    print(f"   Local metrics: F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}", file=sys.stderr)
except Exception as e:
    print(f"   No local training run found: {e}", file=sys.stderr)
PYEOF
        ) || true

        if [[ -n "${PROMOTE_HPARAMS}" ]]; then
            echo "--- Step 2/2: Model Training (promoting best config) ---"
            echo "Promoting: ${PROMOTE_HPARAMS}"
            echo ""
        else
            echo "--- Step 2/2: Model Training (full grid search — no local run found) ---"
            echo ""
        fi
    else
        echo "--- Step 2/2: Model Training (full grid search — local mode) ---"
        echo ""
    fi

    TRAIN_CMD="python ${TRAIN_FLOW} ${K8S_FLAG} run --data_run_id ${DATA_RUN_ID}"
    if [[ -n "${PROMOTE_HPARAMS}" ]]; then
        TRAIN_CMD="${TRAIN_CMD} --promote_hparams '${PROMOTE_HPARAMS}'"
    fi
    echo "Running: ${TRAIN_CMD}"
    echo ""

    if [[ -n "${PROMOTE_HPARAMS}" ]]; then
        python "${TRAIN_FLOW}" ${K8S_FLAG} run \
            --data_run_id "${DATA_RUN_ID}" \
            --promote_hparams "${PROMOTE_HPARAMS}" \
            --run-id-file "${RUN_ID_DIR}/training_run_id"
    else
        python "${TRAIN_FLOW}" ${K8S_FLAG} run \
            --data_run_id "${DATA_RUN_ID}" \
            --run-id-file "${RUN_ID_DIR}/training_run_id"
    fi

    TRAINING_RUN_ID=$(cat "${RUN_ID_DIR}/training_run_id")
    echo ""
    echo "✅ Training complete — run ID: ${TRAINING_RUN_ID}"

    # Print best metrics
    python3 -c "
from metaflow import Flow
run = Flow('FraudTrainingFlow')['${TRAINING_RUN_ID}']
m = run['end'].task.data.best_metrics
hp = run['end'].task.data.best_hparams
print(f'   Model: RandomForest n={hp[\"n_estimators\"]}, depth={hp[\"max_depth\"]}')
print(f'   F1={m[\"f1\"]:.4f}  AUC={m[\"auc_roc\"]:.4f}  Precision={m[\"precision\"]:.4f}  Recall={m[\"recall\"]:.4f}')
" 2>/dev/null || true

    echo ""
    echo "============================================"
    echo " Training complete"
    echo "============================================"
    echo ""
fi

# ----------------------------------------------------------------------------
# 2. Deploy dashboard (unless --train-only)
# ----------------------------------------------------------------------------

if [[ "${SKIP_DEPLOY}" == "true" ]]; then
    echo "Skipping deploy (--train-only)."
    echo ""
    echo "To deploy with this model:"
    echo "  ./deploy.sh --deploy-only"
    exit 0
fi

echo "============================================"
echo " Phase 2: Dashboard Deploy"
echo "============================================"
echo ""

# --- Config swap ---
restore_config() {
    if [[ -f "${CONFIG_BAK}" ]]; then
        echo ""
        echo "Restoring ${CONFIG} from ${CONFIG_BAK}..."
        mv "${CONFIG_BAK}" "${CONFIG}"
        echo "Config restored."
    fi
}
trap restore_config EXIT

echo "Backing up ${CONFIG} → ${CONFIG_BAK}"
cp "${CONFIG}" "${CONFIG_BAK}"

# Read locked cloud endpoint from config
CLOUD_URL=$(python3 -c "
import yaml
with open('${CONFIG_BAK}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('cloud_url', ''))
")

CLOUD_API_KEY=$(python3 -c "
import yaml
with open('${CONFIG_BAK}') as f:
    c = yaml.safe_load(f) or {}
print(c.get('cloud_api_key', ''))
")

if [[ -z "${CLOUD_URL}" ]]; then
    echo "ERROR: cloud_url not set in ${CONFIG}. Lock an endpoint in dev mode first."
    exit 1
fi

# Write prod config — includes pipeline section for auto-load
python3 -c "
import yaml
prod = {
    'mode': 'prod',
    'cloud_url': '''${CLOUD_URL}''',
    'cloud_api_key': '''${CLOUD_API_KEY}''',
    'pipeline': {
        'mode': 'kubernetes',
        'auto_load_model': True,
        'flows_dir': 'flows',
        'scoring': {
            'block': 0.8,
            'review': 0.5,
            'llm_trigger': 0.3,
        },
    },
}
with open('${CONFIG}', 'w') as f:
    yaml.dump(prod, f, default_flow_style=False, sort_keys=False)
"

echo "Wrote prod ${CONFIG}:"
cat "${CONFIG}"
echo ""

# --- Requirements ---
cat > requirements.txt <<'EOF'
streamlit>=1.37.0
pandas
pyyaml
numpy
plotly
requests
scikit-learn
metaflow
openai
EOF

echo "Created requirements.txt"
echo ""

# --- Deploy ---
echo "Deploying ${APP_NAME} to Outerbounds..."
echo ""

# Build entrypoint — configure outerbounds on the pod so the app can
# reach the Metaflow metadata service and load trained model artifacts
ENTRYPOINT="cd /root/code-package"
ENTRYPOINT="${ENTRYPOINT} && for f in *py; do mv \"\$f\" \"\${f%py}.py\" 2>/dev/null; done"
ENTRYPOINT="${ENTRYPOINT} && for f in *yml; do mv \"\$f\" \"\${f%yml}.yml\" 2>/dev/null; done"
if [[ -n "${OBP_PERIMETER_KEY}" ]]; then
    ENTRYPOINT="${ENTRYPOINT} && outerbounds configure ${OBP_PERIMETER_KEY}"
    echo "Will configure outerbounds on pod with perimeter key"
else
    echo "WARNING: OBP_PERIMETER_KEY not set — deployed app may not find Metaflow artifacts"
    echo "         Set it: export OBP_PERIMETER_KEY=<your-key> && ./deploy.sh"
fi
ENTRYPOINT="${ENTRYPOINT} && streamlit run app.py --server.port ${PORT} --server.headless true"

outerbounds app deploy \
    --name "${APP_NAME}" \
    --app-type web \
    --port "${PORT}" \
    --package-src-path . \
    --package-suffixes py,yml \
    --cpu 2 \
    --memory 4096 \
    --min-replicas 1 \
    --max-replicas 2 \
    --python "${PYTHON_VERSION}" \
    --dep-from-requirements requirements.txt \
    --public-access \
    --description "Credit Fraud Risk Dashboard - Anaconda AI Catalyst" \
    -- bash -c "${ENTRYPOINT}"

# restore_config runs automatically via the EXIT trap

echo ""
echo "============================================"
echo " Deploy complete"
echo "============================================"
echo ""
echo " Training run IDs:"
[[ -f "${RUN_ID_DIR}/data_prep_run_id" ]] && echo "   Data Prep: $(cat ${RUN_ID_DIR}/data_prep_run_id)"
[[ -f "${RUN_ID_DIR}/training_run_id" ]] && echo "   Training:  $(cat ${RUN_ID_DIR}/training_run_id)"
echo ""
echo " Dashboard auto-loads the latest trained model on startup."
echo " Pipeline tab is hidden in prod mode."