#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# deploy.sh — Train model + deploy dashboard to Outerbounds
#
# Usage:
#   ./deploy.sh                  # full pipeline: train on K8s → deploy
#   ./deploy.sh --deploy-only    # skip training, deploy with latest model
#   ./deploy.sh --train-only     # run pipeline on K8s, don't deploy app
#   ./deploy.sh --local-train    # train locally, then deploy app
#
# Prerequisites:
#   - outerbounds CLI installed and authenticated
#   - metaflow configured for Outerbounds (metadata service + S3 datastore)
#   - conda env with scikit-learn, metaflow, streamlit
# ============================================================================

APP_NAME="credit-fraud-melliott"
PORT=8501
PYTHON_VERSION="3.12"
CONFIG="config.yml"
CONFIG_BAK="config.yml.bak"
FLOWS_DIR="flows"
RUN_ID_DIR=".run_ids"
OBP_PERIMETER_KEY="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"

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
        echo " Phase 1: Training (KUBERNETES)"
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
    echo "--- Step 2/2: Model Training (4-config hyperparameter search) ---"
    echo "Running: python ${TRAIN_FLOW} ${K8S_FLAG} run --data_run_id ${DATA_RUN_ID}"
    echo ""

    python "${TRAIN_FLOW}" ${K8S_FLAG} run \
        --data_run_id "${DATA_RUN_ID}" \
        --run-id-file "${RUN_ID_DIR}/training_run_id"

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
    'streaming': {
        'total_transactions': 200,
        'batch_size': 20,
        'batch_delay': 0
    }
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
outerbounds
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