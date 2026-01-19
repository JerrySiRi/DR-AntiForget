#!/bin/bash
set -euo pipefail
set -x
#SBATCH --gpus=1
#SBATCH -p gpu

# e.g. sbatch -p gpu --gpus=4 ./AIME25_Qwen3-8B.sh

# --- Local-first dataset snapshot setup ---
export SCIEVAL_DATA_ROOT=/data/home/scyb546/datasets

AIME25_REPO_ID="math-ai/aime25"
AIME25_LOCAL_DIR="${SCIEVAL_DATA_ROOT}/$(echo "${AIME25_REPO_ID}" | sed 's#/#__#g')"

# Download only if local snapshot dir does not exist or is empty.
if [ ! -d "${AIME25_LOCAL_DIR}" ] || [ -z "$(ls -A "${AIME25_LOCAL_DIR}" 2>/dev/null)" ]; then
  uv run python ../scieval/offline_download.py \
    --repo_id "${AIME25_REPO_ID}" \
    --root "${SCIEVAL_DATA_ROOT}"
else
  echo "[AIME25] Found local dataset snapshot: ${AIME25_LOCAL_DIR}"
fi

# --- Offline mode env vars (force local read) ---
export PYTHONPATH=/data/home/scyb546/multi-lora/eval:${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run python ../run.py \
  --data AIME25 \
  --model Qwen3-8B \
  --mode all \
  --use-vllm \
  --work-dir ../outputs/local \
  --verbose