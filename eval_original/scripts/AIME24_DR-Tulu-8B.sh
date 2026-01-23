#!/bin/bash
set -euo pipefail
set -x
#SBATCH --gpus=1
#SBATCH -p gpu

# e.g. sbatch -p gpu --gpus=4 ./AIME24_DR-Tulu-8B.sh

# --- Local-first dataset snapshot setup ---
export SCIEVAL_DATA_ROOT=/data/home/scyb546/datasets

AIME24_REPO_ID="HuggingFaceH4/aime_2024"
AIME24_LOCAL_DIR="${SCIEVAL_DATA_ROOT}/$(echo "${AIME24_REPO_ID}" | sed 's#/#__#g')"

# Download only if local snapshot dir does not exist or is empty.
if [ ! -d "${AIME24_LOCAL_DIR}" ] || [ -z "$(ls -A "${AIME24_LOCAL_DIR}" 2>/dev/null)" ]; then
  uv run python ../scieval/offline_download.py \
    --repo_id "${AIME24_REPO_ID}" \
    --root "${SCIEVAL_DATA_ROOT}"
else
  echo "[AIME24] Found local dataset snapshot: ${AIME24_LOCAL_DIR}"
fi

# --- Offline mode env vars (force local read) ---
export PYTHONPATH=/data/home/scyb546/multi-lora/eval:${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

rm -rf ../outputs/local/AIME24/DR-Tulu-8B
uv run python ../run.py \
  --data AIME24 \
  --model DR-Tulu-8B \
  --mode all \
  --use-vllm \
  --work-dir ../outputs/local \
  --verbose