#!/bin/bash
set -euo pipefail
set -x
#SBATCH --gpus=4
#SBATCH -p gpu

# e.g. sbatch -p gpu --gpus=4 ./MMLU_Tongyi-DeepResearch-30B-A3B-11003.sh

# --- Local-first dataset snapshot setup ---
export DATA_ROOT=/data/home/scyb546/datasets
MMLU_REPO_ID="cais/mmlu"
MMLU_LOCAL_DIR="${DATA_ROOT}/$(echo "${MMLU_REPO_ID}" | sed 's#/#__#g')"


# Download only if local snapshot dir does not exist or is empty.
if [ ! -d "${MMLU_LOCAL_DIR}" ] || [ -z "$(ls -A "${MMLU_LOCAL_DIR}" 2>/dev/null)" ]; then
  uv run python ../../offline_download.py \
    --repo_id "${MMLU_REPO_ID}" \
    --root "${DATA_ROOT}"
else
  echo "[MMLU] Found local dataset snapshot: ${MMLU_LOCAL_DIR}"
fi

# --- Offline mode env vars (force local read) ---
export MODEL_PATH=/data/home/scyb546/models/Tongyi-DeepResearch-30B-A3B
export SERVED_MODEL_NAME=tongyi-deepresearch-30b-a3b
export PYTHONPATH=/data/home/scyb546/multi-lora/eval:${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1


# -- vLLM server -- #
#! &是把vllm server放在后台运行
VLLM_PORT=11003

uv run python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.97 \
    --trust-remote-code \
    --max-model-len 50000 \
    --port ${VLLM_PORT} \
    --max-num-seqs 200 \
    --enable-chunked-prefill \
    --enable-prefix-caching &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT


# Wait until vLLM is ready.
MAX_WAIT_SECS=600
for i in $(seq 1 ${MAX_WAIT_SECS}); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[ERROR] vLLM process ${VLLM_PID} exited before becoming healthy" >&2
    wait "${VLLM_PID}" || true
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[vLLM] Healthy on port ${VLLM_PORT} (after ${i}s)"
    break
  fi

  sleep 1
  if [ "${i}" -eq "${MAX_WAIT_SECS}" ]; then
    echo "[ERROR] vLLM did not become healthy on port ${VLLM_PORT} within ${MAX_WAIT_SECS}s" >&2
    exit 1
  fi
done


# -- evalscope evaluation scripts -- #
uv run evalscope eval \
 --model ${SERVED_MODEL_NAME} \
 --api-url http://127.0.0.1:${VLLM_PORT}/v1 \
 --eval-type openai_api \
 --repeats 1 \
 --datasets mmlu \
 --dataset-hub local \
 --dataset-dir ${MMLU_LOCAL_DIR} \
 --dataset-args '{"mmlu": {"aggregation": "mean", "few_shot_num": 5, "filters": {"remove_until": "</think>"}, "local_path": "'${MMLU_LOCAL_DIR}'"}}' \
 --generation-config '{"max_tokens": 38912, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}' \
 --timeout 60000 \
 --stream \
