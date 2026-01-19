#!/bin/bash
set -x
#SBATCH --gpus=1 
#SBATCH -p gpu   

# e.g. sbatch -p gpu --gpus=4 ./ChemBench_Qwen3-8B.sh 

#! Read Offline datasets
export PYTHONPATH=/data/home/scyb546/multi-lora/eval:$PYTHONPATH
export SCIEVAL_DATA_ROOT=/data/home/scyb546/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run python ../run.py \
  --data ChemBench \
  --model Qwen3-8B \
  --mode all \
  --use-vllm \
  --work-dir ../outputs/local \
  --verbose