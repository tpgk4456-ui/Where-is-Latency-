#!/usr/bin/env bash
set -euo pipefail

HOST="${LLAMA70B_HOST:-127.0.0.1}"
PORT="${LLAMA70B_PORT:-8002}"
SERVED_MODEL_NAME="${LLAMA70B_SERVED_MODEL_NAME:-llama3.3:70b-awq}"
MODEL_PATH="${LLAMA70B_MODEL_PATH:-/home/ysree/.cache/huggingface/hub/models--lambda--Llama-3.3-70B-Instruct-AWQ-4bit/snapshots/a70257cf10f368114a66115c315def76a1227e26}"
CUDA_DEVICES="${LLAMA70B_CUDA_VISIBLE_DEVICES:-0,1}"
GPU_MEMORY_UTILIZATION="${LLAMA70B_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${LLAMA70B_MAX_MODEL_LEN:-2048}"
ENV_NAME="${LLAMA70B_CONDA_ENV:-agentscope}"
CONDA_ROOT="${LLAMA70B_CONDA_ROOT:-/home/ysree/miniconda3}"
CUDA_RUNTIME_LIB="${CONDA_ROOT}/envs/${ENV_NAME}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${LD_LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR:-/tmp}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --quantization awq_marlin \
  --enforce-eager \
  --disable-custom-all-reduce
