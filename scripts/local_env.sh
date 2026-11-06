#!/usr/bin/env bash
# Source from a WSL shell: source scripts/local_env.sh
LOCAL_AI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LOCAL_AI_ROOT
export HF_HOME="$LOCAL_AI_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$LOCAL_AI_ROOT/.cache"
export TRITON_CACHE_DIR="$LOCAL_AI_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_AI_ROOT/.cache/torchinductor"
export UNSLOTH_COMPILE_LOCATION="$LOCAL_AI_ROOT/.cache/unsloth_compiled_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export UNSLOTH_DISABLE_STATISTICS=1
export DO_NOT_TRACK=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export JUPYTER_CONFIG_DIR="$LOCAL_AI_ROOT/.cache/jupyter/config"
export JUPYTER_RUNTIME_DIR="$LOCAL_AI_ROOT/.cache/jupyter/runtime"
export IPYTHONDIR="$LOCAL_AI_ROOT/.cache/ipython"

