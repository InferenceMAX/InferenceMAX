#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-llama-3.3-70b-fp8.html#run-the-inference-benchmark

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ "$CONC" -gt "16" ]]; then
        export VLLM_ROCM_USE_AITER_MHA=1
    fi
fi

set -x
vllm serve $MODEL --port=$PORT \
--swap-space=64 \
--gpu-memory-utilization=0.94 \
--dtype=auto --kv-cache-dtype=fp8 \
--distributed-executor-backend=mp --tensor-parallel-size=$TP \
--max-model-len=$MAX_MODEL_LEN \
--max-seq-len-to-capture=$MAX_MODEL_LEN \
--max-num-seqs=$CONC \
--max-num-batched-tokens=131072 \
--no-enable-prefix-caching \
--async-scheduling \
--disable-log-requests
