#!/usr/bin/bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

pip install flashinfer-python==0.3.0

export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

FUSION_FLAG='{'\
'"pass_config": {"enable_fi_allreduce_fusion": true, "enable_attn_fusion": true, "enable_noop": true},'\
'"custom_ops": ["+quant_fp8", "+rms_norm"],'\
'"cudagraph_mode": "FULL_DECODE_ONLY",'\
'"splitting_ops": []'\
'}'

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-batched-tokens 8192 --max-num-seqs 512 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config "$FUSION_FLAG" \
--disable-log-requests
