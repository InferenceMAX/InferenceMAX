#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
# PORT_OFFSET

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

hf download $MODEL


pip install datasets pandas

nvidia-smi

pip install --upgrade --force-reinstall flashinfer-python==0.3.1.post1

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}  
fi

cat > config.yaml << EOF
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

export TORCH_CUDA_ARCH_LIST="10.0"
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

set -x
PYTHONNOUSERSITE=1 vllm serve $MODEL --host 0.0.0.0 --port $PORT --config config.yaml \
--gpu-memory-utilization 0.9 --tensor-parallel-size $TP --max-num-seqs 512 \
--disable-log-requests > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    # Ignore intel_extension_for_pytorch import errors
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]] && [[ ! "$line" =~ "intel_extension_for_pytorch" ]]; then
		sleep 5
		tail -n100 $SERVER_LOG
        echo "JOB $SLURM_JOB_ID ran on NODE $SLURMD_NODENAME"
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ \
--result-filename $RESULT_FILENAME.json
