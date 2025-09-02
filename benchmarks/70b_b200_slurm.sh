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

set -x
hf download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

#nccl update
pip uninstall -y nvidia-nccl-cu12
pip install nvidia-nccl-cu12==2.26.2.post1

pip uninstall -y flashinfer-python
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
git checkout 9720182476ede910698f8d783c29b2ec91cec023
cd flashinfer
pip install .

export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

FUSION_FLAG='{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'


export TORCH_CUDA_ARCH_LIST="10.0"
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-num-seqs 512 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config ${FUSION_FLAG} \
--disable-log-requests > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
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