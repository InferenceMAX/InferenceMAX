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
# HF_HUB_CACHE_MOUNT
# GITHUB_WORKSPACE

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

while [ -n "$(docker ps -aq)" ]; do
  sleep 1
done

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"
port=8888

docker network create $network_name

set -x
docker run --rm -d --ipc host --shm-size=16g --network $network_name --name $server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-e HF_TOKEN=$HF_TOKEN -e HF_HUB_CACHE=$HF_HUB_CACHE \
-e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 -e VLLM_ROCM_USE_AITER_MHA=0 \
-e VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1 -e VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1 \
-e TRITON_HIP_PRESHUFFLE_SCALES=1 -e VLLM_USE_AITER_TRITON_GEMM=1 \
--entrypoint=vllm \
$IMAGE \
serve $MODEL --port=$port \
--tensor-parallel-size=$TP \
--no-enable-prefix-caching \
--compilation-config='{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64], "full_cuda_graph": true}' \
--block-size=64
--disable-log-requests

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network $network_name --name $client_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ -e HF_TOKEN=$HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=python3 \
$IMAGE \
bench_serving/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json

docker stop $server_name
docker network rm $network_name
