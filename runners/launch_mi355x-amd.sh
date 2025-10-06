#!/usr/bin/env bash

# === Workflow-defined Env Vars ===
# IMAGE
# MODEL
# TP
# HF_HUB_CACHE
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# CONC
# GITHUB_WORKSPACE
# RESULT_FILENAME
# HF_TOKEN

HF_HUB_CACHE_MOUNT="/nfsdata/hf_hub_cache-1/"  # Temp solution
PORT=8888

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

docker network create $network_name

set -x
docker run --rm -d --ipc=host --shm-size=16g --network=$network_name --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e ISL -e OSL \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_mi355x_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

if [[ "$MODEL" == "amd/DeepSeek-R1-0528-MXFP4-Preview" || "$MODEL" == "deepseek-ai/DeepSeek-R1-0528" ]]; then
  NUM_PROMPTS=$(( CONC * 50 ))
else
  NUM_PROMPTS=$(( CONC * 10 ))
fi

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network=$network_name --name=$client_name \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=python3 \
$IMAGE \
bench_serving/benchmark_serving.py \
--model=$MODEL --backend=vllm --base-url="http://$server_name:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$NUM_PROMPTS \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=/workspace/ --result-filename=$RESULT_FILENAME.json

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi


while [ -n "$(docker ps -aq)" ]; do
    docker stop $server_name
    docker network rm $network_name
    sleep 5
done
