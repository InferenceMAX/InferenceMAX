#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/dev/shm/hf_hub_cache/"
PORT=8888

network_name="bmk-net"
server_name="bmk-server"
client_name="bmk-client"

docker network create $network_name

set -x
docker run --rm -d --network $network_name --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e TORCH_CUDA_ARCH_LIST="10.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
${IMAGE/\#/\/} \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_b200_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

git clone https://github.com/kimbochen/bench_serving.git

set -x
docker run --rm --network $network_name --name $client_name \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
${IMAGE/\#/\/} \
-lc "pip install -q datasets pandas && \
python3 bench_serving/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://$server_name:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json"

while [ -n "$(docker ps -aq)" ]; do
    docker stop $server_name
    docker network rm $network_name
    sleep 5
done
