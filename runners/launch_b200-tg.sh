#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/dev/shm/hf_hub_cache/"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
PORT=8888

server_name="bmk-server"

set -x
docker run --rm -d --network host --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT -e EP_SIZE \
-e TORCH_CUDA_ARCH_LIST="10.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
-e RANDOM_RANGE_RATIO -e RESULT_FILENAME -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}_docker.sh"
