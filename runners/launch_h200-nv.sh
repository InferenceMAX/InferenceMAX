#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
export HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

PARTITION="dgx-h200"
SQUASH_FILE="/raid/image_${MODEL_CODE}_h200.sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --no-shell
JOB_ID=$(squeue -u $USER -h -o %A)

set -x
srun --jobid=$JOB_ID --pty bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$(realpath $SQUASH_FILE) \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_h200_slurm.sh

scancel $JOB_ID
