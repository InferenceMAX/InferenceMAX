#!/usr/bin/env bash

GHA_CACHE_DIR="/raid/"
PORT_OFFSET=${USER: -1}

JOB_SCRIPT=$(mktemp $GITHUB_WORKSPACE/slurm-XXXXXX.sh)
cat > $JOB_SCRIPT <<-EOF
#!/usr/bin/env bash

huggingface-cli download $MODEL
pip3 install --user sentencepiece

set -x
port=$(( 8888 + $PORT_OFFSET ))
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port \$port --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--disable-radix-cache --decode-log-interval 1 --cuda-graph-bs 4 8 16 32 64 128 256 --cuda-graph-max-bs 256 --max-running-requests 512 \
> /workspace/server_\${SLURM_JOB_ID}.log 2>&1 &

set +x
while ! grep -q "The server is fired up and ready to roll!" /workspace/server_\${SLURM_JOB_ID}.log; do
    if grep -iq "error" /workspace/server_\${SLURM_JOB_ID}.log; then
        grep -iC5 "error" /workspace/server_\${SLURM_JOB_ID}.log
        exit 1
    fi
    tail -n10 /workspace/server_\${SLURM_JOB_ID}.log
    sleep 5
done
tail -n10 /workspace/server_\${SLURM_JOB_ID}.log

set -x
git clone https://github.com/kimbochen/bench_serving.git 
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:\$port \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ \
--result-filename $RESULT_FILENAME.json
EOF

set -x
srun --partition=dgx-h200 --gres=gpu:$TP --exclusive \
--container-image=$IMAGE \
--container-name=dsv3-$USER \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$GHA_CACHE_DIR/hf_hub_cache/:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash < $JOB_SCRIPT

rm -f $JOB_SCRIPT
