#!/usr/bin/env bash

echo "JOB $SLURM_JOB_ID running on NODE $SLURMD_NODENAME"

huggingface-cli download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
PORT=$(( 8888 + $PORT_OFFSET ))

pip install --upgrade --force-reinstall flashinfer-python==0.3.0post1

export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 128 --max-running-requests 128 \
--mem-fraction-static 0.82 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
--enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL --disable-radix-cache \
--attention-backend trtllm_mla --stream-interval 30 --enable-flashinfer-trtllm-moe --quantization fp8 \
> $SERVER_LOG 2>&1 &

set +x
IGNORE_PAT="Ignore import error when loading sglang.srt.models.glm4v_moe: No module named 'transformers.models.glm4v_moe'"

while IFS= read -r line; do
  printf '%s\n' "$line"

  # Skip the known benign "Ignore import error ..." line
  if [[ "$line" == *"$IGNORE_PAT"* ]]; then
    continue
  fi

  # Keep your original "error" trap for everything else
  if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
    sleep 5
    tail -n100 "$SERVER_LOG"
    echo "JOB ${SLURM_JOB_ID:-NA} ran on NODE ${SLURMD_NODENAME:-unknown}"
    exit 1
  fi

  # Break when server is ready
  if [[ "$line" == *"The server is fired up and ready to roll"* ]]; then
    break
  fi
# Start tail from the beginning so we don't miss early lines
done < <(tail -n +1 -F "$SERVER_LOG")

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
