#!/usr/bin/env bash

# === Required Env Vars ===
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
set -euo pipefail

# Create a basic vLLM config
cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
export TORCH_CUDA_ARCH_LIST="9.0"
PORT=${PORT:-8888}

# Start server in the background, shld be openai/gpt-oss-120b
set -x
PYTHONNOUSERSITE=1 vllm serve "$MODEL" --host=0.0.0.0 --port="$PORT" \
  --config config.yaml \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size="$TP" \
  --max-num-seqs="$CONC" \
  --disable-log-requests > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

cleanup() {
  set +e
  if ps -p "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for server readiness via log message
echo "Waiting for vLLM server to become ready on port $PORT ..."
READY=0
for i in $(seq 1 600); do
  if grep -q "Application startup complete" "$SERVER_LOG"; then
    READY=1
    break
  fi
  if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited prematurely. Last 200 lines of log:" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$READY" -ne 1 ]]; then
  echo "Timed out waiting for vLLM server readiness." >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi
echo "vLLM server up"
RUN_MODE=${RUN_MODE:-benchmark}

if [[ "$RUN_MODE" == "eval" ]]; then
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
  OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
  OPENAI_COMP_BASE="$OPENAI_SERVER_BASE/v1/completions"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  # Ensure bench_serving is present (mirror benchmark behavior)
  git config --global --add safe.directory /workspace || true
  if [[ ! -d bench_serving ]]; then
    git clone https://github.com/oseltamivir/bench_serving.git
  fi

  # Deps for lm-eval
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true
  # Temporary: workaround known harness issue
  python3 -m pip install -q --no-cache-dir --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main" || true

  echo "Using model: $MODEL"

  # Clean up previous eval results if any
  rm -rf "/workspace/${EVAL_RESULT_DIR}"/* 2>/dev/null || true
  mkdir -p "/workspace/${EVAL_RESULT_DIR}"

  set -x
  # BEFORE calling `lm_eval` in your script:
  PATCH_DIR="$(mktemp -d)"
cat > "$PATCH_DIR/sitecustomize.py" <<'PY'
import re, sys, unicodedata
from lm_eval.filters import extraction as ex

def _s(x):  # coerce to str
    return x if isinstance(x, str) else ""

# --- Patch RegexFilter.apply (used by many datasets) ---
_orig_regex_apply = ex.RegexFilter.apply
def _safe_regex_apply(self, resps, docs):
    out = []
    for inst in resps:  # inst is a list of candidate responses for one doc
        filtered = []
        for resp in inst:
            txt = _s(resp)
            m = self.regex.findall(txt)
            if m:
                m = m[self.group_select]
                if isinstance(m, tuple):
                    m = [t for t in m if t]
                    m = m[0] if m else self.fallback
                m = m.strip()
            else:
                m = self.fallback
            filtered.append(m)
        out.append(filtered)
    return out
ex.RegexFilter.apply = _safe_regex_apply

# --- Patch MultiChoiceRegexFilter.apply (used by GSM8K flexible-extract) ---
_orig_mc_apply = ex.MultiChoiceRegexFilter.apply
def _safe_mc_apply(self, resps, docs):
    def find_match(regex, resp, convert_dict={}):
        txt = _s(resp)
        match = regex.findall(txt)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m]
                if match:
                    match = match[0]
        if match:
            match = match.strip()
            if match in convert_dict:
                return convert_dict[match]
            return match
        return None

    punct_tbl = dict.fromkeys(
        i for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)).startswith("P")
    )

    def filter_ignores(st):
        st = _s(st)
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)
        if self.ignore_case:
            st = st.lower()
        if self.ignore_punctuation:
            st = st.translate(punct_tbl)
        return st

    out = []
    for r, doc in zip(resps, docs):
        # Build fallback regexes from choices (A, B, C, ...) as in upstream
        fallback_regexes, choice_to_alpha = [], {}
        next_alpha = "A"
        without_paren, without_paren_to_target = [], {}
        for c in doc.get("choices", []):
            m = filter_ignores(c.strip())
            fallback_regexes.append(re.escape(m))
            choice_to_alpha[m] = f"({next_alpha})"
            without_paren.append(next_alpha)
            without_paren_to_target[next_alpha] = f"({next_alpha})"
            next_alpha = chr(ord(next_alpha) + 1)

        fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else None
        without_paren_regex = re.compile(rf":[\s]*({'|'.join(without_paren)})") if without_paren else None

        filtered = []
        for resp in r:
            m = find_match(self.regex, resp)
            if not m and fallback_regex:
                m = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
            if not m and without_paren_regex:
                m = find_match(without_paren_regex, resp, without_paren_to_target)
            if not m:
                m = self.fallback
            filtered.append(m)
        out.append(filtered)
    return out

ex.MultiChoiceRegexFilter.apply = _safe_mc_apply
PY

  export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"
  python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks ${EVAL_TASK:-gsm8k} \
    --num_fewshot ${NUM_FEWSHOT:-5} \
    --batch_size 2 \
    --output_path "/workspace/${EVAL_RESULT_DIR}" \
    --model_args "model=$MODEL,base_url=$OPENAI_CHAT_BASE,api_key=$OPENAI_API_KEY,eos_string=</s>,max_retries=3,num_concurrent=32,tokenized_requests=False" \
    --gen_kwargs "max_tokens=8192,temperature=0,top_p=1"
  set +x

  # Append a Markdown table to the GitHub Actions job summary using helper in bench_serving
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    python3 bench_serving/lm_eval_to_md.py \
      --results-dir "/workspace/${EVAL_RESULT_DIR}" \
      --task "${EVAL_TASK:-gsm8k}" \
      --framework "${FRAMEWORK}" \
      --precision "${PRECISION}" \
      --tp "${TP:-1}" \
      --ep "${EP_SIZE:-1}" \
      --dp-attention "${DP_ATTENTION:-false}" \
      >> "$GITHUB_STEP_SUMMARY" || true
  fi

  echo "Evaluation completed. Results in /workspace/${EVAL_RESULT_DIR}"
  exit 0

else

  # Default values for optional vars used by the benchmark
  ISL=${ISL:-256}
  OSL=${OSL:-256}
  RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
  RESULT_FILENAME=${RESULT_FILENAME:-results}

  # Install deps and run benchmark in the same container
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir datasets pandas || true
  if [[ ! -d bench_serving ]]; then
    git clone https://github.com/kimbochen/bench_serving.git
  fi

  set -x
  python3 bench_serving/benchmark_serving.py \
    --model="$MODEL" \
    --backend=vllm \
    --base-url="http://0.0.0.0:$PORT" \
    --dataset-name=random \
    --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio="$RANDOM_RANGE_RATIO" \
    --num-prompts=$(( CONC * 10 )) --max-concurrency="$CONC" \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/ \
    --result-filename="$RESULT_FILENAME.json"
  set +x
fi
