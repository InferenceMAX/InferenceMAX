#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
# RESULT_FILENAME

export HF_MODULES_CACHE="/tmp/hf_modules_cache/"
export SGLANG_USE_AITER=1

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs 128 > $SERVER_LOG 2>&1 &
set +x


## Ensure benching scripts present
git config --global --add safe.directory /workspace || true
if [[ ! -d bench_serving ]]; then
    git clone https://github.com/kimbochen/bench_serving.git
fi

#
## Deps for lm-eval
#python3 -m pip install -q --upgrade pip || true
python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true
# Temporary: workaround known harness issue
python3 -m pip install -q --no-cache-dir --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main" || true

#
## Wait for vllm server to start up
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == *"The server is fired up and ready to roll"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url "http://0.0.0.0:$PORT" \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json
set +x

#######

#
## Evals setup 
# !TODO clean env vars
EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
OPENAI_COMP_BASE="$OPENAI_SERVER_BASE/v1/completions"
OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

# Patch to convert bypass regex error if content field is empty
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
set -x
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
