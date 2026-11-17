#!/usr/bin/env bash
# Every (model, dataset) job, run in parallel with a concurrency cap.
#
# Cap is 5. Each process peaks near 148 MB while sklearn trains the MLP and
# settles near 85 MB; with ~1.2 GB free, five staggered starts keep the peaks
# from colliding. The real ceiling is not RAM anyway -- every model's extraction
# calls funnel through one Together account, so more processes just queue there.
#
# Anchors, SHAP, LIME and CP-SAT never run: the loop replays the stored ctx.
#
set -u
cd "$(dirname "$0")/.."
CAP="${RERUN_CAP:-5}"
LOGS=research/results/logs
mkdir -p "$LOGS"

# Ordered dataset-major, so concurrent jobs sit on DIFFERENT generation
# providers. Model-major ordering put five Llama jobs in flight at once, and
# every one of them hit Together for generation AND for extraction -- one key,
# one rate limit, so they serialised behind each other (measured: 1.19 vs 8.50
# rows/min on sibling jobs). Interleaved, at most one concurrent job bills
# Together for generation; the rest use xAI / OpenAI / Anthropic / Groq and only
# their extraction calls share the Together limit.
jobs_list=()
for ds in cancer heart diabetes employee compas; do
  for spec in \
    "gridllama33 meta-llama/Llama-3.3-70B-Instruct-Turbo together" \
    "grok420 grok-4.20-0309-non-reasoning xai" \
    "gpt56 gpt-5.6-terra openai" \
    "sonnet5 claude-sonnet-5 anthropic" \
    "gptoss120b openai/gpt-oss-120b groq" \
    "glm52 zai-org/GLM-5.2 together" ; do
    jobs_list+=("$ds|$spec")
  done
done

echo "queued ${#jobs_list[@]} jobs, cap=$CAP, start $(date +%H:%M:%S)"
for j in "${jobs_list[@]}"; do
  ds="${j%%|*}"; rest="${j#*|}"
  set -- $rest
  tag="$1"; model="$2"; prov="$3"
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 5; done
  echo "-> $tag/$ds $(date +%H:%M:%S)"
  # The groq registry entry carries SLEEP_BETWEEN=5.0, applied twice per refine
  # round (once before the generation call, once before extraction) -- 10 s of
  # pure sleep per round, which capped gpt-oss-120b at ~1 row/min. That is a
  # fixed delay, not Groq's rate limit: 429s are already handled by key rotation
  # and backoff across the three GROQ keys. 1.0 s keeps a courtesy gap without
  # dominating the run. Every other provider is already 0.0.
  if [ "$prov" = "groq" ]; then SLEEP_OVERRIDE=1.0; else SLEEP_OVERRIDE=0.0; fi
  XAI_SLEEP="$SLEEP_OVERRIDE" \
  python research/rerun_dialogue.py "$ds" "$tag" "$model" "$prov" \
      > "$LOGS/rerun_${tag}_${ds}.log" 2>&1 &
  sleep 8   # stagger so MLP-training peaks do not align
done
wait
echo "ALL DONE $(date +%H:%M:%S)"
