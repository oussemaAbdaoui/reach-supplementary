#!/usr/bin/env bash
# Re-run the COP arms on V-fitted rules, merge each back into its model's grid,
# then extend the k-sweep past the certified lengths so cop_best can pick a k*
# that is not capped below the range every other arm uses.
#
# Waits for the machine to go idle first: CP-SAT and the LLM runs contend, and
# anchors alone needs ~700 MB.
cd "$(dirname "$0")/.." || exit 1

while tasklist 2>/dev/null | grep -qi python; do sleep 30; done
echo "=== machine idle; COP arm re-runs start $(date +%H:%M:%S) ==="

export XAI_MAX_TOKENS_CAP=32768
export XAI_ANCHOR_BATCH=10
export XAI_N_INSTANCES=15
export XAI_SAVE_TRACES=1

for TAG in cogito671b llama33_newprompts; do
  if [ "$TAG" = "cogito671b" ]; then
    MODEL="deepcogito/cogito-v2-1-671b"
  else
    MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"
  fi
  for ARM in cop cop_k; do
    echo "--- $TAG / $ARM ---"
    XAI_ARMS="$ARM" XAI_OUT_TAG="$TAG" python run_comparison_nn.py cancer "$MODEL" 2>&1
    python research/merge_cop_arm.py cancer "$TAG" "$ARM" 2>&1
  done
done

echo "=== COP arms merged $(date +%H:%M:%S); extended k-sweep to k=18 starts ==="
XAI_KMAX=18 XAI_CSV_TAG=_V18 python run_cop_sweep.py cancer 2>&1
echo "=== pipeline complete $(date +%H:%M:%S) ==="
