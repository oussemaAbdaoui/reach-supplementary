#!/usr/bin/env bash
# Re-runs the whole queue until every (model, dataset) grid is fully redone.
#
# Safe to run repeatedly: rerun_dialogue.py holds an exclusive lock per grid and
# flags each finished row with v5_rerun=1, so a retry resumes rather than
# repeats. Together answers a 402 "credit limit exceeded" for up to five minutes
# after a top-up while balances propagate, so a pass that dies early is retried
# rather than treated as fatal.
set -u
cd "$(dirname "$0")/.."
LOGS=research/results/logs
PASSES="${RERUN_PASSES:-6}"

for pass in $(seq 1 "$PASSES"); do
  echo "########## PASS $pass  $(date +%H:%M:%S) ##########"
  rm -f "$LOGS"/.lock_* 2>/dev/null
  RERUN_CAP="${RERUN_CAP:-3}" bash research/run_rerun_all.sh
  remaining=$(python - <<'PY'
import glob, pandas as pd
todo = 0
for f in glob.glob('research/results/*_nn_comparison_*.csv'):
    if any(s in f for s in ('_failed', '_superseded', '.bak', 'nollm')):
        continue
    tag = f.rsplit('_nn_comparison_', 1)[-1][:-4]
    if tag not in ('gridllama33','glm52','grok420','gpt56','gptoss120b'):
        continue
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if 'v5_rerun' in d.columns:
        todo += int((d['v5_rerun'] != 1).sum())
    else:
        todo += len(d)
print(todo)
PY
)
  echo "PASS $pass done -- rows still to redo: $remaining"
  [ "$remaining" = "0" ] && { echo "QUEUE COMPLETE $(date +%H:%M:%S)"; exit 0; }
  echo "sleeping 180s before next pass (balance propagation / rate limits)"
  sleep 180
done
echo "exhausted $PASSES passes; $remaining rows still to redo"
