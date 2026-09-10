#!/usr/bin/env bash
# Fires the judge for a generator the moment all five of its grids finish the
# v5 re-run, so judging overlaps the remaining generation instead of following it.
#
# Two things this has to get right:
#   * run_judge.py resumes on (instance, arm, strategy, pass), so the stale v5
#     rows are deleted first -- otherwise it skips exactly the rows we need
#     rescored. v4 and v6 judgements are left untouched and are NOT re-run.
#   * gpt-oss-120b is the judge, so gptoss120b is never judged (own family).
#
# Judge runs on Together: gpt-oss-120b GENERATION also uses Groq, and the two
# together saturated its 8k TPM (gptoss120b fell 7.6 -> 1.4 rows/min while three
# judge jobs sat in 210-270s backoffs). The appendix already documents the judge
# as served through Together and Groq, so this is consistent with the paper.
set -u
cd "$(dirname "$0")/.."
LOGS=research/results/logs
mkdir -p "$LOGS"
TAGS="gridllama33 glm52 grok420 gpt56 sonnet5"

while true; do
  pending=0
  for tag in $TAGS; do
    [ -f "$LOGS/.judged_$tag" ] && continue
    pending=1
    ready=$(python - "$tag" <<'PY'
import sys, glob, pandas as pd
tag = sys.argv[1]
files = [f'research/results/{d}_nn_comparison_{tag}.csv'
         for d in ('cancer','heart','diabetes','employee','compas')]
for f in files:
    try:
        d = pd.read_csv(f)
    except Exception:
        print('no'); sys.exit()
    if 'v5_rerun' not in d.columns or int((d['v5_rerun'] != 1).sum()):
        print('no'); sys.exit()
print('yes')
PY
)
    if [ "$ready" = "yes" ]; then
      echo "=== $tag grids complete -- stripping stale v5 judgements $(date +%H:%M:%S)"
      python - "$tag" <<'PY'
import sys, glob, pandas as pd
tag = sys.argv[1]
for f in glob.glob(f'research/results/*_nn_judge_{tag}.csv'):
    d = pd.read_csv(f)
    n = len(d)
    d = d[d['strategy'] != 'v5']
    d.to_csv(f, index=False)
    print(f'  {f.split("/")[-1]}: dropped {n-len(d)} v5 rows, kept {len(d)}')
PY
      echo "=== launching judge for $tag on groq $(date +%H:%M:%S)"
      XAI_OUT_TAG="$tag" XAI_PROVIDER=together XAI_JUDGE_PASSES=3 \
        nohup python run_judge.py > "$LOGS/judge_${tag}.log" 2>&1 &
      touch "$LOGS/.judged_$tag"
    fi
  done
  [ "$pending" = "0" ] && { echo "all generators judged/launched $(date +%H:%M:%S)"; exit 0; }
  sleep 60
done
