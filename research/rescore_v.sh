#!/usr/bin/env bash
# Re-score U / V / E for generators whose v5 re-run is complete.
#
# The grid CSVs store score_v5 from precision_hat(), which samples
# Uniform({0,1}^d) -- the sampler the paper rejects (sec 5.2). Every figure the
# paper quotes comes from this re-scoring, so nothing in the corrected dialogue
# column is quotable until this runs.
#
# Local CPU only: rebuilds the tier and queries the MLP, no API calls, so it
# runs alongside the judge without competing for any rate limit.
#
#   bash research/rescore_v.sh <tag> [<tag> ...]
set -u
cd "$(dirname "$0")/.."
for tag in "$@"; do
  echo "=== rescore $tag  $(date +%H:%M:%S) ==="
  python - "$tag" <<'PY'
import sys, os, importlib.util
tag = sys.argv[1]
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location(
    'cnfs', os.path.join('research', 'check_nn_feasible_sampling.py'))
m = importlib.util.module_from_spec(spec)
sys.argv = ['check_nn_feasible_sampling.py']       # module reads argv at import
spec.loader.exec_module(m)                          # __name__ != '__main__', main() stays put
DATASETS = ['cancer', 'heart', 'diabetes', 'employee', 'compas']
out = f'nn_feasible_{tag}.csv'
print(f'  run_sweep(DATASETS, "_{tag}", "{out}")', flush=True)
m.run_sweep(DATASETS, f'_{tag}', out)
PY
  echo "=== $tag done  $(date +%H:%M:%S) ==="
done
echo "ALL RESCORES DONE $(date +%H:%M:%S)"
