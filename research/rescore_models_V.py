"""Re-score the per-model cancer grids (now carrying the V-fitted COP arms)
under U / V / E, using the same estimator as check_nn_feasible_sampling.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from check_nn_feasible_sampling import run_sweep

for suffix, out in [('_cogito671b', 'nn_feasible_sampling_cogito671b.csv'),
                    ('_llama33_newprompts', 'nn_feasible_sampling_llama33_newprompts.csv')]:
    print(f'\n================ {suffix} ================', flush=True)
    run_sweep(['cancer'], suffix, out)
