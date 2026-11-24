"""Compare per-transaction test predictions: original run (test pass right after training)
vs the evaluation-only run (evaluation/both). Expected: same transactions, same labels,
probabilities differing slightly where test-time neighbour sampling drew differently.

Run on HPC (or wherever both result folders are):
    python scripts/hpc/tests/compare_evaluation_with_original.py [RUN_DIR]
"""

import pickle
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
               '/data/leuven/362/vsc36278/AML_work_study/experiments/small_HI/split_0.6_0.2/system/'
               'FedGraphSimple/GINe__emlps/batching__ibm_hp/20260712_195549')


def load(path):
    with open(path, 'rb') as f:
        lv = pickle.load(f)['laundering_values']
    return lv.set_index('indices')


for seed_dir in sorted(RUN_DIR.glob('seed_*')):
    original_pkl = seed_dir / 'metrics_laundering_values.pkl'
    eval_pkl = RUN_DIR / 'evaluation' / 'both' / seed_dir.name / 'metrics_laundering_values.pkl'
    if not (original_pkl.exists() and eval_pkl.exists()):
        continue
    orig, ev = load(original_pkl), load(eval_pkl)
    common = orig.index.intersection(ev.index)
    o, e = orig.loc[common], ev.loc[common]
    diff = np.abs(o['pred_probabilities'].values - e['pred_probabilities'].values)
    changed = diff > 1e-6
    y = o['true_y'].values == 1

    print(f"{seed_dir.name}: {len(common)} common transactions "
          f"(original {len(orig)}, evaluation {len(ev)})")
    print(f"  same labels: {(o['pred_label'].values == e['pred_label'].values).all()}   "
          f"label flips: {int((o['pred_label'].values != e['pred_label'].values).sum())}")
    print(f"  probability diff: max {diff.max():.2e}   mean {diff.mean():.2e}   "
          f"changed (>1e-6): {int(changed.sum())} ({changed.mean():.2%}), "
          f"of which illicit: {int((changed & y).sum())}")
    print(f"  correlation of probabilities: {np.corrcoef(o['pred_probabilities'], e['pred_probabilities'])[0, 1]:.6f}")
