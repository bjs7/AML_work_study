# %%
import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

from lib.analysis_functions import assert_paths_exist
from lib.scenarios import build_scenario_map
from result_io.load_results import load_experiment

EVAL_MODE   = 'comparable'
WRITING_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol')
OUT_DIR     = WRITING_DIR / 'figs' / 'pr_curves'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ============================================================
# SCENARIOS
# ============================================================

scenario_ids = ["S1", "S2", "S3", "S4", "F1", "P2", "V1"]
scenario_map = build_scenario_map(scenario_ids, eval_mode=EVAL_MODE)
assert_paths_exist(scenario_map)

# Display labels and visual style per scenario
SCENARIO_STYLE = {
    "S1": dict(label="Full-info (R0, BN)",      color="#1f77b4", ls="--",  lw=1.5),
    "S2": dict(label="Full-info (R1, LN)",       color="#1f77b4", ls="-",   lw=2.0),
    "S3": dict(label="Individual (batching)",    color="#ff7f0e", ls="--",  lw=1.5),
    "S4": dict(label="Individual (full batch)",  color="#ff7f0e", ls="-",   lw=1.5),
    "F1": dict(label="FedAvg",                   color="#2ca02c", ls="-",   lw=2.0),
    "P2": dict(label="FedProx (μ=0.1)",          color="#9467bd", ls="-",   lw=2.0),
    "V1": dict(label="SplitFed",                 color="#d62728", ls="-",   lw=2.0),
}


# %%
# ============================================================
# PR CURVE EXTRACTION
# ============================================================

def _get_scores(lv):
    """
    Pick the right probability column from a laundering_values DataFrame.
    Full-info experiments store scores in pred_probabilities; federated/
    individual experiments store them in avg_prob (average across parties).
    """
    if lv['pred_probabilities'].max() > 0:
        return lv['pred_probabilities'].values
    return lv['avg_prob'].values


def collect_pr_curves(scenario_map):
    """
    For each scenario load all seeds, pool probability scores and true_y,
    then compute a single PR curve over the pooled predictions.

    Returns dict: scenario_id -> {
        'precision': array, 'recall': array, 'thresholds': array,
        'ap': float,   # average precision (area under curve)
        'n_seeds': int,
        'score_col': str,  # which column was used
    }
    """
    results = {}
    for sid, info in scenario_map.items():
        exp = load_experiment(Path(info['path']))
        all_probs, all_labels = [], []
        score_col = None
        for seed_data in exp.seed_results.values():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            scores = _get_scores(lv)
            if score_col is None:
                score_col = ('pred_probabilities'
                             if lv['pred_probabilities'].max() > 0 else 'avg_prob')
            all_probs.append(scores)
            all_labels.append(lv['true_y'].values)

        if not all_probs:
            print(f"  WARNING: no predictions found for {sid}")
            continue

        y_score = np.concatenate(all_probs)
        y_true  = np.concatenate(all_labels)
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        results[sid] = {
            'precision':  prec,
            'recall':     rec,
            'thresholds': thr,
            'ap':         ap,
            'n_seeds':    len(all_probs),
            'score_col':  score_col,
        }
        print(f"  {sid}: AP={ap:.4f}  (pooled {len(all_probs)} seeds, "
              f"{int(y_true.sum())}/{len(y_true)} positives, score={score_col})")

    return results


print("Collecting PR curves...")
pr_data = collect_pr_curves(scenario_map)


# %%
# ============================================================
# PLOT
# ============================================================

def plot_pr_curves(pr_data, scenario_ids, title="Precision–Recall curves",
                   out_path=None, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    for sid in scenario_ids:
        if sid not in pr_data:
            continue
        d = pr_data[sid]
        st = SCENARIO_STYLE[sid]
        label = f"{st['label']}  (AP={d['ap']:.3f})"
        ax.plot(d['recall'], d['precision'],
                color=st['color'], ls=st['ls'], lw=st['lw'], label=label)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {out_path}")
    plt.show()
    return fig


# All scenarios together
plot_pr_curves(
    pr_data, scenario_ids,
    title=f"Precision–Recall curves — comparable ({EVAL_MODE})",
    out_path=OUT_DIR / f"pr_curves_all_{EVAL_MODE}.pdf",
)

# %%
# Federated methods only (F1, P2, V1) vs the two full-info upper bounds (S1, S2)
plot_pr_curves(
    pr_data, ["S1", "S2", "F1", "P2", "V1"],
    title=f"Precision–Recall — federated vs. full-info ({EVAL_MODE})",
    out_path=OUT_DIR / f"pr_curves_federated_vs_fullinfo_{EVAL_MODE}.pdf",
    figsize=(6, 4.5),
)

# %%
# Print AP summary table
print("\nAverage Precision summary:")
print(f"{'Scenario':<6}  {'Name':<30}  {'AP':>6}  {'Seeds':>5}")
print("-" * 55)
for sid in scenario_ids:
    if sid not in pr_data:
        continue
    d  = pr_data[sid]
    st = SCENARIO_STYLE[sid]
    print(f"{sid:<6}  {st['label']:<30}  {d['ap']:>6.4f}  {d['n_seeds']:>5}")

# %%
