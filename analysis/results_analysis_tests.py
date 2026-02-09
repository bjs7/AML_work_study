# %%

import sys
import os
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')

from results.load_results import find_experiments, load_experiment
import json
import pickle
import pandas as pd
import numpy as np


# %% ========== Set paths ==========

# ==============================================================================================
# ==================================== SET EXPERIMENT PATHS ====================================
# ==============================================================================================

def get_experiment_path(fl_algo, model, data_size='small', ilicit_level='HI',
                        eval_mode='system', testing=False,
                        # GNN flags
                        emlps=False, ports=False, tds=False, reverse_mp=False,
                        # Data flags (order must match save_results.py)
                        batching=False, batchnorm=False, ibm_fe=False, ibm_hp=False,
                        use_global_stats=False, normalize_currency=False,
                        bank_filter=None, loss_ratio=None, batch_size=8192,
                        # FedAvg/FedProx params
                        weighting='proportional', client_fraction=1.0,
                        num_local_epochs=1, num_rounds=100, mu=0.0):

    base_path = '/home/nam_07/projects/AML_work_study/experiments'
    if testing:
        base_path += '/testing'
    base_path += f"/{data_size}_{ilicit_level}/split_0.6_0.2/{eval_mode}/{fl_algo}"

    # Algo subfolder for FedAvg/FedProx
    if fl_algo in ('FedAvg', 'FedProx'):
        algo_subfolder = f"{weighting}_C{client_fraction}_E{num_local_epochs}"
        if num_rounds != 100:
            algo_subfolder += f"_R{num_rounds}"
        if mu > 0:
            algo_subfolder += f"_mu{mu}"
        base_path += f"/{algo_subfolder}"

    # Model folder with GNN flags
    model_folder = model
    if emlps: model_folder += '__emlps'
    if ports: model_folder += '__ports'
    if tds: model_folder += '__tds'
    if reverse_mp: model_folder += '__reverse_mp'
    base_path += f"/{model_folder}"

    # Data flags folder (order matches save_results.py)
    data_args = []
    if batching: data_args.append('batching')
    if batchnorm: data_args.append('batchnorm')
    if ibm_fe: data_args.append('ibm_fe')
    if ibm_hp: data_args.append('ibm_hp')
    if use_global_stats: data_args.append('use_global_stats')
    if normalize_currency: data_args.append('normalize_currency')
    if bank_filter: data_args.append(f'bank_filter_{bank_filter}')
    if loss_ratio is not None: data_args.append(f'loss_ratio_{loss_ratio}')
    if batch_size != 8192: data_args.append(f'batch_size_{batch_size}')

    data_folder = '__'.join(data_args) if data_args else 'default'

    return os.path.join(base_path, data_folder)


# 10 test experiments (matching multiple_jobs_test.sh, excluding LI)
experiments_paths = (
    # Baseline scenarios
    #get_experiment_path(fl_algo='full_info', model='GINe', batching=True, ibm_hp=True, emlps=True, testing=True),
    get_experiment_path(fl_algo='individual', model='GINe', batching=True, ibm_hp=True, emlps=True, batchnorm=True),
    get_experiment_path(fl_algo='FedAvg', model='GINe', batching=True, ibm_hp=True, emlps=True, num_local_epochs=5, client_fraction=0.1),
)

# %% ========== Place Results in dictionary ==========

# ============================================================================================
# ==================================== Packing of Results ====================================
# ============================================================================================

results = {}

for idx, path in enumerate(experiments_paths):
    #len('/home/nam_07/projects/AML_work_study/experiments/')
    results[path[49:]] = load_experiment(path)


# %% ========== Get f1 scores ==========

# ==============================================================================================
# ==================================== Obtain F1 scores ====================================
# ==============================================================================================

def get_f1_scores(results):

    for path, result in results.items():
        f1_mean = result.aggregated_stats['f1']['mean']
        diff_min = f1_mean - min(result.aggregated_stats['f1']['values'])
        diff_max = max(result.aggregated_stats['f1']['values']) - f1_mean
        plus_min_value = max(diff_min, diff_max) * 100
        f1_mean *= 100
        
        print(f"=== f1 score {path}")
        print(f"mean: {round(f1_mean, 2)}, plus_min: {round(plus_min_value, 2)}")


get_f1_scores(results)

def print_metric_summary(result, metric="f1", scale=100):
    stats = result.aggregated_stats[metric]
    mean = stats["mean"] * scale
    std = stats["std"] * scale
    mn  = min(stats["values"]) * scale
    mx  = max(stats["values"]) * scale
    print(f"{metric}: {mean:.2f} ± {std:.2f} (min={mn:.2f}, max={mx:.2f})")

for exp_path, result in results.items():
    print_metric_summary(result, metric='precision')


for exp_path, result in results.items():
    print_metric_summary(result, metric='recall')


# %%


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers: confusion counts from laundering_values ---
def confusion_from_laundering_df(df: pd.DataFrame):
    y = df["true_y"].to_numpy()
    yhat = df["pred_label"].to_numpy()

    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    tn = int(np.sum((y == 0) & (yhat == 0)))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}

def pattern_recall_from_laundering_df(df: pd.DataFrame):
    if "Pattern" not in df.columns:
        return None
    pos = df[df["true_y"] == 1].copy()
    if pos.empty:
        return pd.DataFrame(columns=["Pattern", "P", "TP", "recall"])
    grouped = pos.groupby("Pattern")
    out = grouped.apply(lambda g: pd.Series({
        "P": int(len(g)),
        "TP": int(np.sum(g["pred_label"].to_numpy() == 1)),
    })).reset_index()
    out["recall"] = out["TP"] / out["P"]
    return out

# --- formatting: mean ± std for slides ---
def mean_std(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))

def fmt_mean_std(m, s, scale=100.0, decimals=2):
    return f"{m*scale:.{decimals}f} ± {s*scale:.{decimals}f}"

def fmt_int_mean_std(m, s, decimals=1):
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

# --- core extraction from a loaded ExperimentResults object ---
def summarize_experiment(exp, use_aggregated_stats=True):
    """
    exp: ExperimentResults from load_experiment()
    Returns per-seed summary + aggregated (mean/std) summary.
    """
    seed_rows = []
    for seed, seed_data in sorted(exp.seed_results.items()):
        lv = seed_data.get("laundering_values", None)
        metrics = seed_data.get("metrics", None)

        if lv is None:
            continue

        conf = confusion_from_laundering_df(lv)
        row = {
            "seed": seed,
            "tp": conf["tp"],
            "fp": conf["fp"],
            "fn": conf["fn"],
            "tn": conf["tn"],
            "precision_from_labels": conf["precision"],
            "recall_from_labels": conf["recall"],
            "f1_from_labels": conf["f1"],
        }

        # If your pipeline already stores final metrics (recommended), keep them too:
        if metrics is not None:
            for k in ["f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy"]:
                if k in metrics:
                    row[k] = metrics[k]
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, {}

    # Prefer your stored metrics if present; otherwise use label-based
    def pick(metric_name, fallback):
        if metric_name in seed_df.columns and seed_df[metric_name].notna().all():
            return seed_df[metric_name].tolist()
        return seed_df[fallback].tolist()

    f1_vals = pick("f1", "f1_from_labels")
    p_vals  = pick("precision", "precision_from_labels")
    r_vals  = pick("recall", "recall_from_labels")

    summary = {}
    summary["f1_mean"], summary["f1_std"] = mean_std(f1_vals)
    summary["precision_mean"], summary["precision_std"] = mean_std(p_vals)
    summary["recall_mean"], summary["recall_std"] = mean_std(r_vals)

    for c in ["tp", "fp", "fn"]:
        summary[f"{c}_mean"], summary[f"{c}_std"] = mean_std(seed_df[c].tolist())

    return seed_df, summary

# --- write LaTeX tables (simple, robust) ---
def df_to_latex_table(df: pd.DataFrame, out_path: Path, caption=None, label=None):
    # Keep it minimal for slides
    latex = df.to_latex(index=False, escape=False, column_format="l" * df.shape[1])
    if caption or label:
        wrapped = []
        wrapped.append(r"\begin{table}")
        wrapped.append(r"\centering")
        wrapped.append(latex)
        if caption:
            wrapped.append(rf"\caption{{{caption}}}")
        if label:
            wrapped.append(rf"\label{{{label}}}")
        wrapped.append(r"\end{table}")
        latex = "\n".join(wrapped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)

# --- main: build scenario summary table for slides ---
def build_scenario_summary_table(scenario_map, out_dir="tables"):
    """
    scenario_map: dict of { "S1": {"name": "...", "path": "..."} , ... }
    """
    rows = []
    per_seed = {}

    for sid, info in scenario_map.items():
        
        exp = load_experiment(info["path"])  # uses your loader
        seed_df, summ = summarize_experiment(exp)
        
        #exp = load_experiment(info["path"])  # uses your loader
        #exp = load_experiment(scenario_map['S1']['path'])        
        #seed_df, summ = summarize_experiment(results[exp_path])
        
        per_seed[sid] = seed_df

        rows.append({
            "ID": sid,
            "Scenario": info["name"],
            "F1": fmt_mean_std(summ.get("f1_mean", 0), summ.get("f1_std", 0)),
            "Prec": fmt_mean_std(summ.get("precision_mean", 0), summ.get("precision_std", 0)),
            "Rec": fmt_mean_std(summ.get("recall_mean", 0), summ.get("recall_std", 0)),
            "TP": fmt_int_mean_std(summ.get("tp_mean", 0), summ.get("tp_std", 0)),
            "FP": fmt_int_mean_std(summ.get("fp_mean", 0), summ.get("fp_std", 0)),
            "FN": fmt_int_mean_std(summ.get("fn_mean", 0), summ.get("fn_std", 0)),
        })

    summary_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    summary_df.to_csv(out_dir / "scenario_summary.csv", index=False)
    df_to_latex_table(summary_df, out_dir / "scenario_summary.tex")
    return summary_df, per_seed

# --- build pattern recall summary and a top-K plot ---
def build_pattern_recall(scenario_map, out_fig="figs/pattern_recall_topk.pdf", top_k=6):
    all_rows = []
    for sid, info in scenario_map.items():
        exp = load_experiment(info["path"])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get("laundering_values", None)
            if lv is None or "Pattern" not in lv.columns:
                continue
            pr = pattern_recall_from_laundering_df(lv)
            if pr is None or pr.empty:
                continue
            pr["ID"] = sid
            pr["Scenario"] = info["name"]
            pr["seed"] = seed
            all_rows.append(pr)

    if not all_rows:
        return None, None

    pr_df = pd.concat(all_rows, ignore_index=True)

    # Aggregate over seeds: mean/std recall per (scenario, pattern) and support
    agg = pr_df.groupby(["ID", "Scenario", "Pattern"]).agg(
        P=("P", "mean"),                 # support (same across seeds usually)
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
    ).reset_index()

    Path("tables").mkdir(exist_ok=True)
    agg.to_csv("tables/pattern_recall_summary.csv", index=False)

    # Choose top-K patterns by overall support across scenarios
    top_patterns = (agg.groupby("Pattern")["P"].sum().sort_values(ascending=False).head(top_k).index.tolist())

    # Simple plot: for each scenario, plot recall_mean for top patterns
    scenarios = agg["Scenario"].unique().tolist()
    x = np.arange(len(top_patterns))

    plt.figure(figsize=(11, 4))
    for scen in scenarios:
        sub = agg[(agg["Scenario"] == scen) & (agg["Pattern"].isin(top_patterns))].set_index("Pattern")
        y = [sub.loc[p, "recall_mean"] if p in sub.index else 0.0 for p in top_patterns]
        plt.plot(x, y, marker="o", label=scen)

    plt.xticks(x, [str(p) for p in top_patterns])
    plt.ylim(0, 1.0)
    plt.xlabel("Pattern (top-K by support)")
    plt.ylabel("Recall on illicit edges (TP / P)")
    plt.legend()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    return pr_df, agg



experiments_paths = (
    get_experiment_path(fl_algo='individual', model='GINe', batching=True, ibm_hp=True, emlps=True, batchnorm=True),
    get_experiment_path(fl_algo='FedAvg', model='GINe', batching=True, ibm_hp=True, emlps=True, num_local_epochs=5, client_fraction=0.1),
)



scenario_map = {
    "S1": {"name": "Individual (R0, BN)", "path": experiments_paths[0]},
    "S2": {"name": "FedAvg (R1, LN)", "path": experiments_paths[1]},
}


scenario_map = {
    "S1": {"name": "Full-info (R0, BN)", "path": "/abs/path/to/S1"},
    "S2": {"name": "Full-info (R1, LN)", "path": "/abs/path/to/S2"},
    "S3": {"name": "Individual (R1, batch)", "path": "/abs/path/to/S3"},
    "S4": {"name": "Individual (R1, full)", "path": "/abs/path/to/S4"},
    "S5": {"name": "FedAvg (R1)", "path": "/abs/path/to/S5"},
    "S6": {"name": "FedProx (R1, mu=0.1)", "path": "/abs/path/to/S6"},
}




scenario_df, per_seed = build_scenario_summary_table(scenario_map, out_dir="tables")
build_pattern_recall(scenario_map, out_fig="figs/pattern_recall_topk.pdf", top_k=6)







# %%


from results.load_results import _resolve_run_folder

# here set the run ID for the tests, if that has to be done!

exp_path_with_run_ids = {}
for exp in experiments_paths:
    exp_path_with_run_ids[exp] = None


for exp_path, rund_id in exp_path_with_run_ids.items():
    exp_path_with_run_ids[exp_path] = _resolve_run_folder(Path(exp_path), rund_id)



import pickle, numpy as np
from pathlib import Path

def extract_confusion_from_run(exp_path):
    exp_path = Path(exp_path)
    seed_dirs = sorted([d for d in exp_path.glob("seed_*") if d.is_dir()],
                       key=lambda p: int(p.name.split("_")[1]))

    tp_list, fp_list, fn_list = [], [], []
    for sd in seed_dirs:
        with open(sd / "metrics_laundering_values.pkl", "rb") as f:
            data = pickle.load(f)

        lv = data["laundering_values"]  # dataframe
        true_y = np.asarray(lv["true_y"])
        pred_y = np.asarray(lv["pred_label"])

        tp = np.sum((true_y == 1) & (pred_y == 1))
        fp = np.sum((true_y == 0) & (pred_y == 1))
        fn = np.sum((true_y == 1) & (pred_y == 0))

        tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)

    def summarize(x):
        return float(np.mean(x)), float(np.std(x)), int(np.min(x)), int(np.max(x))

    return {
        "tp": summarize(tp_list),
        "fp": summarize(fp_list),
        "fn": summarize(fn_list),
        "n_seeds": len(seed_dirs),
    }


for path, path_with_id in exp_path_with_run_ids.items():
    print(extract_confusion_from_run(path_with_id))




exp_path = result
exp_path = experiments_paths[1]




exp_path = str(path_with_id)











# %%

import pickle

for exp_path in experiments_paths:

    laun_values = []
    false_pos = []

    for seed in [1,2,3,4]:
        path = exp_path + f'/seed_{seed}/'

        with open(path + 'metrics_laundering_values.pkl', 'rb') as f:
            data = pickle.load(f)

        launderings_found = sum((data['laundering_values']['true_y'] == 1) & (data['laundering_values']['pred_label'] == 1))
        false_positives = sum((data['laundering_values']['true_y'] == 0) & (data['laundering_values']['pred_label'] == 1))

        laun_values.append(launderings_found)
        false_pos.append(false_positives)

    laun_values_mean = np.mean(laun_values)
    laun_diff_min = laun_values_mean - min(laun_values)
    laun_diff_max = max(laun_values) - laun_values_mean
    laun_plus_min_value = max(laun_diff_max, laun_diff_min)

    false_pos_mean = np.mean(false_pos)
    false_pos_diff_min = false_pos_mean - min(false_pos)
    false_pos_diff_max = max(false_pos) - false_pos_mean
    false_pos_plus_min_value = max(false_pos_diff_max, false_pos_diff_min)

    print(f"=== Found illicit cases {path[49:]}")
    print(f"mean: {laun_values_mean}, plus_min: {laun_plus_min_value}")

    print(f"=== False positive cases {path[49:]}")
    print(f"mean: {false_pos_mean}, plus_min: {false_pos_plus_min_value}")





# %%

experiments_paths


for path, result in results.items():
    test123 = result.aggregated_stats

list(test123)

# Extract all metrics
metrics_to_extract = ['precision', 'recall', 'roc_auc', 'pr_auc']
metrics_results = {}

for path, result in results.items():
    metrics_results[path] = {}
    
    for metric in metrics_to_extract:
        mean = result.aggregated_stats[metric]['mean']
        diff_min = mean - min(result.aggregated_stats[metric]['values'])
        diff_max = max(result.aggregated_stats[metric]['values']) - mean
        plus_min_value = max(diff_min, diff_max) * 100
        mean *= 100
        
        metrics_results[path][metric] = {
            'mean': mean,
            'plus_min': plus_min_value
        }







# %%

top_30_by_edges = [68, 4, 0, 2, 1682, 2279, 9898, 11300, 542, 13342, 1703, 147, 4881, 50, 18, 13341, 1643, 61, 1633, 1533, 12208, 11794, 11, 2278, 5611, 3394, 14091, 22, 108, 25]
top_30_banks_f1 = {}

for seed in [1,2,3,4]:
    path = experiments_paths[2] + f'/seed_{seed}/'
    with open(path + 'metrics_laundering_values.pkl', 'rb') as f:
        data = pickle.load(f)

    top_30 = dict(sorted(data['party_performance'].items(), key=lambda x: x[1]['f1'], reverse=True)[:30])
    for bank_id in top_30_by_edges:
        if seed == 1:
            top_30_banks_f1[bank_id] = [data['party_performance'][bank_id]['f1']]
        else:
            top_30_banks_f1[bank_id].append(data['party_performance'][bank_id]['f1'])


top_30_banks_f1_means_std = {}
for bank_id, values in top_30_banks_f1.items():
    top_30_banks_f1_means_std[bank_id] = {'mean': np.mean(values), 'std': np.std(values)}



import matplotlib.pyplot as plt
import numpy as np

keys = list(top_30_banks_f1_means_std.keys())
means = [top_30_banks_f1_means_std[k]['mean'] for k in keys]
stds = [top_30_banks_f1_means_std[k]['std'] for k in keys]

plt.figure(figsize=(12, 6))
plt.errorbar(range(len(keys)), means, yerr=stds, fmt='o', capsize=3, alpha=0.6)
plt.xlabel('Bank ID')
plt.ylabel('Mean ± range (min/max deviation from mean)')
plt.title('Mean values of individual f1 score')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%








# %%

print(f"Loaded experiment from: {results[0]}")
print(f"Available seeds: {list(results[0].seed_results.keys())}")
print()


print(json.dumps(results['small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats, indent=2))






print("=== Aggregated Results ===")
print(json.dumps(results[0].aggregated_stats, indent=2))
print()

# Summary table
metrics_df = pd.DataFrame({
    metric: {
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max']
    }
    for metric, stats in results['small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.items()
    if metric != 'best_seed' and isinstance(stats, dict)
}).T

print("\n=== Summary Table ===")
print(metrics_df)
print(f"\nBest seed: {results[0].aggregated_stats.get('best_seed', 'N/A')}")
print()








# %%



# Load the experiment
results = load_experiment(experiment_path)

print(f"Loaded experiment from: {experiment_path}")
print(f"Available seeds: {list(results.seed_results.keys())}")
print()






# %%
# Find all individual GINe experiments
experiment_paths = find_experiments(
    '/home/nam_07/projects/AML_work_study/experiments',
    fl_algo='individual',
    model_name='GINe'
)

# Shows: ['/path/to/experiments/.../individual/GINe/default', 
#         '/path/to/experiments/.../individual/GINe/ibm_fe']
print(experiment_paths)

# %%




results = load_experiment(experiment_paths[0])

# Access the data
print(results.config)
print(results.aggregated_stats['f1']['mean'])
print(results.hyperparameters)




# %%

paths = find_experiments(
    '/home/nam_07/projects/AML_work_study/experiments',
    size='small',
    fl_algo='individual'
)

# 2. Load each and extract F1 scores
for path in paths:
    exp = load_experiment(path)
    print(f"Experiment: {exp.config['model']['model_name']}")
    print(f"Settings: {path.split('/')[-1]}")  # Shows 'default' or 'ibm_fe' etc.
    print(f"Mean F1: {exp.aggregated_stats['f1']['mean']:.3f}")
    print(f"Best seed: {exp.aggregated_stats['best_seed']}")
    print()

# %%
# ============================================================================
# LOAD YOUR SPECIFIC EXPERIMENT
# ============================================================================

experiment_path = '/home/nam_07/projects/AML_work_study/experiments/small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'

# Load the experiment
results = load_experiment(experiment_path)

print(f"Loaded experiment from: {experiment_path}")
print(f"Available seeds: {list(results.seed_results.keys())}")
print()

# %%
# ============================================================================
# VIEW EXPERIMENT CONFIGURATION
# ============================================================================

print("=== Experiment Configuration ===")
print(json.dumps(results.config, indent=2))
print()

# %%
# ============================================================================
# VIEW HYPERPARAMETERS
# ============================================================================

print("=== Hyperparameters ===")
print(json.dumps(results.hyperparameters, indent=2))
print()

# %%
# ============================================================================
# VIEW AGGREGATED RESULTS (SUMMARY ACROSS ALL SEEDS)
# ============================================================================

print("=== Aggregated Results ===")
print(json.dumps(results.aggregated_stats, indent=2))
print()

# Summary table
metrics_df = pd.DataFrame({
    metric: {
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max']
    }
    for metric, stats in results.aggregated_stats.items()
    if metric != 'best_seed' and isinstance(stats, dict)
}).T

print("\n=== Summary Table ===")
print(metrics_df)
print(f"\nBest seed: {results.aggregated_stats.get('best_seed', 'N/A')}")
print()

# %%
# ============================================================================
# ANALYZE INDIVIDUAL SEED RESULTS
# ============================================================================

print("=== Individual Seed Results ===")
for seed_num, seed_data in sorted(results.seed_results.items()):
    print(f"\n--- Seed {seed_num} ---")

    if 'metrics' in seed_data:
        print("Metrics:", seed_data['metrics'])

    if 'laundering_values' in seed_data:
        lv = seed_data['laundering_values']
        print(f"Laundering values shape/info: {type(lv)}")
        if isinstance(lv, (list, np.ndarray)):
            print(f"  Length: {len(lv)}")
            if len(lv) > 0:
                print(f"  Sample (first 5): {lv[:5]}")

    if 'model' in seed_data:
        print(f"Model loaded: {type(seed_data['model'])}")

print()

# %%
# ============================================================================
# DETAILED ANALYSIS OF BEST SEED
# ============================================================================

best_seed = results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.get('best_seed')
if best_seed and best_seed in results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].seed_results:
    print(f"=== Detailed Analysis of Best Seed ({best_seed}) ===")
    best_seed_data = results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].seed_results[best_seed]

    print(f"\nMetrics: {best_seed_data.get('metrics', {})}")

    if 'laundering_values' in best_seed_data:
        lv = best_seed_data['laundering_values']
        print(f"\nLaundering values statistics:")
        if isinstance(lv, (list, np.ndarray)):
            lv_array = np.array(lv)
            print(f"  Count: {len(lv_array)}")
            print(f"  Mean: {np.mean(lv_array):.4f}")
            print(f"  Std: {np.std(lv_array):.4f}")
            print(f"  Min: {np.min(lv_array):.4f}")
            print(f"  Max: {np.max(lv_array):.4f}")

print()

# %%
# ============================================================================
# COMPARE PERFORMANCE ACROSS SEEDS
# ============================================================================

print("=== Seed-by-Seed Comparison ===")

# Extract metrics for each seed from aggregated_stats
seed_comparison = {}
for metric, stats in results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.items():
    if metric != 'best_seed' and isinstance(stats, dict) and 'values' in stats:
        seed_comparison[metric] = stats['values']

# Create DataFrame
comparison_df = pd.DataFrame(seed_comparison)
comparison_df.index = [f"seed_{i+1}" for i in range(len(comparison_df))]

print(comparison_df)
print()

# %%
