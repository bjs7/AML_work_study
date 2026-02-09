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

scenario_df, per_seed = build_scenario_summary_table(scenario_map, out_dir="tables")
build_pattern_recall(scenario_map, out_fig="figs/pattern_recall_topk.pdf", top_k=6)



# Main Scenarios

experiments_paths = (
    # --- Single experiments ---
    get_experiment_path(fl_algo='full_info', model='GINe', ibm_fe=True, batching=True, ibm_hp=True, emlps=True, batchnorm=True),
    get_experiment_path(fl_algo='full_info', model='GINe', ibm_hp=True, batching=True),
    get_experiment_path(fl_algo='full_info', model='GINe', ibm_hp=True, batching=True, emlps=True),
    get_experiment_path(fl_algo='individual', model='GINe', ibm_hp=True, batching=True, emlps=True),
    get_experiment_path(fl_algo='individual', model='GINe', ibm_hp=True, emlps=True),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, num_local_epochs=1, client_fraction=1.0, weighting='uniform'),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, num_local_epochs=5, client_fraction=0.1),
    # --- Packed Group 1: FedAvg batch/epoch sensitivity ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, num_local_epochs=1, client_fraction=1.0, weighting='uniform'),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, batch_size=4096),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, num_local_epochs=5, client_fraction=0.1),
    # --- Packed Group 2: Local epochs ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=1, client_fraction=0.1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=10, client_fraction=0.1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=25, client_fraction=0.1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, num_rounds=50),
    # --- Packed Group 3: Client fraction + weighting ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.25),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.50),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, weighting='uniform'),
    # --- Packed Group 4: FedProx ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, mu=0.01),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, mu=0.1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, mu=1.0),
    # --- Packed Group 5: Bank filters ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, bank_filter='no_top10'),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, bank_filter='no_top1'),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, bank_filter='no_bottom10'),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, bank_filter='no_bottom5pct'),
    # --- Packed Group 6: Loss ratio + currency ---
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, loss_ratio=1),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, loss_ratio=980),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, loss_ratio=50),
    get_experiment_path(fl_algo='FedAvg', model='GINe', ibm_hp=True, emlps=True, batching=True, num_local_epochs=5, client_fraction=0.1, normalize_currency=True),
)






scenario_map = {
    "S1": {"name": "Individual (R0, BN)", "path": experiments_paths[0]},
    "S2": {"name": "FedAvg (R1, LN)", "path": experiments_paths[1]},
}










