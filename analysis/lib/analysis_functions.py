import sys
import os
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')

from result_io.load_results import load_experiment
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

FIGS_DIR = Path(__file__).resolve().parent.parent / 'figs'


def get_experiment_path(fl_algo, model, data_size='small', ilicit_level='HI',
                        eval_mode='system', testing=False,
                        # GNN flags
                        emlps=False, ports=False, tds=False, reverse_mp=False,
                        # Data flags (order must match save_results.py)
                        batching=False, batchnorm=False, ibm_fe=False, ibm_hp=False,
                        use_global_stats=False, normalize_currency=False,
                        bank_filter=None, loss_ratio=None, batch_size=8192,
                        batching_mode='lazy_link_neighbor',
                        # FedAvg/FedProx params
                        weighting='proportional', client_fraction=1.0,
                        num_local_epochs=1, num_rounds=100, mu=0.0,
                        validate_every=1,
                        # FedGraph (SplitFed) param
                        aggregation='shared'):

    base_path = '/home/nam_07/projects/AML_work_study/experiments'
    if testing:
        base_path += '/testing'
    base_path += f"/{data_size}_{ilicit_level}/split_0.6_0.2/{eval_mode}/{fl_algo}"

    if fl_algo in ('FedAvg', 'FedProx'):
        algo_subfolder = f"{weighting}_C{client_fraction}_E{num_local_epochs}"
        if num_rounds != 100:
            algo_subfolder += f"_R{num_rounds}"
        if mu > 0:
            algo_subfolder += f"_mu{mu}"
        if validate_every != 1:
            algo_subfolder += f"_ve{validate_every}"
        base_path += f"/{algo_subfolder}"
    elif fl_algo == 'FedGraph':
        base_path += f"/{aggregation}"

    model_folder = model
    if emlps: model_folder += '__emlps'
    if ports: model_folder += '__ports'
    if tds: model_folder += '__tds'
    if reverse_mp: model_folder += '__reverse_mp'
    base_path += f"/{model_folder}"

    data_args = []
    if batching: data_args.append('batching')
    if batchnorm: data_args.append('batchnorm')
    if ibm_fe: data_args.append('ibm_fe')
    if ibm_hp: data_args.append('ibm_hp')
    if use_global_stats: data_args.append('use_global_stats')
    if batching and batching_mode != 'lazy_link_neighbor': data_args.append(f'bm_{batching_mode}')
    if normalize_currency: data_args.append('normalize_currency')
    if bank_filter: data_args.append(f'bank_filter_{bank_filter}')
    if loss_ratio is not None: data_args.append(f'loss_ratio_{loss_ratio}')
    if batch_size != 8192: data_args.append(f'batch_size_{batch_size}')

    data_folder = '__'.join(data_args) if data_args else 'default'
    return os.path.join(base_path, data_folder)


def assert_paths_exist(scenario_map):
    for k, v in scenario_map.items():
        p = Path(v["path"])
        if not p.exists():
            print(f"[MISSING] {k}: {p}")
        else:
            print(f"[OK] {k}: {p}")


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


def pattern_precision_from_laundering_df(df_full: pd.DataFrame, df_pos: pd.DataFrame = None):
    """Per-pattern precision = TP_P / (TP_P + FP_total).

    df_full : the complete laundering-values DataFrame for the test split
              (must include true_y==0 rows so FP can be counted).
    df_pos  : optional subset to restrict TP counting to (e.g. non-fragmented
              illicit transactions).  If None, all illicit rows of df_full are used.

    FP_total is always taken from df_full regardless of df_pos, so precision is
    not degenerate when df_pos contains only illicit transactions.

    FP carry no pattern label (they are legitimate transactions predicted as
    illicit), so FP_total is the same for every pattern — only TP_P varies
    across patterns.
    """
    if "Pattern" not in df_full.columns:
        return None
    FP_total = int(((df_full["pred_label"] == 1) & (df_full["true_y"] == 0)).sum())
    pos_df = df_pos if df_pos is not None else df_full
    pos = pos_df[pos_df["true_y"] == 1].copy()
    if pos.empty:
        return pd.DataFrame(columns=["Pattern", "FP", "TP", "precision"])
    grouped = pos.groupby("Pattern")
    out = grouped.apply(lambda g: pd.Series({
        "FP": FP_total,
        "TP": int(np.sum(g["pred_label"].to_numpy() == 1)),
    })).reset_index()
    out["precision"] = out["TP"] / (out["TP"] + out["FP"])
    return out


# --- formatting ---

def mean_std(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))

def fmt_mean_std(m, s, scale=100.0, decimals=2):
    return f"{m*scale:.{decimals}f} ± {s*scale:.{decimals}f}"

def fmt_int_mean_std(m, s, decimals=1):
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


# --- core extraction from a loaded ExperimentResults object ---

def summarize_experiment(exp, use_aggregated_stats=True):
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

        if metrics is not None:
            for k in ["f1", "precision", "recall", "roc_auc", "pr_auc", "accuracy"]:
                if k in metrics:
                    row[k] = metrics[k]
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, {}

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


def summarize_experiment_numeric(exp):
    seed_df, summ = summarize_experiment(exp)
    out = {
        "f1_mean": summ.get("f1_mean", 0.0),
        "f1_std": summ.get("f1_std", 0.0),
        "precision_mean": summ.get("precision_mean", 0.0),
        "precision_std": summ.get("precision_std", 0.0),
        "recall_mean": summ.get("recall_mean", 0.0),
        "recall_std": summ.get("recall_std", 0.0),
        "tp_mean": summ.get("tp_mean", 0.0),
        "tp_std": summ.get("tp_std", 0.0),
        "fp_mean": summ.get("fp_mean", 0.0),
        "fp_std": summ.get("fp_std", 0.0),
        "fn_mean": summ.get("fn_mean", 0.0),
        "fn_std": summ.get("fn_std", 0.0),
    }
    return out, seed_df


# --- write LaTeX tables ---

def _latex_escape(text: str) -> str:
    """Escape bare % and _ in a string for LaTeX (skip already-escaped sequences)."""
    import re
    text = re.sub(r'(?<!\\)%', r'\\%', text)
    text = re.sub(r'(?<!\\)_', r'\\_', text)
    return text


def _escape_df_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with LaTeX-escaped column names and string cell values."""
    df = df.copy()
    df.columns = [_latex_escape(str(c)) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: _latex_escape(str(v)) if isinstance(v, str) else v)
    return df


def _fmt_scaled(v, scale=1.0, decimals=2, auto_widen=False, max_decimals=6):
    """Format v*scale to `decimals` places; if auto_widen and that rounds a
    nonzero value to 0, keep adding decimals (up to max_decimals) for THIS
    value only, so small values (e.g. 0.0020) aren't lost while the rest of
    the column stays at `decimals`."""
    if pd.isna(v):
        return ""
    scaled = v * scale
    d = decimals
    if auto_widen:
        while d < max_decimals and scaled != 0 and round(scaled, d) == 0:
            d += 1
    return f"{scaled:.{d}f}"


def pct_col_formats(cols, decimals=2, auto_widen=True):
    """Convenience for the common case: rescale a set of 0-1 fraction columns
    to the house style (0-100 percentage scale, 2 decimals, auto-widen small
    values) for LaTeX output via df_to_latex_table's col_formats param."""
    return {c: {"scale": 100, "decimals": decimals, "auto_widen": auto_widen} for c in cols}


def df_to_latex_table(df: pd.DataFrame, out_path: Path, caption=None, label=None,
                      row_group_sizes=None, col_formats: dict = None, decimals: int = 2):
    """col_formats: optional {column_name: {"scale": 1.0, "decimals": 2, "auto_widen": False}}
    to rescale/format specific columns independently of the blanket float_format
    below (e.g. house style: percentage scale, 2 decimals — see pct_col_formats).
    Columns not listed keep the default `%.{decimals}f` formatting (decimals=2
    matches prior hardcoded behavior, so existing callers are unaffected)."""
    df = df.copy()
    for col, spec in (col_formats or {}).items():
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda v: _fmt_scaled(
            v, spec.get('scale', 1.0), spec.get('decimals', decimals), spec.get('auto_widen', False)))
    df = _escape_df_for_latex(df)
    latex = df.to_latex(index=False, escape=False, float_format=f"%.{decimals}f",
                        column_format="l" * df.shape[1])

    if row_group_sizes:
        lines = latex.split('\n')
        header_midrule = next(i for i, l in enumerate(lines) if l.strip() == r'\midrule')
        boundaries = set()
        cumsum = 0
        for size in row_group_sizes[:-1]:
            cumsum += size
            boundaries.add(cumsum - 1)
        new_lines = []
        data_row_idx = 0
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i > header_midrule and line.strip().endswith(r'\\') and not line.strip().startswith('\\'):
                if data_row_idx in boundaries:
                    new_lines.append(r'\midrule')
                data_row_idx += 1
        latex = '\n'.join(new_lines)

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


# --- table builders ---

def build_scenario_summary_table(scenario_map, out_dir="tables", order=None, out_name="scenario_summary",
                                  groups=None, csv_dir=None):
    rows = []
    per_seed = {}

    if order is None:
        ids = list(scenario_map.keys())
    else:
        ids = [sid for sid in order if sid in scenario_map]
        missing = [sid for sid in order if sid not in scenario_map]
        if missing:
            print(f"[WARN] These IDs were in 'order' but not in scenario_map: {missing}")

    for sid in ids:
        info = scenario_map[sid]
        exp = load_experiment(info["path"])
        seed_df, summ = summarize_experiment(exp)

        per_seed[sid] = seed_df
        rows.append({
            "ID": sid,
            "Scenario": info["name"],
            r"F1 (\%)": fmt_mean_std(summ.get("f1_mean", 0), summ.get("f1_std", 0)),
            r"Precision (\%)": fmt_mean_std(summ.get("precision_mean", 0), summ.get("precision_std", 0)),
            r"Recall (\%)": fmt_mean_std(summ.get("recall_mean", 0), summ.get("recall_std", 0)),
            "TP": fmt_int_mean_std(summ.get("tp_mean", 0), summ.get("tp_std", 0)),
            "FP": fmt_int_mean_std(summ.get("fp_mean", 0), summ.get("fp_std", 0)),
            "FN": fmt_int_mean_std(summ.get("fn_mean", 0), summ.get("fn_std", 0)),
        })

    summary_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(exist_ok=True, parents=True)
    summary_df.to_csv(csv_path / f"{out_name}.csv", index=False)

    row_group_sizes = None
    if groups is not None:
        actual = set(ids)
        row_group_sizes = [sum(1 for s in sids if s in actual) for sids in groups.values()]
        row_group_sizes = [s for s in row_group_sizes if s > 0]

    df_to_latex_table(summary_df, out_dir / f"{out_name}.tex", row_group_sizes=row_group_sizes)
    return summary_df, per_seed


def build_delta_table(scenario_map, baseline_id, out_dir="tables", out_name="scenario_deltas_vs_baseline",
                      order=None, include_std=False, groups=None, csv_dir=None):
    if baseline_id not in scenario_map:
        raise ValueError(f"baseline_id='{baseline_id}' not in scenario_map keys: {list(scenario_map.keys())}")

    if order is None:
        ids = list(scenario_map.keys())
    else:
        ids = [sid for sid in order if sid in scenario_map]

    base_exp = load_experiment(scenario_map[baseline_id]["path"])
    base, _ = summarize_experiment_numeric(base_exp)

    rows = []
    for sid in ids:
        exp = load_experiment(scenario_map[sid]["path"])
        cur, _ = summarize_experiment_numeric(exp)

        row = {
            "ID": sid,
            "Scenario": scenario_map[sid]["name"],
            "ΔF1 (pp)": (cur["f1_mean"] - base["f1_mean"]) * 100.0,
            "ΔPrec (pp)": (cur["precision_mean"] - base["precision_mean"]) * 100.0,
            "ΔRec (pp)": (cur["recall_mean"] - base["recall_mean"]) * 100.0,
            "ΔTP": (cur["tp_mean"] - base["tp_mean"]),
            "ΔFP": (cur["fp_mean"] - base["fp_mean"]),
            "ΔFN": (cur["fn_mean"] - base["fn_mean"]),
        }

        if include_std:
            row["ΔF1 std"] = np.sqrt(cur["f1_std"]**2 + base["f1_std"]**2) * 100.0
            row["ΔPrec std"] = np.sqrt(cur["precision_std"]**2 + base["precision_std"]**2) * 100.0
            row["ΔRec std"] = np.sqrt(cur["recall_std"]**2 + base["recall_std"]**2) * 100.0
            row["ΔTP std"] = np.sqrt(cur["tp_std"]**2 + base["tp_std"]**2)
            row["ΔFP std"] = np.sqrt(cur["fp_std"]**2 + base["fp_std"]**2)
            row["ΔFN std"] = np.sqrt(cur["fn_std"]**2 + base["fn_std"]**2)

        rows.append(row)

    df = pd.DataFrame(rows)

    round_cols = ["ΔF1 (pp)", "ΔPrec (pp)", "ΔRec (pp)", "ΔTP", "ΔFP", "ΔFN"]
    for c in round_cols:
        df[c] = df[c].astype(float).round(2)

    if include_std:
        for c in ["ΔF1 std", "ΔPrec std", "ΔRec std", "ΔTP std", "ΔFP std", "ΔFN std"]:
            df[c] = df[c].astype(float).round(2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f"{out_name}.csv", index=False)

    row_group_sizes = None
    if groups is not None:
        actual = set(ids)
        row_group_sizes = [sum(1 for s in sids if s in actual) for sids in groups.values()]
        row_group_sizes = [s for s in row_group_sizes if s > 0]

    df_to_latex_table(df, out_dir / f"{out_name}.tex", row_group_sizes=row_group_sizes)
    return df


# --- pattern reference data ---

PATTERN_NAMES = {
    0: 'NONE', 1: 'Fan-Out', 2: 'Fan-In', 3: 'Cycle',
    4: 'Gather-Scatter', 5: 'Scatter-Gather', 6: 'Stack',
    7: 'Random', 8: 'Bipartite', 9: 'Unknown',
}

PATTERN_DESCRIPTIONS = {
    0: 'Legitimate transaction (no laundering)',
    1: 'Single source fans out to many destinations (splitting/smurfing)',
    2: 'Many sources funnel into a single destination (aggregation)',
    3: 'Circular chain of transfers returning to the originator',
    4: 'Fan-in followed by fan-out: gather from many, then scatter',
    5: 'Fan-out followed by fan-in: scatter to many, then re-gather',
    6: 'Sequential linear chain of transfers (layering)',
    7: 'Structurally random transactions mimicking normal activity',
    8: 'Transfers between two disjoint groups of accounts',
    9: 'Labelled illicit but no matching pattern found in patterns file',
}


def build_pattern_overview_table(raw_df, out_dir='tables',
                                  out_name_desc='pattern_descriptions',
                                  out_name_stats='pattern_stats'):
    """Generate two complementary pattern reference tables.

    1. Descriptions table  (ID, Pattern, Description) — no counts.
    2. Stats table         (ID, Pattern, Count, %)    — no description.

    Args:
        raw_df : formatted transactions DataFrame with 'Pattern' and 'Is Laundering' columns.
    Returns:
        desc_df, stats_df
    """
    illicit_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_df = pd.DataFrame([
        {'ID': pid, 'Pattern': PATTERN_NAMES[pid], 'Description': PATTERN_DESCRIPTIONS[pid]}
        for pid in illicit_ids
    ])
    desc_df.to_csv(out_dir / f'{out_name_desc}.csv', index=False)
    df_to_latex_table(desc_df, out_dir / f'{out_name_desc}.tex')

    counts = raw_df[raw_df['Is Laundering'] == 1].groupby('Pattern').size()
    total_illicit = counts.sum()
    stats_df = pd.DataFrame([
        {'ID': pid, 'Pattern': PATTERN_NAMES[pid],
         'Count': int(counts.get(pid, 0)),
         '%': round(counts.get(pid, 0) / total_illicit * 100, 1)}
        for pid in illicit_ids
    ])
    stats_df.to_csv(out_dir / f'{out_name_stats}.csv', index=False)
    df_to_latex_table(stats_df, out_dir / f'{out_name_stats}.tex')

    return desc_df, stats_df


def build_pattern_stats_wide(raw_df, out_dir='tables',
                              out_name='pattern_stats_wide', csv_dir=None):
    """Transposed pattern statistics table: pattern names as columns, metrics as rows.

    Layout: 3 data rows (ID, Count, %) x 10 cols (label + 9 pattern names).
    Pattern names are column headers; ID, Count, % are the row metrics.

    Returns:
        df : wide DataFrame with columns ['Metric', *pattern_names].
    """
    illicit_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    counts = raw_df[raw_df['Is Laundering'] == 1].groupby('Pattern').size()
    total_illicit = counts.sum()

    cell = {pid: {
        'ID':    pid,
        'Count': int(counts.get(pid, 0)),
        '%':     round(counts.get(pid, 0) / total_illicit * 100, 1),
    } for pid in illicit_ids}

    pnames = [PATTERN_NAMES[pid] for pid in illicit_ids]

    rows = [
        {'Metric': 'ID',    **{PATTERN_NAMES[pid]: cell[pid]['ID']    for pid in illicit_ids}},
        {'Metric': 'Count', **{PATTERN_NAMES[pid]: cell[pid]['Count'] for pid in illicit_ids}},
        {'Metric': '%',     **{PATTERN_NAMES[pid]: cell[pid]['%']     for pid in illicit_ids}},
    ]

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex')
    return df


def build_comparable_filter_impact_wide(
    system_test_raw_df, comparable_test_raw_df,
    out_dir='tables', out_name='comparable_filter_impact_wide', csv_dir=None,
):
    """Wide three-group table showing how the comparable filter affects each pattern.

    Pattern 9 (unknown / no AttemptID) is excluded; covers patterns 1–8 only.

    Columns : pattern names (8) + Overall.
    Row groups:
      1. System (before filter)  : Attempts, Txns, Txn %
      2. Comparable (after filter): Attempts, Partial (N), Partial (%), Txns, Txn %
      3. Filter impact            : Txns lost, Txns lost (%)

    Attempts is repeated in the Comparable group for readability (values are
    identical to System since every attempt has ≥1 comparable transaction).
    Partial rows come before Txns in the Comparable group to keep a consistent
    ordering across groups.
    Txn % uses the system-split total as the denominator in both data groups so
    the rows are directly comparable.
    Txns lost (%) is the per-pattern loss rate (lost / system txns for that pattern).
    Partial attempt = an attempt missing ≥1 transaction after comparable filtering.
    """
    illicit_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    def _illicit(df):
        mask = (
            (df['Is Laundering'] == 1) &
            (df['AttemptID'] >= 0) &
            (df['Pattern'] >= 1) &
            (df['Pattern'] <= 8)
        )
        return df[mask].copy()

    sys_il  = _illicit(system_test_raw_df)
    comp_il = _illicit(comparable_test_raw_df)

    sys_txns      = sys_il.groupby('Pattern').size()
    sys_attempts  = sys_il.groupby('Pattern')['AttemptID'].nunique()
    comp_txns     = comp_il.groupby('Pattern').size()
    total_sys     = int(sys_txns.sum())
    total_comp    = int(comp_txns.sum())

    # Per-attempt completeness: how many sys-split txns survived in comparable
    sys_per_att  = sys_il.groupby(['AttemptID', 'Pattern']).size().rename('sys_n').reset_index()
    comp_per_att = comp_il.groupby('AttemptID').size().rename('comp_n').reset_index()
    att_df = sys_per_att.merge(comp_per_att, on='AttemptID', how='left')
    att_df['comp_n']  = att_df['comp_n'].fillna(0).astype(int)
    att_df['partial'] = att_df['comp_n'] < att_df['sys_n']
    partial_n     = att_df.groupby('Pattern')['partial'].sum()
    total_partial = int(att_df['partial'].sum())
    total_att     = int(att_df['AttemptID'].nunique())

    pnames = [PATTERN_NAMES.get(pid, f'P{pid}') for pid in illicit_ids]

    def _p(pid, series, default=0):
        return series.get(pid, default)

    groups = [
        ('System', [
            ('Attempts',
             {pid: int(_p(pid, sys_attempts))      for pid in illicit_ids},
             total_att),
            ('Txns',
             {pid: int(_p(pid, sys_txns))          for pid in illicit_ids},
             total_sys),
            ('Txn %',
             {pid: round(100 * _p(pid, sys_txns) / total_sys, 1) for pid in illicit_ids},
             100.0),
        ]),
        ('Comparable', [
            ('Attempts',
             {pid: int(_p(pid, sys_attempts))      for pid in illicit_ids},
             total_att),
            ('Txns',
             {pid: int(_p(pid, comp_txns))         for pid in illicit_ids},
             total_comp),
            ('Txn %',
             {pid: round(100 * _p(pid, comp_txns) / total_sys, 1) for pid in illicit_ids},
             round(100 * total_comp / total_sys, 1)),
        ]),
        ('Filter impact', [
            ('Partial (N)',
             {pid: int(_p(pid, partial_n)) for pid in illicit_ids},
             total_partial),
            ('Partial (%)',
             {pid: round(100 * _p(pid, partial_n) / max(int(_p(pid, sys_attempts)), 1), 1)
              for pid in illicit_ids},
             round(100 * total_partial / max(total_att, 1), 1)),
            ('Txns lost',
             {pid: int(_p(pid, sys_txns)) - int(_p(pid, comp_txns)) for pid in illicit_ids},
             total_sys - total_comp),
            ('Txns lost (%)',
             {pid: round(100 * (int(_p(pid, sys_txns)) - int(_p(pid, comp_txns))) /
                         max(int(_p(pid, sys_txns)), 1), 1)
              for pid in illicit_ids},
             round(100 * (total_sys - total_comp) / total_sys, 1)),
        ]),
    ]

    records = []
    for grp_label, metrics in groups:
        for metric, pat_vals, overall_val in metrics:
            rec = {'Group': grp_label, 'Metric': metric, 'Overall': overall_val}
            for pid in illicit_ids:
                rec[PATTERN_NAMES.get(pid, f'P{pid}')] = pat_vals[pid]
            records.append(rec)

    df = pd.DataFrame(records)

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)

    # --- custom LaTeX with group-header rows ---
    n_data_cols = len(illicit_ids) + 1          # 8 patterns + Overall
    n_total_cols = 1 + n_data_cols              # Metric + data
    col_spec = 'l' + 'r' * n_data_cols

    def _esc(s):
        return str(s).replace('%', r'\%')

    group_tex = {
        'System':        r'\textit{System (before filter)}',
        'Comparable':    r'\textit{Comparable (after filter)}',
        'Filter impact': r'\textit{Filter impact}',
    }

    lines = [
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        ' & '.join([''] + pnames + ['Overall']) + r' \\',
        r'\midrule',
    ]
    prev_grp = None
    for rec in records:
        grp = rec['Group']
        if grp != prev_grp:
            if prev_grp is not None:
                lines.append(r'\midrule')
            lines.append(
                r'\multicolumn{' + str(n_total_cols) + r'}{l}{'
                + group_tex.get(grp, grp) + r'} \\'
            )
            prev_grp = grp
        is_pct = '%' in rec['Metric']
        def _fmt(v):
            return f'{v:.2f}' if is_pct else str(int(v))
        vals = [_esc(_fmt(rec[p])) for p in pnames] + [_esc(_fmt(rec['Overall']))]
        lines.append(_esc(rec['Metric']) + ' & ' + ' & '.join(vals) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}']
    tex = '\n'.join(lines)
    with open(out_dir / f'{out_name}.tex', 'w') as f:
        f.write(tex)

    return df


# --- pattern recall builders ---


def build_pattern_recall(scenario_map, out_fig=None, top_k=9):
    if out_fig is None:
        out_fig = FIGS_DIR / 'pattern_analysis' / 'pattern_recall_topk.pdf'
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

    agg = pr_df.groupby(["ID", "Scenario", "Pattern"]).agg(
        P=("P", "mean"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
    ).reset_index()

    csv_dir = FIGS_DIR.parent / "tables" / "pattern_analysis"
    csv_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(csv_dir / "pattern_recall_summary.csv", index=False)

    top_patterns = (agg.groupby("Pattern")["P"].sum().sort_values(ascending=False).head(top_k).index.tolist())
    scenarios = agg["Scenario"].unique().tolist()
    x = np.arange(len(top_patterns))

    plt.figure(figsize=(11, 4))
    for scen in scenarios:
        sub = agg[(agg["Scenario"] == scen) & (agg["Pattern"].isin(top_patterns))].set_index("Pattern")
        y = [sub.loc[p, "recall_mean"] if p in sub.index else 0.0 for p in top_patterns]
        plt.plot(x, y, marker="o", label=scen)

    plt.xticks(x, [PATTERN_NAMES.get(p, str(p)) for p in top_patterns], rotation=15, ha='right')
    plt.ylim(0, 1.0)
    plt.xlabel("Pattern (top-K by support)")
    plt.ylabel("Recall on illicit edges (TP / P)")
    plt.legend()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    return pr_df, agg


def build_pattern_recall_comparison(
    scenario_map,
    scenario_ids,
    out_dir="tables",
    out_name="pattern_recall_compare",
    top_k=10,
    top_by="support_baseline",
    baseline_id=None,
    value="recall_mean",
    plot=False,
    out_fig=None,
    filter_indices=None,
):
    """filter_indices: optional set of lv['indices'] values to restrict to
    (e.g. indices of non-fragmented illicit test transactions).
    """
    if out_fig is None:
        out_fig = FIGS_DIR / 'pattern_analysis' / 'pattern_recall_compare.pdf'
    all_rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]["path"])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get("laundering_values", None)
            if lv is None or "Pattern" not in lv.columns:
                continue
            lv_full = lv  # full test split — needed for FP count in precision
            if filter_indices is not None:
                lv = lv[lv['indices'].isin(filter_indices)]
            pr = pattern_recall_from_laundering_df(lv)
            if pr is None or pr.empty:
                continue
            pp = pattern_precision_from_laundering_df(lv_full, df_pos=lv)
            if pp is not None and not pp.empty:
                pr = pr.merge(pp[["Pattern", "FP", "precision"]], on="Pattern", how="left")
            else:
                pr["FP"] = 0
                pr["precision"] = 0.0
            pr["ID"] = sid
            pr["Scenario"] = scenario_map[sid]["name"]
            pr["seed"] = seed
            all_rows.append(pr)

    if not all_rows:
        print("[WARN] No pattern recall data found for given scenarios.")
        return None, None

    pr_df = pd.concat(all_rows, ignore_index=True)

    agg = pr_df.groupby(["ID", "Scenario", "Pattern"]).agg(
        P=("P", "mean"),
        TP=("TP", "mean"),
        FP=("FP", "mean"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
    ).reset_index()

    if top_by == "support_baseline":
        if baseline_id is None:
            baseline_id = scenario_ids[0]
        base_support = agg[agg["ID"] == baseline_id].groupby("Pattern")["P"].sum().sort_values(ascending=False)
        patterns = base_support.head(top_k).index.tolist()
    else:
        overall_support = agg.groupby("Pattern")["P"].sum().sort_values(ascending=False)
        patterns = overall_support.head(top_k).index.tolist()

    sub = agg[agg["Pattern"].isin(patterns)].copy()
    pivot = sub.pivot_table(index="Pattern", columns="ID", values=value, aggfunc="mean").fillna(0.0)

    if baseline_id is not None and baseline_id in scenario_ids:
        support_col = sub[sub["ID"] == baseline_id].set_index("Pattern")["P"]
        pivot.insert(0, "P (baseline)", support_col.reindex(pivot.index).fillna(0).astype(int))
    else:
        support_mean = sub.groupby("Pattern")["P"].mean()
        pivot.insert(0, "P (avg)", support_mean.reindex(pivot.index).fillna(0).astype(int))

    pivot = pivot.reset_index()
    pivot["Pattern"] = pivot["Pattern"].map(PATTERN_NAMES)
    for sid in scenario_ids:
        if sid in pivot.columns:
            pivot[sid] = pivot[sid].astype(float).round(3)

    # Reorder scenario columns to match scenario_ids order
    ordered_sid_cols = [sid for sid in scenario_ids if sid in pivot.columns]
    non_sid_cols = [c for c in pivot.columns if c not in ordered_sid_cols]
    pivot = pivot[non_sid_cols + ordered_sid_cols]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_dir / f"{out_name}.csv", index=False)
    df_to_latex_table(pivot, out_dir / f"{out_name}.tex", col_formats=pct_col_formats(ordered_sid_cols))

    if plot:
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        x = np.arange(len(patterns))
        tick_labels = [PATTERN_NAMES.get(p, str(p)) for p in pivot["Pattern"].tolist()]
        plt.figure(figsize=(11, 4))
        for sid in scenario_ids:
            if sid not in pivot.columns:
                continue
            y = pivot[sid].to_numpy()
            plt.plot(x, y, marker="o", label=sid)
        plt.xticks(x, tick_labels, rotation=15, ha='right')
        plt.ylim(0, 1.0)
        plt.xlabel("Pattern")
        plt.ylabel("Recall (TP/P)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_fig)
        plt.close()

    return pivot, agg


def pattern_recall_delta_vs_baseline(pivot_df, scenario_ids, baseline_id, out_dir="tables", out_name="pattern_recall_delta"):
    df = pivot_df.copy()
    if baseline_id not in df.columns:
        raise ValueError(f"baseline_id '{baseline_id}' not in pivot columns: {df.columns.tolist()}")

    for sid in scenario_ids:
        if sid == baseline_id or sid not in df.columns:
            continue
        df[sid] = (df[sid] - df[baseline_id]).astype(float).round(3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{out_name}.csv", index=False)
    sid_cols = [sid for sid in scenario_ids if sid in df.columns]
    df_to_latex_table(df, out_dir / f"{out_name}.tex", col_formats=pct_col_formats(sid_cols))
    return df


def build_pattern_recall_precision_combined(
    scenario_map,
    scenario_ids,
    out_dir="tables",
    out_name="pattern_recall_precision",
    top_k=10,
    top_by="support_baseline",
    baseline_id=None,
    filter_indices=None,
):
    """Combined recall + precision table with grouped column headers.

    Output LaTeX layout:
      Pattern | P | <------ Recall ------> | <----- Precision ----->
              |   |  S2   F1   P2   V1     |  S2   F1   P2   V1
    """
    pivot_recall, agg = build_pattern_recall_comparison(
        scenario_map, scenario_ids,
        top_k=top_k, top_by=top_by, baseline_id=baseline_id,
        value="recall_mean", filter_indices=filter_indices,
    )
    if pivot_recall is None:
        return None, None

    pivot_prec, _ = build_pattern_recall_comparison(
        scenario_map, scenario_ids,
        top_k=top_k, top_by=top_by, baseline_id=baseline_id,
        value="precision_mean", filter_indices=filter_indices,
    )
    if pivot_prec is None:
        return None, None

    p_col = "P (baseline)" if "P (baseline)" in pivot_recall.columns else "P (avg)"
    sid_cols = [s for s in scenario_ids if s in pivot_recall.columns]
    n_sid = len(sid_cols)

    combined = pivot_recall[["Pattern", p_col] + sid_cols].copy()
    combined.columns = ["Pattern", p_col] + [f"R_{s}" for s in sid_cols]
    prec_by_pat = pivot_prec.set_index("Pattern")
    for sid in sid_cols:
        combined[f"Prec_{sid}"] = combined["Pattern"].map(prec_by_pat[sid])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / f"{out_name}.csv", index=False)

    # Column ranges for cmidrule (1-indexed; col 1=Pattern, 2=P, 3..=recall, then precision)
    rec_start  = 3
    rec_end    = 2 + n_sid
    prec_start = rec_end + 1
    prec_end   = rec_end + n_sid
    col_fmt    = "l" + "r" * (1 + 2 * n_sid)
    sid_str    = " & ".join(sid_cols)

    def _esc(s):
        return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    rows = [
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        f"Pattern & $P$ & "
        f"\\multicolumn{{{n_sid}}}{{c}}{{Recall}} & "
        f"\\multicolumn{{{n_sid}}}{{c}}{{Precision}} \\\\",
        f"\\cmidrule(lr){{{rec_start}-{rec_end}}}"
        f"\\cmidrule(lr){{{prec_start}-{prec_end}}}",
        f" & & {sid_str} & {sid_str} \\\\",
        r"\midrule",
    ]
    for _, row in combined.iterrows():
        r_vals   = " & ".join(_fmt_scaled(row[f'R_{s}'], scale=100, decimals=2, auto_widen=True)    for s in sid_cols)
        prec_vals = " & ".join(_fmt_scaled(row[f'Prec_{s}'], scale=100, decimals=2, auto_widen=True) for s in sid_cols)
        rows.append(f"{_esc(row['Pattern'])} & {int(row[p_col])} & {r_vals} & {prec_vals} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]

    tex_path = out_dir / f"{out_name}.tex"
    tex_path.write_text("\n".join(rows) + "\n")
    print(f"  Combined recall+precision table → {tex_path}")
    return combined, agg


# =============================================================================
# Cross-bank structure analysis
# =============================================================================


def load_raw_df(size='small', ir='HI'):
    """Load the formatted transactions CSV (already sorted by timestamp)."""
    from configs.paths import get_data_path
    path = Path(get_data_path()) / f"AML_work_study/data/formatted_transactions_{size}_{ir}.csv" #_withpatterns
    return pd.read_csv(path)


def enrich_raw_df_with_pattern_degree(raw_df, size='small', ir='HI'):
    """Add a 'pattern_degree' column to raw_df by parsing the original patterns file.

    The degree is determined per laundering attempt:
      - Fan-Out / Fan-In / Cycle / Random / Gather-Scatter: the explicit max degree or
        hop count from the attempt header (e.g. 'Max 16-degree Fan-Out').
      - Stack / Bipartite / Scatter-Gather: no explicit degree in the header, so the
        number of transactions in the attempt is used instead.

    The join works via EdgeID, which equals the original row index in HI-Small_Trans.csv
    before the timestamp sort applied by format_kaggle_files.py.

    Returns a copy of raw_df with a new nullable-integer column 'pattern_degree'.
    """
    import re, csv
    from configs.paths import get_data_path

    data_root = Path(get_data_path()) / 'AML_work_study' / 'data'
    patterns_path = data_root / f'{ir}-{size.capitalize()}_Patterns.txt'
    trans_path    = data_root / f'{ir}-{size.capitalize()}_Trans.csv'

    degree_re = re.compile(r'Max (\d+)')

    # Pass 1: parse patterns file → key → degree
    # Buffer each attempt's transaction keys, then assign degree at END.
    # Degree = header value if present, else transaction count for that attempt.
    key_to_degree = {}
    current_header_degree = None
    current_keys = []
    with open(patterns_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                m = degree_re.search(line)
                current_header_degree = int(m.group(1)) if m else None
                current_keys = []
            elif line.startswith('END LAUNDERING ATTEMPT'):
                degree = current_header_degree if current_header_degree is not None else len(current_keys)
                for key in current_keys:
                    key_to_degree[key] = degree
                current_keys = []
                current_header_degree = None
            elif current_keys is not None and line:
                parts = line.split(',')
                if len(parts) == 11:
                    current_keys.append(tuple(parts[:10]))

    # Pass 2: read original Trans CSV → build {row_index: degree}
    row_degree = {}
    with open(trans_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            # columns: Timestamp, From Bank, Account, To Bank, Account,
            #          Amount Received, Receiving Currency, Amount Paid,
            #          Payment Currency, Payment Format, Is Laundering
            key = tuple(row[:10])
            if key in key_to_degree:
                row_degree[i] = key_to_degree[key]

    degree_series = pd.Series(row_degree, dtype='Int64', name='pattern_degree')
    raw_df = raw_df.copy()
    raw_df['pattern_degree'] = raw_df['EdgeID'].map(degree_series)
    return raw_df


def recompute_txn_count_degree(split_df):
    """Correct pattern_degree for Stack / Scatter-Gather / Bipartite attempts.

    NOTE: currently unused. Removed from all callers to keep pattern_degree
    consistent — all patterns use their full structural degree from the patterns
    file (via enrich_raw_df_with_pattern_degree), regardless of split. Patterns
    5/6/8 before this correction used split-local counts while all others used
    full-attempt values, which was inconsistent. Revisit if split-local degree
    is explicitly needed for degree-bucketed recall analysis.

    These three patterns have no explicit degree in the attempt header so
    enrich_raw_df_with_pattern_degree assigns pattern_degree = transaction count
    of the FULL attempt (read from the patterns file).  After comparable
    filtering, fragmented attempts have fewer transactions visible, making the
    inherited degree inflated.  This function updates pattern_degree to the
    actual transaction count within split_df.

    Patterns corrected: 5 (Scatter-Gather), 6 (Stack), 8 (Bipartite).
    All other patterns carry header-declared degrees (fan width, hop count, …)
    and are left unchanged.  For complete attempts the correction is a no-op
    since the count matches the original degree.
    """
    TXN_COUNT_PATTERNS = {5, 6, 8}

    if 'pattern_degree' not in split_df.columns or 'AttemptID' not in split_df.columns:
        return split_df.copy()

    illicit = split_df[
        (split_df['Is Laundering'] == 1) &
        (split_df['AttemptID'] >= 0) &
        (split_df['Pattern'].isin(TXN_COUNT_PATTERNS))
    ]
    actual_counts = illicit.groupby('AttemptID').size()

    result = split_df.copy()
    mask = (
        (result['Is Laundering'] == 1) &
        (result['AttemptID'] >= 0) &
        (result['Pattern'].isin(TXN_COUNT_PATTERNS))
    )
    result.loc[mask, 'pattern_degree'] = (
        result.loc[mask, 'AttemptID'].map(actual_counts).astype('Int64')
    )
    return result


def load_comparable_banks(size='small', ir='HI', split_perc=(0.6, 0.2)):
    """Return the set of 630 individual bank IDs used in comparable evaluation.

    These are the banks from rb['individual']['banks'] in the relevant_banks JSON.
    Pass the returned set as party_banks to visibility functions so that max_vis
    is computed only over FL-participating banks, not non-630 intermediaries.
    Returns None if the JSON is not found.
    """
    import json
    from configs.paths import get_data_path
    rb_path = (Path(get_data_path()) /
               f"AML_work_study/experiments/relevant_banks"
               f"/{size}_{ir}__split_{split_perc[0]}_{split_perc[1]}.json")
    if not rb_path.exists():
        print(f"[WARN] relevant_banks JSON not found: {rb_path}")
        return None
    with open(rb_path) as f:
        rb = json.load(f)
    return set(rb['individual']['banks'])


def reconstruct_test_raw_df(raw_df, split_perc=(0.6, 0.2), comparable=True,
                             size='small', ir='HI'):
    """Reconstruct the test split DataFrame with the same reindexed integer index
    as laundering_values['indices'], so the two can be joined directly.

    In comparable mode, applies the same individual-bank index filter used during
    training, ensuring the row count and ordering match the saved laundering_values.

    Returns:
        test_raw_df : DataFrame indexed by [lv_offset, lv_offset+n_test) with all
                      raw_df columns (from_id, to_id, From Bank, To Bank, etc.).
        lv_offset   : int — n_train + n_vali (starting value of the test index block).
    """
    import torch, json, itertools
    from configs.paths import get_data_path

    timestamps = torch.tensor(raw_df['Timestamp'].to_numpy(), dtype=torch.float32)
    y = torch.tensor(raw_df['Is Laundering'].to_numpy(), dtype=torch.long)

    # Inline temporal split — replicates raw_data_processing.split_indices exactly
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    daily_inds, daily_trans = [], []
    for day in range(n_days):
        l, r = day * 24 * 3600, (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    d_ts = np.array(daily_trans)
    I = list(range(len(d_ts)))
    test_perc = round(1 - sum(split_perc), 10)
    split_perc_full = list(split_perc) + [test_perc]
    split_scores = {}
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            s = sum(totals)
            props = [v / s for v in totals]
            split_scores[(i, j)] = max(abs(v - t) / t for v, t in zip(props, split_perc_full))
    i_star, j_star = min(split_scores, key=split_scores.get)
    split = [list(range(i_star)), list(range(i_star, j_star)), list(range(j_star, len(d_ts)))]
    split_inds = {k: [daily_inds[day] for day in split[k]] for k in range(3)}
    all_positions = [np.concatenate([t.numpy() for t in split_inds[i]]) for i in range(3)]
    all_positions = [np.concatenate(split_inds[i]) for i in range(3)]

    if comparable:
        rb_path = (Path(get_data_path()) /
                   f"AML_work_study/experiments/relevant_banks"
                   f"/{size}_{ir}__split_{split_perc[0]}_{split_perc[1]}.json")
        with open(rb_path) as f:
            rb = json.load(f)
        individual_set = set(rb['individual']['indices'])

        train_pos = all_positions[0][np.isin(all_positions[0], list(individual_set))]
        vali_pos  = all_positions[1][np.isin(all_positions[1], list(individual_set))]
        test_pos  = all_positions[2][np.isin(all_positions[2], list(individual_set))]
    else:
        train_pos, vali_pos, test_pos = all_positions

    lv_offset = len(train_pos) + len(vali_pos)
    test_raw_df = raw_df.iloc[test_pos].copy()
    test_raw_df.index = pd.RangeIndex(lv_offset, lv_offset + len(test_pos))
    return test_raw_df, lv_offset


def reconstruct_train_raw_df(raw_df, split_perc=(0.6, 0.2), comparable=True,
                              size='small', ir='HI'):
    """Reconstruct the train split DataFrame using the same daily-boundary split.

    In comparable mode, applies the same individual-bank index filter used during
    training (identical to reconstruct_test_raw_df). Returns a 0-based RangeIndex.
    """
    import torch, json, itertools
    from configs.paths import get_data_path

    timestamps = torch.tensor(raw_df['Timestamp'].to_numpy(), dtype=torch.float32)

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_inds, daily_trans = [], []
    for day in range(n_days):
        l, r = day * 24 * 3600, (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    d_ts = np.array(daily_trans)
    I = list(range(len(d_ts)))
    test_perc = round(1 - sum(split_perc), 10)
    split_perc_full = list(split_perc) + [test_perc]
    split_scores = {}
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            s = sum(totals)
            props = [v / s for v in totals]
            split_scores[(i, j)] = max(abs(v - t) / t for v, t in zip(props, split_perc_full))
    i_star, j_star = min(split_scores, key=split_scores.get)
    split = [list(range(i_star)), list(range(i_star, j_star)), list(range(j_star, len(d_ts)))]
    split_inds = {k: [daily_inds[day] for day in split[k]] for k in range(3)}
    all_positions = [np.concatenate(split_inds[i]) for i in range(3)]

    if comparable:
        rb_path = (Path(get_data_path()) /
                   f"AML_work_study/experiments/relevant_banks"
                   f"/{size}_{ir}__split_{split_perc[0]}_{split_perc[1]}.json")
        with open(rb_path) as f:
            rb = json.load(f)
        individual_set = set(rb['individual']['indices'])
        train_pos = all_positions[0][np.isin(all_positions[0], list(individual_set))]
    else:
        train_pos = all_positions[0]

    train_raw_df = raw_df.iloc[train_pos].copy()
    train_raw_df.index = pd.RangeIndex(len(train_pos))
    return train_raw_df


def enrich_lv_with_raw(lv_df, test_raw_df):
    """Add From Bank, To Bank, EdgeID, is_cross_bank columns to a laundering_values df.

    Joins on lv_df['indices'] <-> test_raw_df.index (both use the reindexed integer block).
    Also passes through AttemptID, n_attempt_banks, txn_class if present in test_raw_df.
    """
    passthrough = ['From Bank', 'To Bank', 'EdgeID', 'pattern_degree',
                   'AttemptID', 'Pattern', 'n_attempt_banks', 'txn_class']
    cols = [c for c in passthrough if c in test_raw_df.columns and c not in lv_df.columns]
    enriched = lv_df.copy()
    enriched = enriched.join(test_raw_df[cols], on='indices', how='left')
    if 'From Bank' in enriched.columns and 'To Bank' in enriched.columns:
        enriched['is_cross_bank'] = enriched['From Bank'] != enriched['To Bank']
    return enriched


def _add_attempt_class_to_df(df, raw_df=None):
    """Add n_attempt_banks and txn_class columns to df (in-place copy).

    Computes bank span from raw_df (full dataset) when provided, otherwise from df.
    """
    if 'AttemptID' not in df.columns:
        return df
    span_src = raw_df if raw_df is not None else df
    illicit_all = span_src[(span_src['Is Laundering'] == 1) & (span_src['AttemptID'] >= 0)]
    attempt_n_banks = (
        illicit_all.groupby('AttemptID')
        .apply(lambda g: len(set(g['From Bank'].tolist() + g['To Bank'].tolist())))
        .rename('n_banks')
    )
    out = df.copy()
    out['n_attempt_banks'] = out['AttemptID'].map(attempt_n_banks)

    def _cls(row):
        if row.get('Is Laundering', 1) != 1:
            return None
        cb = row.get('From Bank') != row.get('To Bank')
        if cb:
            return 'CB'
        return 'WB-single' if row.get('n_attempt_banks', 1) == 1 else 'WB-multi'

    out['txn_class'] = out.apply(_cls, axis=1)
    return out


def cross_bank_recall_analysis(scenario_map, scenario_ids, test_raw_df,
                                out_dir='tables', out_name='cross_bank_recall'):
    """Recall on illicit transactions split by within-bank vs cross-bank.

    Reports TP / P separately for three groups:
        'all'          — all illicit transactions
        'within-bank'  — From Bank == To Bank
        'cross-bank'   — From Bank != To Bank

    Returns:
        pivot : wide DataFrame, one row per scenario.
        agg   : long aggregated DataFrame.
    """
    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, test_raw_df)
            illicit = lv_e[lv_e['true_y'] == 1]

            for label, mask in [
                ('all',         pd.Series(True,  index=illicit.index)),
                ('within-bank', illicit['is_cross_bank'] == False),
                ('cross-bank',  illicit['is_cross_bank'] == True),
            ]:
                sub = illicit[mask]
                p  = len(sub)
                tp = int((sub['pred_label'] == 1).sum())
                rows.append({'ID': sid, 'Scenario': scenario_map[sid]['name'],
                             'seed': seed, 'bank_type': label,
                             'P': p, 'TP': tp,
                             'recall': tp / p if p else np.nan})

    raw_rows = pd.DataFrame(rows)
    agg = (raw_rows.groupby(['ID', 'Scenario', 'bank_type'])
           .agg(P=('P', 'mean'), TP=('TP', 'mean'),
                recall_mean=('recall', 'mean'), recall_std=('recall', 'std'))
           .reset_index())

    pivot_rows = []
    for (sid, scen), grp in agg.groupby(['ID', 'Scenario']):
        row = {'ID': sid, 'Scenario': scen}
        for _, r in grp.iterrows():
            bt = r['bank_type']
            row[f'{bt}_P']    = int(round(r['P']))
            row[f'{bt}_TP']   = round(r['TP'], 1)
            row[f'{bt}_rec']  = round(r['recall_mean'], 3)
            row[f'{bt}_rec±'] = round(r['recall_std'], 3)
        pivot_rows.append(row)

    pivot = pd.DataFrame(pivot_rows)
    # Restore scenario_ids row order (groupby sorts alphabetically)
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    pivot = pivot.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pivot.to_csv(Path(out_dir) / f'{out_name}.csv', index=False)

    # LaTeX: TP + recall per group (P is constant across scenarios — goes in caption)
    tex_cols = {
        'Scenario':          'Scenario',
        'all_TP':            'All TP',
        'all_rec':           'All Rec',
        'within-bank_TP':    'WB TP',
        'within-bank_rec':   'WB Rec',
        'cross-bank_TP':     'CB TP',
        'cross-bank_rec':    'CB Rec',
    }
    pivot_tex = pivot[[c for c in tex_cols if c in pivot.columns]].rename(columns=tex_cols)
    rec_cols = [c for c in ['All Rec', 'WB Rec', 'CB Rec'] if c in pivot_tex.columns]
    df_to_latex_table(pivot_tex, Path(out_dir) / f'{out_name}.tex', col_formats=pct_col_formats(rec_cols))
    return pivot, agg


def views_recall_by_pattern(
    scenario_map,
    scenario_ids,
    test_raw_df,
    comparable_banks,
    filter_indices=None,
    out_dir='tables',
    out_name='views_recall_by_pattern',
    csv_dir=None,
):
    """Per-pattern recall and precision split by number of party views (1 or 2),
    restricted to cross-bank transactions.

    Within-bank transactions (From Bank == To Bank) are excluded entirely: there is
    only one distinct party on such a transaction, so it is not a case of a party
    being ineligible — V1 zero-pads the missing counterparty slot for these rather
    than losing a second, independent view. Including them would silently inflate
    the "2 views" bucket with transactions that never had two views to begin with.

    For the remaining cross-bank transactions, n_views = |{From Bank, To Bank} ∩
    comparable_banks|:
      1 view  — only one of the two (distinct) parties is in the eligible bank set
      2 views — both distinct parties are eligible

    Because every transaction has From Bank / To Bank, FP can also be bucketed by
    n_views, giving a views-specific denominator:
      precision_P_V = TP_P_V / (TP_P_V + FP_V)

    filter_indices (non-fragmented illicit indices) is applied only for TP/P;
    FP are always counted from the full lv so legitimate transactions are included.

    Output LaTeX layout:
      Pattern | <---------- 1 view ----------> | <---------- 2 views ---------->
              |  N   R-S2  R-V1  P-S2  P-V1   |  N   R-S2  R-V1  P-S2  P-V1

    Returns:
        combined : wide DataFrame (one row per Pattern).
    """
    comparable_set = set(comparable_banks)
    recall_rows = []
    prec_rows   = []

    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_full = enrich_lv_with_raw(lv, test_raw_df)
            lv_full = lv_full[lv_full['From Bank'] != lv_full['To Bank']]
            lv_full['n_views'] = (
                lv_full['From Bank'].isin(comparable_set).astype(int)
                + lv_full['To Bank'].isin(comparable_set).astype(int)
            )
            lv_full = lv_full[lv_full['n_views'] > 0]

            # FP per views group — from full lv (includes legitimate transactions)
            fp_by_views = (
                lv_full[(lv_full['pred_label'] == 1) & (lv_full['true_y'] == 0)]
                .groupby('n_views').size()
            )

            # TP/P — restrict to filter_indices for illicit only
            lv_filt = lv_full.copy()
            if filter_indices is not None:
                illicit_mask = lv_filt['true_y'] == 1
                lv_filt = lv_filt[~illicit_mask | lv_filt['indices'].isin(filter_indices)]
            illicit = lv_filt[lv_filt['true_y'] == 1]
            if illicit.empty:
                continue

            for (n_v, pat), grp in illicit.groupby(['n_views', 'Pattern']):
                p  = len(grp)
                tp = int((grp['pred_label'] == 1).sum())
                fp = int(fp_by_views.get(n_v, 0))
                recall_rows.append({
                    'ID': sid, 'seed': seed,
                    'n_views': int(n_v), 'Pattern': int(pat),
                    'P': p, 'TP': tp,
                    'recall': tp / p if p else np.nan,
                })
                prec_rows.append({
                    'ID': sid, 'seed': seed,
                    'n_views': int(n_v), 'Pattern': int(pat),
                    'FP': fp, 'TP': tp,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                })

    if not recall_rows:
        print("[WARN] views_recall_by_pattern: no data collected.")
        return None

    agg_rec = (
        pd.DataFrame(recall_rows)
        .groupby(['ID', 'n_views', 'Pattern'])
        .agg(P=('P', 'mean'), recall_mean=('recall', 'mean'))
        .reset_index()
    )
    agg_prec = (
        pd.DataFrame(prec_rows)
        .groupby(['ID', 'n_views', 'Pattern'])
        .agg(precision_mean=('precision', 'mean'))
        .reset_index()
    )

    patterns_sorted = sorted(agg_rec['Pattern'].unique())
    views_sorted    = sorted(agg_rec['n_views'].unique())

    pattern_rows = []
    for pat in patterns_sorted:
        pname = PATTERN_NAMES.get(pat, str(pat))
        row = {'Pattern': pname}
        for n_v in views_sorted:
            sub_r = agg_rec [(agg_rec ['n_views'] == n_v) & (agg_rec ['Pattern'] == pat)]
            sub_p = agg_prec[(agg_prec['n_views'] == n_v) & (agg_prec['Pattern'] == pat)]
            row[f'{n_v}v_N'] = int(round(sub_r['P'].mean())) if not sub_r.empty else 0
            for sid in scenario_ids:
                r = sub_r[sub_r['ID'] == sid]
                p = sub_p[sub_p['ID'] == sid]
                row[f'{n_v}v_R_{sid}'] = round(float(r['recall_mean'].iloc[0]),    3) if not r.empty else None
                row[f'{n_v}v_P_{sid}'] = round(float(p['precision_mean'].iloc[0]), 3) if not p.empty else None
        pattern_rows.append(row)

    combined = pd.DataFrame(pattern_rows)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir_p
    csv_path.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path / f'{out_name}.csv', index=False)

    # --- LaTeX: two header rows, grouped by views ---
    # Columns per view group: N + R-sid... + P-sid...
    n_sid = len(scenario_ids)
    n_per_view = 1 + 2 * n_sid      # N + recall*n_sid + precision*n_sid
    col_fmt = "l" + "r" * (len(views_sorted) * n_per_view)

    view_spans = []
    for i, nv in enumerate(views_sorted):
        start = 2 + i * n_per_view
        end   = start + n_per_view - 1
        view_spans.append((nv, start, end))

    multicolumns = " & ".join(
        f"\\multicolumn{{{n_per_view}}}{{c}}{{{nv} view{'s' if nv > 1 else ''}}}"
        for nv, _, _ in view_spans
    )
    cmidrule_str = "".join(
        f"\\cmidrule(lr){{{s}-{e}}}" for _, s, e in view_spans
    )
    rec_labels  = " & ".join(f"R-{s}" for s in scenario_ids)
    prec_labels = " & ".join(f"P-{s}" for s in scenario_ids)
    subheaders  = " & ".join(f"$N$ & {rec_labels} & {prec_labels}" for _ in views_sorted)

    def _esc(s):
        return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    latex_lines = [
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        f"Pattern & {multicolumns} \\\\",
        cmidrule_str,
        f" & {subheaders} \\\\",
        r"\midrule",
    ]
    for _, row in combined.iterrows():
        cells = [_esc(row['Pattern'])]
        for n_v in views_sorted:
            cells.append(str(row[f'{n_v}v_N']))
            for sid in scenario_ids:
                v = row[f'{n_v}v_R_{sid}']
                cells.append(_fmt_scaled(v, scale=100, decimals=2, auto_widen=True) if v is not None else "---")
            for sid in scenario_ids:
                v = row[f'{n_v}v_P_{sid}']
                cells.append(_fmt_scaled(v, scale=100, decimals=2, auto_widen=True) if v is not None else "---")
        latex_lines.append(" & ".join(cells) + r" \\")
    latex_lines += [r"\bottomrule", r"\end{tabular}"]

    tex_path = out_dir_p / f'{out_name}.tex'
    tex_path.write_text("\n".join(latex_lines) + "\n")
    print(f"  Views recall+precision table → {tex_path}")
    return combined


def txn_class_recall_analysis(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                               out_dir='tables', out_name='txn_class_recall'):
    """Recall per transaction class (WB-single / WB-multi / CB) for each scenario.

    Requires AttemptID in test_raw_df. Bank span computed from raw_df if provided.
    LaTeX table shows TP + recall per class (P fixed by data, goes in caption).
    """
    enriched_test = _add_attempt_class_to_df(test_raw_df, raw_df=raw_df)
    if 'txn_class' not in enriched_test.columns:
        print("[WARN] txn_class could not be computed — AttemptID missing.")
        return None, None

    classes = ['WB-single', 'WB-multi', 'CB']
    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, enriched_test)
            illicit = lv_e[lv_e['true_y'] == 1]
            for cls in classes:
                sub = illicit[illicit['txn_class'] == cls]
                p  = len(sub)
                tp = int((sub['pred_label'] == 1).sum())
                rows.append({'ID': sid, 'Scenario': scenario_map[sid]['name'],
                             'seed': seed, 'txn_class': cls,
                             'P': p, 'TP': tp,
                             'recall': tp / p if p else np.nan})

    raw_rows = pd.DataFrame(rows)
    agg = (raw_rows.groupby(['ID', 'Scenario', 'txn_class'])
           .agg(P=('P', 'mean'), TP=('TP', 'mean'),
                recall_mean=('recall', 'mean'), recall_std=('recall', 'std'))
           .reset_index())

    pivot_rows = []
    for (sid, scen), grp in agg.groupby(['ID', 'Scenario']):
        row = {'ID': sid, 'Scenario': scen}
        for _, r in grp.iterrows():
            cls = r['txn_class']
            row[f'{cls}_P']    = int(round(r['P']))
            row[f'{cls}_TP']   = round(r['TP'], 1)
            row[f'{cls}_rec']  = round(r['recall_mean'], 3)
            row[f'{cls}_rec±'] = round(r['recall_std'], 3)
        pivot_rows.append(row)

    pivot = pd.DataFrame(pivot_rows)
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    pivot = pivot.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pivot.to_csv(Path(out_dir) / f'{out_name}.csv', index=False)

    tex_cols = {'Scenario': 'Scenario'}
    for cls, short in [('WB-single', 'WBS'), ('WB-multi', 'WBM'), ('CB', 'CB')]:
        tex_cols[f'{cls}_TP']  = f'{short} TP'
        tex_cols[f'{cls}_rec'] = f'{short} Rec'
    pivot_tex = pivot[[c for c in tex_cols if c in pivot.columns]].rename(columns=tex_cols)
    rec_cols = [c for c in tex_cols.values() if c.endswith(' Rec')]
    rec_cols = [c for c in rec_cols if c in pivot_tex.columns]
    df_to_latex_table(pivot_tex, Path(out_dir) / f'{out_name}.tex', col_formats=pct_col_formats(rec_cols))
    return pivot, agg


def attempt_level_recall_analysis(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                                   out_dir='tables', out_name='attempt_level_recall'):
    """Recall by detection threshold: patterns + Overall as columns, rows grouped by threshold.

    Four row groups: ≥1, ≥50%, 100% (attempt-level), Individual (transaction-level TP/P).
    Each group contains one row per scenario.
    Columns: one per laundering pattern present in the test split, plus Overall.

    Requires AttemptID and Pattern columns in test_raw_df (from formatted_transactions CSV).
    """
    enriched_test = _add_attempt_class_to_df(test_raw_df, raw_df=raw_df)
    if 'AttemptID' not in enriched_test.columns or 'Pattern' not in enriched_test.columns:
        print("[WARN] AttemptID or Pattern missing — skipping attempt_level_recall_analysis.")
        return None, None

    illicit_ref = enriched_test[
        (enriched_test['Is Laundering'] == 1) &
        (enriched_test['AttemptID'] >= 0) &
        (enriched_test['Pattern'] >= 1) &
        (enriched_test['Pattern'] <= 8)
    ]
    patterns_present = sorted(illicit_ref['Pattern'].unique())

    def _attempt_recalls(sub):
        if len(sub) == 0:
            return np.nan, np.nan, np.nan
        ag = sub.groupby('AttemptID').agg(
            n_txns=('pred_label', 'count'),
            n_det=('pred_label', lambda x: (x == 1).sum()),
        ).reset_index()
        p = len(ag)
        if p == 0:
            return np.nan, np.nan, np.nan
        r_any  = (ag['n_det'] >= 1).sum() / p
        r_half = (ag['n_det'] >= ag['n_txns'] * 0.5).sum() / p
        r_all  = (ag['n_det'] == ag['n_txns']).sum() / p
        return r_any, r_half, r_all

    def _txn_recall(sub):
        if len(sub) == 0:
            return np.nan
        return (sub['pred_label'] == 1).sum() / len(sub)

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, enriched_test)
            if 'Pattern' not in lv_e.columns:
                continue
            illicit = lv_e[
                (lv_e['true_y'] == 1) &
                lv_e['AttemptID'].notna() &
                (lv_e['AttemptID'] >= 0) &
                (lv_e['Pattern'] >= 1) &
                (lv_e['Pattern'] <= 8)
            ].copy()

            row = {'ID': sid, 'Scenario': scenario_map[sid]['name'], 'seed': seed}
            for pat in patterns_present:
                sub = illicit[illicit['Pattern'] == pat]
                r_any, r_half, r_all = _attempt_recalls(sub)
                row[f'p{pat}_any']  = r_any
                row[f'p{pat}_half'] = r_half
                row[f'p{pat}_all']  = r_all
                row[f'p{pat}_txn']  = _txn_recall(sub)
            r_any, r_half, r_all = _attempt_recalls(illicit)
            row['overall_any']  = r_any
            row['overall_half'] = r_half
            row['overall_all']  = r_all
            row['overall_txn']  = _txn_recall(illicit)
            rows.append(row)

    raw_rows = pd.DataFrame(rows)
    agg_spec = {}
    for pat in patterns_present:
        for m in ['any', 'half', 'all', 'txn']:
            c = f'p{pat}_{m}'
            agg_spec[c] = (c, 'mean')
    for m in ['any', 'half', 'all', 'txn']:
        c = f'overall_{m}'
        agg_spec[c] = (c, 'mean')

    agg = (raw_rows.groupby(['ID', 'Scenario'])
           .agg(**agg_spec)
           .reset_index())
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    agg = agg.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    # Four row groups: three attempt-level thresholds + transaction-level
    thresh_info = [('>=1', 'any'), ('>=50%', 'half'), ('100%', 'all'), ('Individual', 'txn')]
    pat_col_names = {pat: PATTERN_NAMES.get(pat, f'P{pat}') for pat in patterns_present}
    data_cols = [pat_col_names[p] for p in patterns_present] + ['Overall']

    long_rows = []
    for thresh_label, thresh_key in thresh_info:
        for _, row in agg.iterrows():
            r = {'Threshold': thresh_label, 'ID': row['ID'], 'Scenario': row['Scenario']}
            for pat in patterns_present:
                r[pat_col_names[pat]] = row[f'p{pat}_{thresh_key}']
            r['Overall'] = row[f'overall_{thresh_key}']
            long_rows.append(r)

    pivot = pd.DataFrame(long_rows)
    for col in data_cols:
        pivot[col] = pivot[col].round(3)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pivot.to_csv(Path(out_dir) / f'{out_name}.csv', index=False)

    # LaTeX: threshold groups as row blocks, patterns + Overall as columns
    n_scen = len(scenario_ids)
    col_format = 'll' + 'c' * len(data_cols)

    def _tex(s):
        return (s.replace('&', r'\&')
                 .replace('%', r'\%')
                 .replace('>=', r'$\geq$'))

    header = 'Threshold & Scenario & ' + ' & '.join(_tex(c) for c in data_cols) + r' \\'

    body_lines = []
    for t_idx, (thresh_label, thresh_key) in enumerate(thresh_info):
        sub = pivot[pivot['Threshold'] == thresh_label]
        for s_idx, (_, row) in enumerate(sub.iterrows()):
            thresh_cell = (rf'\multirow{{{n_scen}}}{{*}}{{{_tex(thresh_label)}}}'
                           if s_idx == 0 else '')
            cells = [thresh_cell, str(row['Scenario'])]
            for col in data_cols:
                val = row[col]
                cells.append(_fmt_scaled(val, scale=100, decimals=2, auto_widen=True) if pd.notna(val) else '--')
            body_lines.append(' & '.join(cells) + r' \\')
        if t_idx < len(thresh_info) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (Path(out_dir) / f'{out_name}.tex').write_text('\n'.join(latex_lines))

    return pivot, agg


def pattern_cross_bank_profile(test_raw_df, out_dir='tables',
                                out_name='pattern_cross_bank_profile'):
    """For each laundering pattern in the test set, compute cross-bank fraction.

    Patterns with high cross-bank fraction are structurally harder for FL because
    no single bank's subgraph contains the full k-hop context around those edges.

    Returns:
        DataFrame: Pattern, Pattern_name, total, cross_bank_count, cross_bank_pct.
    """
    df = test_raw_df.copy()
    if 'is_cross_bank' not in df.columns:
        df['is_cross_bank'] = df['From Bank'] != df['To Bank']

    illicit = df[df['Is Laundering'] == 1]
    if 'Pattern' not in illicit.columns:
        print("[WARN] No Pattern column in test_raw_df — skipping.")
        return None

    grp = (illicit.groupby('Pattern')
           .agg(total=('is_cross_bank', 'count'),
                cross_bank_count=('is_cross_bank', 'sum'))
           .reset_index())
    grp['cross_bank_pct'] = grp['cross_bank_count'] / grp['total']
    grp['Pattern_name'] = grp['Pattern'].map(PATTERN_NAMES).fillna('?')
    grp = grp[['Pattern', 'Pattern_name', 'total', 'cross_bank_count', 'cross_bank_pct']]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grp.to_csv(Path(out_dir) / f'{out_name}.csv', index=False)
    return grp


# =============================================================================
# Pattern scale (degree) analysis
# =============================================================================

def _assign_degree_bins(degrees: pd.Series, n_bins: int = 4):
    """Bin integer degree values into at most n_bins groups labeled by actual range.

    Groups are formed by splitting the sorted unique values into n_bins roughly
    equal chunks. If there are <= n_bins unique values, each value is its own bin.

    Returns:
        mapped  : Series of label strings aligned with the input index (None for NaN).
        ordered : list of label strings in ascending order.
    """
    valid = degrees.dropna().astype(int)
    if valid.empty:
        return degrees.map(lambda _: None), []

    sorted_unique = sorted(valid.unique())
    n_unique = len(sorted_unique)

    if n_unique <= n_bins:
        lmap = {v: str(v) for v in sorted_unique}
        ordered = [str(v) for v in sorted_unique]
    else:
        boundaries = np.linspace(0, n_unique, n_bins + 1).astype(int)
        lmap = {}
        ordered = []
        for i in range(n_bins):
            group = sorted_unique[boundaries[i]:boundaries[i + 1]]
            if not group:
                continue
            lo, hi = group[0], group[-1]
            label = f'{lo}–{hi}' if lo != hi else str(lo)
            for v in group:
                lmap[v] = label
            ordered.append(label)

    result = degrees.map(lambda x: lmap.get(int(x)) if pd.notna(x) else None)
    return result, ordered


def build_pattern_degree_distribution(test_raw_df, out_dir='tables',
                                       out_name='pattern_degree_distribution',
                                       n_bins=4, csv_dir=None):
    """Distribution of laundering-attempt complexity (degree) per pattern type.

    Rows grouped by pattern with \\midrule separators; within each pattern, one
    row per degree bin.  Columns: Pattern, Degree, Count, % of pattern.

    Returns:
        df : long DataFrame.
    """
    illicit = test_raw_df[test_raw_df['Is Laundering'] == 1].copy()
    illicit = illicit.dropna(subset=['pattern_degree'])
    illicit['pattern_degree'] = illicit['pattern_degree'].astype(int)

    rows = []
    group_sizes = []

    for pid, grp in illicit.groupby('Pattern', sort=True):
        pname = PATTERN_NAMES.get(int(pid), str(pid))
        bin_labels, ordered = _assign_degree_bins(grp['pattern_degree'], n_bins)
        grp = grp.copy()
        grp['degree_bin'] = bin_labels.values

        pat_total = len(grp)
        n_rows = 0
        for label in ordered:
            sub = grp[grp['degree_bin'] == label]
            count = len(sub)
            if count == 0:
                continue
            rows.append({
                'Pattern': pname,
                'Degree': label,
                'Count': count,
                '% of pattern': round(100.0 * count / pat_total, 1),
            })
            n_rows += 1
        if n_rows > 0:
            group_sizes.append(n_rows)

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex', row_group_sizes=group_sizes)
    return df


def build_pattern_degree_recall(scenario_map, scenario_ids, test_raw_df,
                                 out_dir='tables', out_name='pattern_degree_recall',
                                 n_bins=4, csv_dir=None):
    """Recall per (pattern x degree bin) for each scenario, averaged across seeds.

    Joins laundering_values with test_raw_df to get pattern_degree, then computes
    TP/P per (pattern, degree bin, scenario).

    Rows grouped by pattern with \\midrule separators.
    Columns: Pattern, Degree, Count (mean P across seeds), recall per scenario ID.

    Returns:
        df : wide DataFrame, or None if no data.
    """
    illicit_ref = test_raw_df[test_raw_df['Is Laundering'] == 1].copy()
    illicit_ref = illicit_ref.dropna(subset=['pattern_degree'])
    illicit_ref['pattern_degree'] = illicit_ref['pattern_degree'].astype(int)

    # Pre-compute per-pattern bins — same binning as distribution table
    bin_defs = {}  # pid -> (val_to_label, ordered_labels)
    for pid, grp in illicit_ref.groupby('Pattern', sort=True):
        bin_labels, ordered = _assign_degree_bins(grp['pattern_degree'], n_bins)
        val_to_label = {}
        for v, l in zip(grp['pattern_degree'].astype(int), bin_labels):
            if l is not None and v not in val_to_label:
                val_to_label[v] = l
        bin_defs[int(pid)] = (val_to_label, ordered)

    seed_rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, test_raw_df)
            illicit = lv_e[lv_e['true_y'] == 1].copy()
            illicit = illicit.dropna(subset=['pattern_degree', 'Pattern'])
            illicit['pattern_degree'] = illicit['pattern_degree'].astype(int)
            illicit['Pattern'] = illicit['Pattern'].astype(int)

            for pid, grp in illicit.groupby('Pattern', sort=True):
                pid = int(pid)
                if pid not in bin_defs:
                    continue
                val_to_label, ordered = bin_defs[pid]
                grp = grp.copy()
                grp['degree_bin'] = grp['pattern_degree'].map(val_to_label)

                for label in ordered:
                    sub = grp[grp['degree_bin'] == label]
                    p = len(sub)
                    tp = int((sub['pred_label'] == 1).sum())
                    seed_rows.append({
                        'Pattern': pid,
                        'Degree': label,
                        'ID': sid,
                        'seed': seed,
                        'P': p,
                        'TP': tp,
                        'recall': tp / p if p else np.nan,
                    })

    if not seed_rows:
        print('[WARN] No pattern degree recall data found.')
        return None

    raw = pd.DataFrame(seed_rows)
    agg = (raw.groupby(['Pattern', 'Degree', 'ID'])
           .agg(recall_mean=('recall', 'mean'), P=('P', 'mean'))
           .reset_index())

    rows = []
    group_sizes = []
    for pid in sorted(bin_defs.keys()):
        pname = PATTERN_NAMES.get(pid, str(pid))
        _, ordered = bin_defs[pid]
        grp_pat = agg[agg['Pattern'] == pid]
        n_rows = 0
        for label in ordered:
            sub = grp_pat[grp_pat['Degree'] == label]
            if sub.empty:
                continue
            p_val = sub['P'].mean()
            row = {'Pattern': pname, 'Degree': label, 'Count': int(round(p_val))}
            for sid in scenario_ids:
                s = sub[sub['ID'] == sid]
                row[sid] = round(float(s['recall_mean'].iloc[0]), 3) if not s.empty else None
            rows.append(row)
            n_rows += 1
        if n_rows > 0:
            group_sizes.append(n_rows)

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    sid_cols = [sid for sid in scenario_ids if sid in df.columns]
    df_to_latex_table(df, out_dir / f'{out_name}.tex', row_group_sizes=group_sizes,
                       col_formats=pct_col_formats(sid_cols))
    return df


def build_pattern_degree_distribution_wide(test_raw_df, out_dir='tables',
                                            out_name='pattern_degree_distribution_wide',
                                            n_bins=4, csv_dir=None):
    """Wide-format degree distribution: patterns as columns, bin ranks as rows.

    Each cell shows the degree range and share: "lo–hi (pct%)".
    Rows are labeled Q1 (lowest degrees) through Q4 (highest).
    Much more compact than the long format for the same information.

    Returns:
        df : wide DataFrame with columns [Bin, *pattern_names].
    """
    illicit = test_raw_df[test_raw_df['Is Laundering'] == 1].copy()
    illicit = illicit.dropna(subset=['pattern_degree'])
    illicit['pattern_degree'] = illicit['pattern_degree'].astype(int)

    pattern_bins = {}  # pname -> list of (range_label, count, pct) in Q1..Qn order
    for pid, grp in illicit.groupby('Pattern', sort=True):
        pname = PATTERN_NAMES.get(int(pid), str(pid))
        bin_labels, ordered = _assign_degree_bins(grp['pattern_degree'], n_bins)
        grp = grp.copy()
        grp['degree_bin'] = bin_labels.values
        pat_total = len(grp)
        bins = []
        for label in ordered:
            sub = grp[grp['degree_bin'] == label]
            count = len(sub)
            pct = round(100.0 * count / pat_total, 1) if pat_total > 0 else 0.0
            bins.append((label, count, pct))
        pattern_bins[pname] = bins

    max_bins = max(len(v) for v in pattern_bins.values())
    pattern_names = list(pattern_bins.keys())

    rows = []
    for i in range(max_bins):
        row = {'Bin': f'Q{i + 1}'}
        for pname in pattern_names:
            bins = pattern_bins.get(pname, [])
            if i < len(bins):
                label, count, pct = bins[i]
                row[pname] = f'{label} ({pct}%)'
            else:
                row[pname] = '--'
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex')
    return df


def build_pattern_degree_recall_wide(scenario_map, scenario_ids, test_raw_df,
                                      out_dir='tables',
                                      out_name='pattern_degree_recall_wide',
                                      n_bins=4, csv_dir=None):
    """Wide-format recall table: patterns as columns, (scenario x bin) rows.

    Rows are grouped by scenario with \\midrule between groups.
    Each cell shows the recall (mean across seeds) formatted to 3 decimal places,
    or '--' when no data.

    Returns:
        df : wide DataFrame with columns [Scenario, Bin, *pattern_names].
    """
    illicit_ref = test_raw_df[test_raw_df['Is Laundering'] == 1].copy()
    illicit_ref = illicit_ref.dropna(subset=['pattern_degree'])
    illicit_ref['pattern_degree'] = illicit_ref['pattern_degree'].astype(int)

    bin_defs = {}  # pid (int) -> (val_to_label, ordered_labels)
    for pid, grp in illicit_ref.groupby('Pattern', sort=True):
        bin_labels, ordered = _assign_degree_bins(grp['pattern_degree'], n_bins)
        val_to_label = {}
        for v, l in zip(grp['pattern_degree'].astype(int), bin_labels):
            if l is not None and v not in val_to_label:
                val_to_label[v] = l
        bin_defs[int(pid)] = (val_to_label, ordered)

    seed_rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, test_raw_df)
            illicit = lv_e[lv_e['true_y'] == 1].copy()
            illicit = illicit.dropna(subset=['pattern_degree', 'Pattern'])
            illicit['pattern_degree'] = illicit['pattern_degree'].astype(int)
            illicit['Pattern'] = illicit['Pattern'].astype(int)

            for pid, grp in illicit.groupby('Pattern', sort=True):
                pid = int(pid)
                if pid not in bin_defs:
                    continue
                val_to_label, ordered = bin_defs[pid]
                grp = grp.copy()
                grp['degree_bin'] = grp['pattern_degree'].map(val_to_label)
                for label in ordered:
                    sub = grp[grp['degree_bin'] == label]
                    p = len(sub)
                    tp = int((sub['pred_label'] == 1).sum())
                    seed_rows.append({
                        'Pattern': pid, 'Degree': label,
                        'ID': sid, 'seed': seed,
                        'P': p, 'TP': tp,
                        'recall': tp / p if p else np.nan,
                    })

    if not seed_rows:
        print('[WARN] No pattern degree recall data found.')
        return None

    raw = pd.DataFrame(seed_rows)
    agg = (raw.groupby(['Pattern', 'Degree', 'ID'])
           .agg(recall_mean=('recall', 'mean'))
           .reset_index())

    max_bins = max(len(v[1]) for v in bin_defs.values())

    rows = []
    group_sizes = []
    for sid in scenario_ids:
        scen_name = scenario_map[sid]['name']
        n_rows = 0
        for i in range(max_bins):
            row = {'Scenario': scen_name, 'Bin': f'Q{i + 1}'}
            for pid in sorted(bin_defs.keys()):
                pname = PATTERN_NAMES.get(pid, str(pid))
                _, ordered = bin_defs[pid]
                if i >= len(ordered):
                    row[pname] = '--'
                    continue
                label = ordered[i]
                sub = agg[(agg['Pattern'] == pid) & (agg['Degree'] == label) & (agg['ID'] == sid)]
                if sub.empty:
                    row[pname] = '--'
                else:
                    row[pname] = _fmt_scaled(sub["recall_mean"].iloc[0], scale=100, decimals=2, auto_widen=True)
            rows.append(row)
            n_rows += 1
        group_sizes.append(n_rows)

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex', row_group_sizes=group_sizes)
    return df


def _build_adj(raw_df):
    """Build node-to-incident-edges adjacency over the full transaction graph."""
    from_ids = raw_df['from_id'].to_numpy(dtype=int)
    to_ids   = raw_df['to_id'].to_numpy(dtype=int)
    is_cross = (raw_df['From Bank'] != raw_df['To Bank']).to_numpy()

    adj = defaultdict(list)
    for idx in range(len(raw_df)):
        adj[int(from_ids[idx])].append((idx, int(to_ids[idx]),   bool(is_cross[idx])))
        adj[int(to_ids[idx])].append(  (idx, int(from_ids[idx]), bool(is_cross[idx])))

    return adj, from_ids, to_ids


def _khop_cross_bank_frac(from_node, to_node, adj, k):
    """BFS k hops from both endpoints of a seed edge; return cross-bank fraction."""
    seen_nodes = {from_node, to_node}
    frontier = {from_node, to_node}
    cross_vals = []

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            for _, neighbor, is_cross in adj[node]:
                cross_vals.append(is_cross)
                if neighbor not in seen_nodes:
                    seen_nodes.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier

    return float(np.mean(cross_vals)) if cross_vals else 0.0


def compute_khop_cross_bank_fracs(raw_df, test_raw_df, k=2):
    """For every row in test_raw_df compute the k-hop neighborhood cross-bank fraction
    in the full raw_df graph (all time periods).

    Returns:
        Series indexed like test_raw_df, values in [0, 1].
    """
    adj, _, _ = _build_adj(raw_df)
    from_ids = test_raw_df['from_id'].to_numpy(dtype=int)
    to_ids   = test_raw_df['to_id'].to_numpy(dtype=int)

    n = len(from_ids)
    fracs = np.zeros(n)
    for i in range(n):
        fracs[i] = _khop_cross_bank_frac(from_ids[i], to_ids[i], adj, k)
        if i % 20_000 == 0 and i > 0:
            print(f"  k-hop cross-bank: {i}/{n} ({100*i/n:.0f}%)")

    return pd.Series(fracs, index=test_raw_df.index, name=f'cb_frac_k{k}')


def neighborhood_recall_analysis(scenario_map, scenario_ids, cb_fracs, test_raw_df,
                                  n_bins=5, out_dir='tables',
                                  out_name='neighborhood_cb_recall',
                                  out_fig=None):
    """Recall vs k-hop neighborhood cross-bank fraction, binned into quantile buckets.

    Args:
        cb_fracs  : Series from compute_khop_cross_bank_fracs.
        n_bins    : number of quantile bins.

    Returns:
        pivot : recall per (bin, scenario_id).
        agg   : long aggregated DataFrame.
    """
    if out_fig is None:
        out_fig = FIGS_DIR / 'pattern_analysis' / 'neighborhood_cb_recall.pdf'
    bin_labels = [f"Q{i+1}" for i in range(n_bins)]
    bin_edges = np.quantile(cb_fracs.values, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, test_raw_df)
            lv_e = lv_e.copy()
            lv_e['cb_frac'] = cb_fracs.reindex(lv_e['indices']).values

            illicit = lv_e[lv_e['true_y'] == 1].copy()
            illicit['bin'] = pd.cut(illicit['cb_frac'], bins=bin_edges,
                                    labels=bin_labels, include_lowest=True)

            for b, grp in illicit.groupby('bin', observed=True):
                p  = len(grp)
                tp = int((grp['pred_label'] == 1).sum())
                rows.append({'ID': sid, 'Scenario': scenario_map[sid]['name'],
                             'seed': seed, 'bin': str(b), 'P': p, 'TP': tp,
                             'recall': tp / p if p else np.nan})

    raw_rows = pd.DataFrame(rows)
    agg = (raw_rows.groupby(['ID', 'Scenario', 'bin'])
           .agg(P=('P', 'mean'), recall_mean=('recall', 'mean'),
                recall_std=('recall', 'std'))
           .reset_index())

    bin_descs = [f"Q{i+1} ({bin_edges[i]*100:.0f}–{(bin_edges[i+1]-1e-9)*100:.0f}%)"
                 for i in range(n_bins)]

    pivot = agg.pivot_table(index='bin', columns='ID', values='recall_mean').reset_index()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pivot.to_csv(Path(out_dir) / f'{out_name}.csv', index=False)

    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(n_bins)
    plt.figure(figsize=(8, 4))
    for sid in scenario_ids:
        sub = agg[agg['ID'] == sid].set_index('bin')
        y    = [sub.loc[b, 'recall_mean'] if b in sub.index else np.nan for b in bin_labels]
        yerr = [sub.loc[b, 'recall_std']  if b in sub.index else np.nan for b in bin_labels]
        plt.errorbar(x, y, yerr=yerr, marker='o', label=sid, capsize=3)
    plt.xticks(x, bin_descs, rotation=15, ha='right')
    plt.ylim(0, 1.0)
    plt.xlabel("Cross-bank fraction in k-hop neighborhood")
    plt.ylabel("Recall (TP / P) on illicit edges")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

    return pivot, agg


def build_attempt_bank_span_table(test_raw_df, raw_df=None, out_dir='tables',
                                  out_name='attempt_bank_span', csv_dir=None,
                                  ref_test_raw_df=None):
    """For each laundering attempt (test split), count distinct banks involved.

    Only counts attempts that have at least one transaction in test_raw_df,
    so the population matches build_attempt_transaction_class_table.

    Bank span and total transaction count are computed from raw_df (full dataset)
    when provided, so the full attempt structure is captured regardless of split.
    Falls back to test_raw_df if raw_df is not given.

    ref_test_raw_df : optional system-eval test split.  When provided (comparable
        evaluation only), two extra columns are added per pattern:
        - Partial (%): fraction of attempts missing at least one transaction
          relative to the system split (completeness < 1.0).
        - Mean completeness (%): average fraction of system-split transactions
          visible in the comparable split across all attempts of that pattern.

    Requires AttemptID column — regenerate CSV via format_kaggle_files.py if missing.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    if 'AttemptID' not in span_src.columns:
        print("[WARN] AttemptID column not found. "
              "Re-run format_kaggle_files.py to regenerate the CSV.")
        return None

    # Attempts that appear in the test split
    test_illicit = test_raw_df[(test_raw_df['Is Laundering'] == 1) & (test_raw_df['AttemptID'] >= 0)]
    test_attempt_ids = set(test_illicit['AttemptID'].unique())

    # Bank span computed from full dataset, restricted to test-split attempts
    illicit_all = span_src[(span_src['Is Laundering'] == 1) & (span_src['AttemptID'] >= 0)]
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]

    # Count distinct banks and total transaction count per attempt (From Bank ∪ To Bank)
    attempt_stats = (
        illicit_all.groupby(['AttemptID', 'Pattern'])
        .apply(lambda g: pd.Series({
            'n_banks': len(set(g['From Bank'].tolist() + g['To Bank'].tolist())),
            'n_txns':  len(g),
        }))
        .reset_index()
    )

    # Completeness: comparable txns / system txns per attempt (comparable eval only)
    has_completeness = ref_test_raw_df is not None
    if has_completeness:
        comp_illicit = test_raw_df[
            (test_raw_df['Is Laundering'] == 1) & (test_raw_df['AttemptID'] >= 0)
        ]
        comp_counts = (
            comp_illicit[comp_illicit['AttemptID'].isin(test_attempt_ids)]
            .groupby('AttemptID').size()
        )
        ref_illicit = ref_test_raw_df[
            (ref_test_raw_df['Is Laundering'] == 1) & (ref_test_raw_df['AttemptID'] >= 0)
        ]
        ref_counts = (
            ref_illicit[ref_illicit['AttemptID'].isin(test_attempt_ids)]
            .groupby('AttemptID').size()
        )
        completeness_s = (comp_counts / ref_counts).rename('completeness')
        attempt_stats = attempt_stats.merge(
            completeness_s.reset_index().rename(columns={'index': 'AttemptID'}),
            on='AttemptID', how='left',
        )
        attempt_stats['completeness'] = attempt_stats['completeness'].fillna(1.0)

    rows = []
    for pid in sorted(attempt_stats['Pattern'].unique()):
        sub = attempt_stats[attempt_stats['Pattern'] == pid]
        n = len(sub)
        row = {
            'Pattern':        pid,
            'Pattern_name':   PATTERN_NAMES.get(pid, '?'),
            'Attempts':       n,
        }
        if has_completeness:
            row['Partial (%)']          = round(100 * (sub['completeness'] < 1.0).sum() / n, 1)
            row['Mean completeness (%)'] = round(100 * sub['completeness'].mean(), 1)
        row.update({
            'Mean txns':      round(sub['n_txns'].mean(), 1),
            'Median txns':    sub['n_txns'].median(),
            'Mean banks':     round(sub['n_banks'].mean(), 2),
            'Median banks':   sub['n_banks'].median(),
            '1-bank (%)':     round(100 * (sub['n_banks'] == 1).sum() / n, 1),
            '2-bank (%)':     round(100 * (sub['n_banks'] == 2).sum() / n, 1),
            '3-bank (%)':     round(100 * (sub['n_banks'] == 3).sum() / n, 1),
            '4+-bank (%)':    round(100 * (sub['n_banks'] >= 4).sum() / n, 1),
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex')

    return df


def build_attempt_transaction_class_table(test_raw_df, raw_df=None, out_dir='tables',
                                          out_name='attempt_txn_class',
                                          csv_dir=None):
    """Classify each illicit transaction in the test split as WB-single, WB-multi, or CB.

    WB-single : From Bank == To Bank  AND  attempt spans only 1 bank
    WB-multi  : From Bank == To Bank  AND  attempt spans 2+ banks
    CB        : From Bank != To Bank  (always part of a multi-bank attempt)

    Bank span per attempt is computed from raw_df (full dataset) when provided,
    so attempts that partially fall outside the test split are correctly classified.
    Falls back to test_raw_df if raw_df is not given.

    Reports counts and percentages per pattern, showing how many
    within-bank transactions are actually embedded in cross-bank attempt
    structures despite not crossing a bank boundary themselves.

    Requires AttemptID column — regenerate CSV via format_kaggle_files.py if missing.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    if 'AttemptID' not in span_src.columns:
        print("[WARN] AttemptID column not found. "
              "Re-run format_kaggle_files.py to regenerate the CSV.")
        return None

    # Bank span per attempt — computed from full dataset for accuracy
    illicit_all = span_src[(span_src['Is Laundering'] == 1) & (span_src['AttemptID'] >= 0)]
    attempt_n_banks = (
        illicit_all.groupby('AttemptID')
        .apply(lambda g: len(set(g['From Bank'].tolist() + g['To Bank'].tolist())))
        .rename('n_banks')
    )

    # Transactions to classify: test split only
    illicit = test_raw_df[(test_raw_df['Is Laundering'] == 1) & (test_raw_df['AttemptID'] >= 0)].copy()
    illicit['n_attempt_banks'] = illicit['AttemptID'].map(attempt_n_banks)
    illicit['is_cross_bank'] = illicit['From Bank'] != illicit['To Bank']

    def _classify(row):
        if row['is_cross_bank']:
            return 'CB'
        return 'WB-single' if row['n_attempt_banks'] == 1 else 'WB-multi'

    illicit['txn_class'] = illicit.apply(_classify, axis=1)

    rows = []
    for pid in sorted(illicit['Pattern'].unique()):
        sub = illicit[illicit['Pattern'] == pid]
        n = len(sub)
        counts = sub['txn_class'].value_counts()
        rows.append({
            'Pattern':          pid,
            'Pattern_name':     PATTERN_NAMES.get(pid, '?'),
            'Txns':             n,
            'WB-single':        counts.get('WB-single', 0),
            'WB-single (%)':    round(100 * counts.get('WB-single', 0) / n, 1),
            'WB-multi':         counts.get('WB-multi', 0),
            'WB-multi (%)':     round(100 * counts.get('WB-multi', 0) / n, 1),
            'CB':               counts.get('CB', 0),
            'CB (%)':           round(100 * counts.get('CB', 0) / n, 1),
        })

    df = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)
    df_to_latex_table(df, out_dir / f'{out_name}.tex')

    return df


def _compute_attempt_max_visibility(illicit_df, party_banks=None):
    """For each attempt, compute the max fraction of its transactions visible to any single bank.

    A bank 'sees' a transaction if it is either the From Bank or the To Bank.
    party_banks : optional set of bank IDs to restrict the max to (e.g. the 630
        comparable banks). When None, all banks in the attempt are considered.
        Pass the comparable bank set for comparable-eval visibility so that
        non-FL-participating intermediaries cannot inflate max_vis.
    Returns a DataFrame with columns: AttemptID, Pattern, n_txns, max_vis.
    """
    records = []
    for attempt_id, grp in illicit_df.groupby('AttemptID'):
        n = len(grp)
        banks = set(grp['From Bank'].tolist() + grp['To Bank'].tolist())
        if party_banks is not None:
            banks = banks & party_banks
        bank_vis = {
            b: int(((grp['From Bank'] == b) | (grp['To Bank'] == b)).sum())
            for b in banks
        }
        max_v = max(bank_vis.values()) / n if bank_vis else np.nan
        records.append({
            'AttemptID':  attempt_id,
            'Pattern':    int(grp['Pattern'].iloc[0]),
            'n_txns':     n,
            'max_vis':    max_v,
        })
    return pd.DataFrame(records)


def build_attempt_visibility_table(test_raw_df, raw_df=None, out_dir='tables',
                                    out_name='attempt_visibility', csv_dir=None,
                                    party_banks=None):
    """Per-pattern table of max single-bank visibility over attempts in the test split.

    For each attempt, max_vis = (transactions visible to the most-involved bank) / total.
    max_vis = 1.0 means one bank sees the entire attempt and could detect it locally.
    max_vis < 0.5 means no single party holds even a majority of the attempt's transactions.

    party_banks : optional set of bank IDs (e.g. the 630 comparable banks).
        When provided, max_vis is computed only over those banks so that
        non-participating intermediaries cannot inflate the metric.

    Columns: Pattern | Pattern_name | Attempts | Mean max-vis (%) |
             vis=100 (%) | vis>=50 (%) | vis<50 (%)
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]

    vis_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)
    if vis_df.empty:
        print("[WARN] No attempts found for visibility table.")
        return None, None

    rows = []
    for pat in sorted(vis_df['Pattern'].unique()):
        sub = vis_df[vis_df['Pattern'] == pat]
        n = len(sub)
        rows.append({
            'Pattern':      pat,
            'Pattern_name': PATTERN_NAMES.get(pat, f'P{pat}'),
            'Attempts':     n,
            'Mean max-vis (%)': round(100 * sub['max_vis'].mean(), 1),
            'vis=100 (%)':      round(100 * (sub['max_vis'] == 1.0).sum() / n, 1),
            'vis>=50 (%)':      round(100 * (sub['max_vis'] >= 0.5).sum() / n, 1),
            'vis<50 (%)':       round(100 * (sub['max_vis'] < 0.5).sum() / n, 1),
        })

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)

    # Rename operator columns to LaTeX math mode before rendering
    tex_rename = {
        'Mean max-vis (%)': 'Mean $v_{max}$ (\\%)',
        'vis=100 (%)':      'vis$=$100 (\\%)',
        'vis>=50 (%)':      'vis$\\geq$50 (\\%)',
        'vis<50 (%)':       'vis$<$50 (\\%)',
    }
    df_to_latex_table(df.rename(columns=tex_rename), out_dir / f'{out_name}.tex')
    return df, vis_df


def build_attempt_bank_coverage_table(test_raw_df, raw_df=None, out_dir='tables',
                                       out_name='attempt_bank_coverage', csv_dir=None,
                                       party_banks=None):
    """Per-pattern distribution of bank-level visibility across all involved banks.

    For each attempt every involved bank is assigned a visibility score:
        vis = (# attempt transactions where bank is From Bank or To Bank) / total transactions

    Banks are then binned into three tiers:
        Full    (vis=100%): bank sees every transaction in the attempt
        Partial (50%<=vis<100%): bank sees more than half but not all
        Low     (vis<50%): bank sees less than half

    Per pattern the table shows:
      - mean total banks per attempt
      - mean number of banks in each tier per attempt
      - mean % of banks per attempt in each tier

    This captures the FL challenge: even when max_vis=100% (one bank sees everything),
    most participating banks may have very low visibility, limiting how many parties
    can contribute meaningful gradient signal for detecting that pattern type.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]

    records = []
    for attempt_id, grp in illicit_all.groupby('AttemptID'):
        n = len(grp)
        banks = set(grp['From Bank'].tolist() + grp['To Bank'].tolist())
        if party_banks is not None:
            banks = banks & party_banks
        bank_vis = {
            b: ((grp['From Bank'] == b) | (grp['To Bank'] == b)).sum() / n
            for b in banks
        }
        n_banks   = len(banks)
        n_full    = sum(1 for v in bank_vis.values() if v == 1.0)
        n_partial = sum(1 for v in bank_vis.values() if 0.5 <= v < 1.0)
        n_low     = sum(1 for v in bank_vis.values() if v < 0.5)
        records.append({
            'AttemptID': attempt_id,
            'Pattern':   int(grp['Pattern'].iloc[0]),
            'n_banks':   n_banks,
            'n_full':    n_full,
            'n_partial': n_partial,
            'n_low':     n_low,
        })

    cov_df = pd.DataFrame(records)
    if cov_df.empty:
        print("[WARN] No attempts found for bank coverage table.")
        return None

    rows = []
    for pat in sorted(cov_df['Pattern'].unique()):
        sub = cov_df[cov_df['Pattern'] == pat]
        n_att      = len(sub)
        mean_banks = sub['n_banks'].mean()
        rows.append({
            'Pattern':           pat,
            'Pattern_name':      PATTERN_NAMES.get(pat, f'P{pat}'),
            'Attempts':          n_att,
            'Banks/attempt':     round(mean_banks, 1),
            # mean count of banks per attempt in each tier
            'vis=100 (mean)':    round(sub['n_full'].mean(), 2),
            'vis>=50 (mean)':    round(sub['n_partial'].mean(), 2),
            'vis<50 (mean)':     round(sub['n_low'].mean(), 2),
            # mean % of banks per attempt in each tier
            'vis=100 (%)':       round(100 * (sub['n_full'] / sub['n_banks']).mean(), 1),
            'vis>=50 (%)':       round(100 * (sub['n_partial'] / sub['n_banks']).mean(), 1),
            'vis<50 (%)':        round(100 * (sub['n_low'] / sub['n_banks']).mean(), 1),
        })

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)

    # LaTeX: show percentage columns only (counts go in CSV for reference)
    tex_df = df[['Pattern_name', 'Attempts', 'Banks/attempt',
                 'vis=100 (%)', 'vis>=50 (%)', 'vis<50 (%)']].rename(columns={
        'Pattern_name':  'Pattern',
        'Banks/attempt': 'Banks',
        'vis=100 (%)':   'vis$=$100 (\\%)',
        'vis>=50 (%)':   'vis$\\geq$50 (\\%)',
        'vis<50 (%)':    'vis$<$50 (\\%)',
    })
    df_to_latex_table(tex_df, out_dir / f'{out_name}.tex')
    return df


def build_attempt_visibility_recall(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                                     out_dir='tables', out_name='attempt_visibility_recall',
                                     csv_dir=None, detection_threshold='any',
                                     party_banks=None):
    """Recall by visibility bucket, plus an Overall row group.

    Row groups: vis=100 / vis>=50 / vis<50 / Overall (all attempts, no vis filter).
    Within each group: one row per scenario.
    Columns: one per laundering pattern present in the test split, plus Overall.

    detection_threshold controls what counts as a detected attempt:
        'any'  — at least 1 transaction predicted positive  (>=1)
        'half' — at least 50% of transactions predicted positive (>=50%)
        'all'  — all transactions predicted positive  (100%)
        'txn'  — transaction-level TP/P (no attempt grouping; denominator = transactions)
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]
    vis_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)

    def _vis_bucket(v):
        if v == 1.0:   return 'vis=100'
        if v >= 0.5:   return 'vis>=50'
        return 'vis<50'
    vis_df['vis_bucket'] = vis_df['max_vis'].apply(_vis_bucket)
    attempt_bucket = vis_df.set_index('AttemptID')['vis_bucket']

    enriched_test = _add_attempt_class_to_df(test_raw_df, raw_df=raw_df)
    patterns_present = sorted(vis_df['Pattern'].unique())
    # 'Overall' bucket = no visibility filter; placed last
    vis_buckets = ['vis=100', 'vis>=50', 'vis<50', 'Overall']
    pat_col_names = {p: PATTERN_NAMES.get(p, f'P{p}') for p in patterns_present}

    use_txn_level = (detection_threshold == 'txn')

    if detection_threshold == 'half':
        def _is_detected(ag):
            return (ag['n_det'] >= ag['n_txns'] * 0.5).sum() / len(ag)
    elif detection_threshold == 'all':
        def _is_detected(ag):
            return (ag['n_det'] == ag['n_txns']).sum() / len(ag)
    else:  # 'any' or 'txn' (txn bypasses this)
        def _is_detected(ag):
            return (ag['n_det'] >= 1).sum() / len(ag)

    def _rec(sub, vb):
        if vb == 'Overall':
            sub_vb = sub
        else:
            mask = sub['AttemptID'].map(lambda a: attempt_bucket.get(a, None) == vb)
            sub_vb = sub[mask]
        if len(sub_vb) == 0:
            return np.nan
        if use_txn_level:
            return (sub_vb['pred_label'] == 1).sum() / len(sub_vb)
        ag = sub_vb.groupby('AttemptID').agg(
            n_txns=('pred_label', 'count'),
            n_det=('pred_label', lambda x: (x == 1).sum()),
        ).reset_index()
        return _is_detected(ag) if len(ag) else np.nan

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, enriched_test)
            if 'Pattern' not in lv_e.columns:
                continue
            illicit = lv_e[
                (lv_e['true_y'] == 1) &
                lv_e['AttemptID'].notna() &
                (lv_e['AttemptID'] >= 0) &
                (lv_e['Pattern'] >= 1) &
                (lv_e['Pattern'] <= 8)
            ].copy()

            row = {'ID': sid, 'Scenario': scenario_map[sid]['name'], 'seed': seed}
            for vb in vis_buckets:
                for pat in patterns_present:
                    row[f'{vb}_p{pat}'] = _rec(illicit[illicit['Pattern'] == pat], vb)
                row[f'{vb}_overall'] = _rec(illicit, vb)
            rows.append(row)

    raw_rows = pd.DataFrame(rows)
    agg_spec = {}
    for vb in vis_buckets:
        for pat in patterns_present:
            c = f'{vb}_p{pat}'
            agg_spec[c] = (c, 'mean')
        agg_spec[f'{vb}_overall'] = (f'{vb}_overall', 'mean')

    agg = (raw_rows.groupby(['ID', 'Scenario'])
           .agg(**agg_spec)
           .reset_index())
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    agg = agg.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    long_rows = []
    for vb in vis_buckets:
        for _, row in agg.iterrows():
            r = {'Vis bucket': vb, 'ID': row['ID'], 'Scenario': row['Scenario']}
            for pat in patterns_present:
                r[pat_col_names[pat]] = row[f'{vb}_p{pat}']
            r['Overall'] = row[f'{vb}_overall']
            long_rows.append(r)

    pivot = pd.DataFrame(long_rows)
    data_cols = [pat_col_names[p] for p in patterns_present] + ['Overall']
    for col in data_cols:
        pivot[col] = pivot[col].round(3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path / f'{out_name}.csv', index=False)

    n_scen = len(scenario_ids)
    col_format = 'll' + 'c' * len(data_cols)

    def _tex(s):
        return (s.replace('%', r'\%')
                 .replace('>=', r'$\geq$')
                 .replace('<=', r'$\leq$')
                 .replace('<', r'$<$')
                 .replace('>', r'$>$'))

    header = 'Vis. bucket & Scenario & ' + ' & '.join(_tex(c) for c in data_cols) + r' \\'
    body_lines = []
    for b_idx, vb in enumerate(vis_buckets):
        sub = pivot[pivot['Vis bucket'] == vb]
        for s_idx, (_, row) in enumerate(sub.iterrows()):
            vb_cell = (rf'\multirow{{{n_scen}}}{{*}}{{{_tex(vb)}}}' if s_idx == 0 else '')
            cells = [vb_cell, str(row['Scenario'])]
            for col in data_cols:
                val = row[col]
                cells.append(_fmt_scaled(val, scale=100, decimals=2, auto_widen=True) if pd.notna(val) else '--')
            body_lines.append(' & '.join(cells) + r' \\')
        if b_idx < len(vis_buckets) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return pivot, agg


def build_attempt_visibility_recall_combined(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                                              out_dir='tables',
                                              out_name='attempt_visibility_recall_combined',
                                              csv_dir=None, party_banks=None):
    """All four detection thresholds in one LaTeX table, stratified by visibility bucket.

    Row structure: Threshold (outer) → Vis. bucket (inner) → Scenario.
    Columns: one per laundering pattern present in the test split, plus Overall.

    Thresholds: >=1 (attempt), >=50% (attempt), 100% (attempt), Individual (txn-level TP/P).
    Vis. buckets: vis=100 / vis>=50 / vis<50 / Overall.

    All four thresholds are computed in one pass over the experiment data.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]
    vis_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)

    def _vis_bucket(v):
        if v == 1.0:   return 'vis=100'
        if v >= 0.5:   return 'vis>=50'
        return 'vis<50'
    vis_df['vis_bucket'] = vis_df['max_vis'].apply(_vis_bucket)
    attempt_bucket = vis_df.set_index('AttemptID')['vis_bucket']

    enriched_test = _add_attempt_class_to_df(test_raw_df, raw_df=raw_df)
    patterns_present = sorted(vis_df['Pattern'].unique())
    vis_buckets = ['vis=100', 'vis>=50', 'vis<50', 'Overall']
    pat_col_names = {p: PATTERN_NAMES.get(p, f'P{p}') for p in patterns_present}
    thresh_configs = [('>=1', 'any'), ('>=50%', 'half'), ('100%', 'all'), ('Individual', 'txn')]

    def _recall_for_thresh(sub_vb, thresh_key):
        if len(sub_vb) == 0:
            return np.nan
        if thresh_key == 'txn':
            return (sub_vb['pred_label'] == 1).sum() / len(sub_vb)
        ag = sub_vb.groupby('AttemptID').agg(
            n_txns=('pred_label', 'count'),
            n_det=('pred_label', lambda x: (x == 1).sum()),
        ).reset_index()
        if len(ag) == 0:
            return np.nan
        if thresh_key == 'half':
            return (ag['n_det'] >= ag['n_txns'] * 0.5).sum() / len(ag)
        if thresh_key == 'all':
            return (ag['n_det'] == ag['n_txns']).sum() / len(ag)
        return (ag['n_det'] >= 1).sum() / len(ag)  # 'any'

    def _filter_vb(sub, vb):
        if vb == 'Overall':
            return sub
        mask = sub['AttemptID'].map(lambda a: attempt_bucket.get(a, None) == vb)
        return sub[mask]

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, enriched_test)
            if 'Pattern' not in lv_e.columns:
                continue
            illicit = lv_e[
                (lv_e['true_y'] == 1) &
                lv_e['AttemptID'].notna() &
                (lv_e['AttemptID'] >= 0) &
                (lv_e['Pattern'] >= 1) &
                (lv_e['Pattern'] <= 8)
            ].copy()

            row = {'ID': sid, 'Scenario': scenario_map[sid]['name'], 'seed': seed}
            for thresh_label, thresh_key in thresh_configs:
                for vb in vis_buckets:
                    sub_vb = _filter_vb(illicit, vb)
                    for pat in patterns_present:
                        key = f'{thresh_key}__{vb}__p{pat}'
                        row[key] = _recall_for_thresh(sub_vb[sub_vb['Pattern'] == pat], thresh_key)
                    row[f'{thresh_key}__{vb}__overall'] = _recall_for_thresh(sub_vb, thresh_key)
            rows.append(row)

    raw_rows = pd.DataFrame(rows)
    value_cols = [c for c in raw_rows.columns if c not in ('ID', 'Scenario', 'seed')]
    agg_spec = {c: (c, 'mean') for c in value_cols}
    agg = (raw_rows.groupby(['ID', 'Scenario']).agg(**agg_spec).reset_index())
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    agg = agg.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    # Build long pivot: Threshold, Vis bucket, Scenario, pattern cols, Overall
    data_cols = [pat_col_names[p] for p in patterns_present] + ['Overall']
    long_rows = []
    for thresh_label, thresh_key in thresh_configs:
        for vb in vis_buckets:
            for _, row in agg.iterrows():
                r = {'Threshold': thresh_label, 'Vis bucket': vb,
                     'ID': row['ID'], 'Scenario': row['Scenario']}
                for pat in patterns_present:
                    r[pat_col_names[pat]] = row[f'{thresh_key}__{vb}__p{pat}']
                r['Overall'] = row[f'{thresh_key}__{vb}__overall']
                long_rows.append(r)

    pivot = pd.DataFrame(long_rows)
    for col in data_cols:
        pivot[col] = pivot[col].round(3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path / f'{out_name}.csv', index=False)

    # LaTeX: threshold outer (\multirow n_vb*n_scen), vis bucket inner (\multirow n_scen)
    n_scen = len(scenario_ids)
    n_vb   = len(vis_buckets)
    col_format = 'lll' + 'c' * len(data_cols)

    def _tex(s):
        return (s.replace('%', r'\%')
                 .replace('>=', r'$\geq$')
                 .replace('<=', r'$\leq$')
                 .replace('<', r'$<$')
                 .replace('>', r'$>$'))

    header = ('Threshold & Vis. bucket & Scenario & '
              + ' & '.join(_tex(c) for c in data_cols) + r' \\')
    last_col_idx = 3 + len(data_cols)

    body_lines = []
    for t_idx, (thresh_label, thresh_key) in enumerate(thresh_configs):
        for v_idx, vb in enumerate(vis_buckets):
            sub = pivot[(pivot['Threshold'] == thresh_label) & (pivot['Vis bucket'] == vb)]
            for s_idx, (_, row) in enumerate(sub.iterrows()):
                thresh_cell = (rf'\multirow{{{n_vb * n_scen}}}{{*}}{{{_tex(thresh_label)}}}'
                               if v_idx == 0 and s_idx == 0 else '')
                vb_cell = (rf'\multirow{{{n_scen}}}{{*}}{{{_tex(vb)}}}'
                           if s_idx == 0 else '')
                cells = [thresh_cell, vb_cell, str(row['Scenario'])]
                for col in data_cols:
                    val = row[col]
                    cells.append(f'{val:.3f}' if pd.notna(val) else '--')
                body_lines.append(' & '.join(cells) + r' \\')
            if v_idx < n_vb - 1:
                body_lines.append(rf'\cmidrule{{2-{last_col_idx}}}')
        if t_idx < len(thresh_configs) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return pivot, agg


def build_attempt_size_recall(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                               out_dir='tables', out_name='attempt_size_recall',
                               csv_dir=None, detection_threshold='any',
                               party_banks=None, patterns=None):
    """Recall by attempt-size bucket (size=1 vs size>=2), plus an Overall row group.

    Diagnostic companion to build_attempt_visibility_recall: for patterns with no
    structural degree floor (Stack, Random, Bipartite), a size=1 attempt is a single
    transaction seen by whichever one bank is party to it, so it is *always* vis=100
    by construction. If the vis=100 bucket in the visibility-recall table is showing
    unexpectedly low recall, this checks whether that is really a visibility effect
    or just the size=1 singletons — which carry the least relational structure for a
    message-passing model — dragging that bucket down.

    Row groups: size=1 / size>=2 / Overall.
    Within each group: one row per scenario.
    Columns: one per pattern in `patterns` (default: patterns present in the data),
    plus Overall.

    detection_threshold: same semantics as build_attempt_visibility_recall
    ('any' / 'half' / 'all' / 'txn').
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]
    size_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)  # reuses n_txns per attempt

    if patterns is not None:
        size_df = size_df[size_df['Pattern'].isin(patterns)]

    size_df['size_bucket'] = np.where(size_df['n_txns'] <= 1, 'size=1', 'size>=2')
    attempt_bucket = size_df.set_index('AttemptID')['size_bucket']

    patterns_present = sorted(size_df['Pattern'].unique())
    size_buckets = ['size=1', 'size>=2', 'Overall']
    pat_col_names = {p: PATTERN_NAMES.get(p, f'P{p}') for p in patterns_present}

    use_txn_level = (detection_threshold == 'txn')

    if detection_threshold == 'half':
        def _is_detected(ag):
            return (ag['n_det'] >= ag['n_txns'] * 0.5).sum() / len(ag)
    elif detection_threshold == 'all':
        def _is_detected(ag):
            return (ag['n_det'] == ag['n_txns']).sum() / len(ag)
    else:  # 'any' or 'txn' (txn bypasses this)
        def _is_detected(ag):
            return (ag['n_det'] >= 1).sum() / len(ag)

    def _rec(sub, sb):
        if sb == 'Overall':
            sub_sb = sub
        else:
            mask = sub['AttemptID'].map(lambda a: attempt_bucket.get(a, None) == sb)
            sub_sb = sub[mask]
        if len(sub_sb) == 0:
            return np.nan
        if use_txn_level:
            return (sub_sb['pred_label'] == 1).sum() / len(sub_sb)
        ag = sub_sb.groupby('AttemptID').agg(
            n_txns=('pred_label', 'count'),
            n_det=('pred_label', lambda x: (x == 1).sum()),
        ).reset_index()
        return _is_detected(ag) if len(ag) else np.nan

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, test_raw_df)
            if 'Pattern' not in lv_e.columns:
                continue
            illicit = lv_e[
                (lv_e['true_y'] == 1) &
                lv_e['AttemptID'].notna() &
                (lv_e['AttemptID'] >= 0) &
                (lv_e['Pattern'] >= 1) &
                (lv_e['Pattern'] <= 8)
            ].copy()
            if patterns is not None:
                illicit = illicit[illicit['Pattern'].isin(patterns)]

            row = {'ID': sid, 'Scenario': scenario_map[sid]['name'], 'seed': seed}
            for sb in size_buckets:
                for pat in patterns_present:
                    row[f'{sb}_p{pat}'] = _rec(illicit[illicit['Pattern'] == pat], sb)
                row[f'{sb}_overall'] = _rec(illicit, sb)
            rows.append(row)

    raw_rows = pd.DataFrame(rows)
    agg_spec = {}
    for sb in size_buckets:
        for pat in patterns_present:
            c = f'{sb}_p{pat}'
            agg_spec[c] = (c, 'mean')
        agg_spec[f'{sb}_overall'] = (f'{sb}_overall', 'mean')

    agg = (raw_rows.groupby(['ID', 'Scenario'])
           .agg(**agg_spec)
           .reset_index())
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    agg = agg.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    long_rows = []
    for sb in size_buckets:
        for _, row in agg.iterrows():
            r = {'Size bucket': sb, 'ID': row['ID'], 'Scenario': row['Scenario']}
            for pat in patterns_present:
                r[pat_col_names[pat]] = row[f'{sb}_p{pat}']
            r['Overall'] = row[f'{sb}_overall']
            long_rows.append(r)

    pivot = pd.DataFrame(long_rows)
    data_cols = [pat_col_names[p] for p in patterns_present] + ['Overall']
    for col in data_cols:
        pivot[col] = pivot[col].round(3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path / f'{out_name}.csv', index=False)

    n_scen = len(scenario_ids)
    col_format = 'll' + 'c' * len(data_cols)

    def _tex(s):
        return (s.replace('%', r'\%')
                 .replace('>=', r'$\geq$')
                 .replace('<=', r'$\leq$')
                 .replace('<', r'$<$')
                 .replace('>', r'$>$'))

    header = 'Size bucket & Scenario & ' + ' & '.join(_tex(c) for c in data_cols) + r' \\'
    body_lines = []
    for b_idx, sb in enumerate(size_buckets):
        sub = pivot[pivot['Size bucket'] == sb]
        for s_idx, (_, row) in enumerate(sub.iterrows()):
            sb_cell = (rf'\multirow{{{n_scen}}}{{*}}{{{_tex(sb)}}}' if s_idx == 0 else '')
            cells = [sb_cell, str(row['Scenario'])]
            for col in data_cols:
                val = row[col]
                cells.append(_fmt_scaled(val, scale=100, decimals=2, auto_widen=True) if pd.notna(val) else '--')
            body_lines.append(' & '.join(cells) + r' \\')
        if b_idx < len(size_buckets) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return pivot, agg


def build_attempt_size_visibility_crosstab(test_raw_df, raw_df=None, out_dir='tables',
                                            out_name='attempt_size_visibility_crosstab',
                                            csv_dir=None, party_banks=None, patterns=None):
    """Counts attempts by (pattern, size bucket, visibility bucket) to show directly
    how collinear attempt size and single-bank visibility are for patterns with no
    structural degree floor. A size=1 attempt is always vis=100 by construction, so
    this table is expected to show size=1 attempts concentrated entirely in the
    vis=100 column, and size>=2 attempts spread across the lower buckets.

    Columns: vis=100 | vis>=50 | vis<50, plus a '% of vis=100 that are size=1' summary.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]
    vis_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)

    if patterns is not None:
        vis_df = vis_df[vis_df['Pattern'].isin(patterns)]
    if vis_df.empty:
        print("[WARN] No attempts found for size/visibility crosstab.")
        return None

    def _vis_bucket(v):
        if v == 1.0:   return 'vis=100'
        if v >= 0.5:   return 'vis>=50'
        return 'vis<50'
    vis_df['vis_bucket']  = vis_df['max_vis'].apply(_vis_bucket)
    vis_df['size_bucket'] = np.where(vis_df['n_txns'] <= 1, 'size=1', 'size>=2')

    rows = []
    for pat in sorted(vis_df['Pattern'].unique()):
        sub = vis_df[vis_df['Pattern'] == pat]
        counts = sub.groupby(['size_bucket', 'vis_bucket']).size()
        n_vis100 = int(counts.get(('size=1', 'vis=100'), 0) + counts.get(('size>=2', 'vis=100'), 0))
        n_vis100_size1 = int(counts.get(('size=1', 'vis=100'), 0))
        rows.append({
            'Pattern':      pat,
            'Pattern_name': PATTERN_NAMES.get(pat, f'P{pat}'),
            'size=1, vis=100':  int(counts.get(('size=1', 'vis=100'), 0)),
            'size=1, vis>=50':  int(counts.get(('size=1', 'vis>=50'), 0)),
            'size=1, vis<50':   int(counts.get(('size=1', 'vis<50'), 0)),
            'size>=2, vis=100': int(counts.get(('size>=2', 'vis=100'), 0)),
            'size>=2, vis>=50': int(counts.get(('size>=2', 'vis>=50'), 0)),
            'size>=2, vis<50':  int(counts.get(('size>=2', 'vis<50'), 0)),
            'vis=100 that are size=1 (%)': round(100 * n_vis100_size1 / n_vis100, 1) if n_vis100 else float('nan'),
        })

    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / f'{out_name}.csv', index=False)

    tex_df = df[['Pattern_name', 'size=1, vis=100', 'size=1, vis>=50', 'size=1, vis<50',
                 'size>=2, vis=100', 'size>=2, vis>=50', 'size>=2, vis<50',
                 'vis=100 that are size=1 (%)']].rename(columns={
        'Pattern_name': 'Pattern',
        'size=1, vis=100':  'size$=$1, vis$=$100',
        'size=1, vis>=50':  'size$=$1, vis$\\geq$50',
        'size=1, vis<50':   'size$=$1, vis$<$50',
        'size>=2, vis=100': 'size$\\geq$2, vis$=$100',
        'size>=2, vis>=50': 'size$\\geq$2, vis$\\geq$50',
        'size>=2, vis<50':  'size$\\geq$2, vis$<$50',
        'vis=100 that are size=1 (%)': 'vis$=$100 that are size$=$1 (\\%)',
    })
    df_to_latex_table(tex_df, out_dir / f'{out_name}.tex')
    return df


def build_attempt_visibility_txn_recall(scenario_map, scenario_ids, test_raw_df, raw_df=None,
                                         out_dir='tables', out_name='attempt_visibility_txn_recall',
                                         csv_dir=None, party_banks=None):
    """Transaction-level recall pooled within each max-visibility bucket.

    Unlike attempt-level recall, this pools all individual illicit transactions that
    belong to attempts in a given visibility tier and computes TP / P (fraction of
    those transactions predicted positive).

    Row groups: vis=100 / vis>=50 / vis<50 / Overall.
    Within each group: one row per scenario.
    Columns: one per laundering pattern present in the test split, plus Overall.
    """
    span_src = raw_df if raw_df is not None else test_raw_df
    illicit_all = span_src[
        (span_src['Is Laundering'] == 1) &
        (span_src['AttemptID'] >= 0) &
        (span_src['Pattern'] >= 1) &
        (span_src['Pattern'] <= 8)
    ].copy()
    test_attempt_ids = set(
        test_raw_df.loc[
            (test_raw_df['Is Laundering'] == 1) &
            (test_raw_df['AttemptID'] >= 0) &
            (test_raw_df['Pattern'] >= 1) &
            (test_raw_df['Pattern'] <= 8),
            'AttemptID'
        ].unique()
    )
    illicit_all = illicit_all[illicit_all['AttemptID'].isin(test_attempt_ids)]
    vis_df = _compute_attempt_max_visibility(illicit_all, party_banks=party_banks)

    def _vis_bucket(v):
        if v == 1.0:   return 'vis=100'
        if v >= 0.5:   return 'vis>=50'
        return 'vis<50'
    vis_df['vis_bucket'] = vis_df['max_vis'].apply(_vis_bucket)
    attempt_bucket = vis_df.set_index('AttemptID')['vis_bucket']

    enriched_test = _add_attempt_class_to_df(test_raw_df, raw_df=raw_df)
    patterns_present = sorted(vis_df['Pattern'].unique())
    vis_buckets = ['vis=100', 'vis>=50', 'vis<50', 'Overall']
    pat_col_names = {p: PATTERN_NAMES.get(p, f'P{p}') for p in patterns_present}

    def _txn_recall(sub, vb):
        if vb == 'Overall':
            sub_vb = sub
        else:
            mask = sub['AttemptID'].map(lambda a: attempt_bucket.get(a, None) == vb)
            sub_vb = sub[mask]
        if len(sub_vb) == 0:
            return np.nan
        return (sub_vb['pred_label'] == 1).sum() / len(sub_vb)

    rows = []
    for sid in scenario_ids:
        exp = load_experiment(scenario_map[sid]['path'])
        for seed, seed_data in exp.seed_results.items():
            lv = seed_data.get('laundering_values')
            if lv is None:
                continue
            lv_e = enrich_lv_with_raw(lv, enriched_test)
            if 'Pattern' not in lv_e.columns:
                continue
            illicit = lv_e[
                (lv_e['true_y'] == 1) &
                lv_e['AttemptID'].notna() &
                (lv_e['AttemptID'] >= 0) &
                (lv_e['Pattern'] >= 1) &
                (lv_e['Pattern'] <= 8)
            ].copy()

            row = {'ID': sid, 'Scenario': scenario_map[sid]['name'], 'seed': seed}
            for vb in vis_buckets:
                for pat in patterns_present:
                    row[f'{vb}_p{pat}'] = _txn_recall(illicit[illicit['Pattern'] == pat], vb)
                row[f'{vb}_overall'] = _txn_recall(illicit, vb)
            rows.append(row)

    raw_rows = pd.DataFrame(rows)
    agg_spec = {}
    for vb in vis_buckets:
        for pat in patterns_present:
            c = f'{vb}_p{pat}'
            agg_spec[c] = (c, 'mean')
        agg_spec[f'{vb}_overall'] = (f'{vb}_overall', 'mean')

    agg = (raw_rows.groupby(['ID', 'Scenario'])
           .agg(**agg_spec)
           .reset_index())
    id_order = {sid: i for i, sid in enumerate(scenario_ids)}
    agg = agg.sort_values('ID', key=lambda s: s.map(id_order)).reset_index(drop=True)

    long_rows = []
    for vb in vis_buckets:
        for _, row in agg.iterrows():
            r = {'Vis bucket': vb, 'ID': row['ID'], 'Scenario': row['Scenario']}
            for pat in patterns_present:
                r[pat_col_names[pat]] = row[f'{vb}_p{pat}']
            r['Overall'] = row[f'{vb}_overall']
            long_rows.append(r)

    pivot = pd.DataFrame(long_rows)
    data_cols = [pat_col_names[p] for p in patterns_present] + ['Overall']
    for col in data_cols:
        pivot[col] = pivot[col].round(3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path / f'{out_name}.csv', index=False)

    n_scen = len(scenario_ids)
    col_format = 'll' + 'c' * len(data_cols)

    def _tex(s):
        return (s.replace('%', r'\%')
                 .replace('>=', r'$\geq$')
                 .replace('<=', r'$\leq$')
                 .replace('<', r'$<$')
                 .replace('>', r'$>$'))

    header = 'Vis. bucket & Scenario & ' + ' & '.join(_tex(c) for c in data_cols) + r' \\'
    body_lines = []
    for b_idx, vb in enumerate(vis_buckets):
        sub = pivot[pivot['Vis bucket'] == vb]
        for s_idx, (_, row) in enumerate(sub.iterrows()):
            vb_cell = (rf'\multirow{{{n_scen}}}{{*}}{{{_tex(vb)}}}' if s_idx == 0 else '')
            cells = [vb_cell, str(row['Scenario'])]
            for col in data_cols:
                val = row[col]
                cells.append(_fmt_scaled(val, scale=100, decimals=2, auto_widen=True) if pd.notna(val) else '--')
            body_lines.append(' & '.join(cells) + r' \\')
        if b_idx < len(vis_buckets) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return pivot, agg


def build_party_presence_table(raw_df, size='small', ir='HI', eval_mode='comparable',
                                split_perc=(0.6, 0.2), out_dir='tables',
                                out_name='party_presence', csv_dir=None):
    """Classify transactions by whether both/one/no party bank is present per split.

    For each split (train / vali / test) reports, for all transactions and illicit
    transactions separately, how many have both banks as participating parties, only
    the From Bank, only the To Bank, or neither.

    Comparable: uses the 'individual' bank set — the same 630 banks for all splits.
    System:     uses FedAvg bank sets — train/vali/test banks differ.

    This quantifies the structural handicap for FL:
      - Both present  — full bilateral signal (SplitFed sees both embeddings)
      - One present   — half-informed for SplitFed (one embedding stays zero);
                        unilateral for horizontal FL (only one bank trains on it)
      - Neither       — transaction fully outside the party set (only full_info sees it)
    """
    import json, itertools
    import torch
    from configs.paths import get_data_path

    # Load relevant_banks JSON (same path logic as reconstruct_test_raw_df)
    rb_path = (Path(get_data_path()) /
               f"AML_work_study/experiments/relevant_banks"
               f"/{size}_{ir}__split_{split_perc[0]}_{split_perc[1]}.json")
    if not rb_path.exists():
        print(f"[WARN] relevant_banks JSON not found: {rb_path}")
        return None
    with open(rb_path) as f:
        rb = json.load(f)

    # Bank sets per split
    if eval_mode == 'comparable':
        individual_set = set(rb['individual']['banks'])
        split_banks = {'train': individual_set, 'vali': individual_set, 'test': individual_set}
    else:  # system — FedAvg has split-specific bank sets
        split_banks = {
            'train': set(rb['FedAvg']['train_banks']),
            'vali':  set(rb['FedAvg']['vali_banks']),
            'test':  set(rb['FedAvg']['test_banks']),
        }

    # Replicate the daily-boundary temporal split (same logic as reconstruct_test_raw_df)
    timestamps = torch.tensor(raw_df['Timestamp'].to_numpy(), dtype=torch.float32)
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_inds = []
    for day in range(n_days):
        l, r = day * 24 * 3600, (day + 1) * 24 * 3600
        daily_inds.append(torch.where((timestamps >= l) & (timestamps < r))[0])
    d_ts = np.array([t.shape[0] for t in daily_inds])
    I = list(range(len(d_ts)))
    test_perc = round(1 - sum(split_perc), 10)
    split_perc_full = list(split_perc) + [test_perc]
    split_scores = {}
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            s = sum(totals)
            props = [v / s for v in totals]
            split_scores[(i, j)] = max(abs(v - t) / t for v, t in zip(props, split_perc_full))
    i_star, j_star = min(split_scores, key=split_scores.get)
    split_day_groups = [list(range(i_star)), list(range(i_star, j_star)), list(range(j_star, len(d_ts)))]
    all_positions = [np.concatenate([daily_inds[d].numpy() for d in days])
                     for days in split_day_groups]

    if eval_mode == 'comparable':
        individual_idx = set(rb['individual']['indices'])
        all_positions = [pos[np.isin(pos, list(individual_idx))] for pos in all_positions]

    split_dfs = {
        'train': raw_df.iloc[all_positions[0]],
        'vali':  raw_df.iloc[all_positions[1]],
        'test':  raw_df.iloc[all_positions[2]],
    }

    def _classify(df_s, banks):
        from_p = df_s['From Bank'].isin(banks)
        to_p   = df_s['To Bank'].isin(banks)
        n = len(df_s)
        return {
            'total':     n,
            'both':      int((from_p & to_p).sum()),
            'from_only': int((from_p & ~to_p).sum()),
            'to_only':   int((~from_p & to_p).sum()),
            'neither':   int((~from_p & ~to_p).sum()),
        }

    rows = []
    for sname, df_s in split_dfs.items():
        banks = split_banks[sname]
        for label, subset in [('All', df_s), ('Illicit', df_s[df_s['Is Laundering'] == 1])]:
            c = _classify(subset, banks)
            n = c['total']
            rows.append({
                'Split':         sname,
                'Type':          label,
                'Total':         n,
                'Both (n)':      c['both'],
                'Both (%)':      round(100 * c['both'] / n, 1) if n else 0.0,
                'From-only (n)': c['from_only'],
                'From-only (%)': round(100 * c['from_only'] / n, 1) if n else 0.0,
                'To-only (n)':   c['to_only'],
                'To-only (%)':   round(100 * c['to_only'] / n, 1) if n else 0.0,
                'Neither (n)':   c['neither'],
                'Neither (%)':   round(100 * c['neither'] / n, 1) if n else 0.0,
            })
    df_out = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path / f'{out_name}.csv', index=False)

    # LaTeX: Type (All / Illicit) as outer row group, splits within
    pct_cols = ['Both (%)', 'From-only (%)', 'To-only (%)', 'Neither (%)']

    def _tex(s):
        return s.replace('%', r'\%')

    n_splits = 3
    types = ['All', 'Illicit']
    col_format = 'll' + 'r' * len(pct_cols)
    header = 'Type & Split & ' + ' & '.join(_tex(c) for c in pct_cols) + r' \\'

    body_lines = []
    for t_idx, ttype in enumerate(types):
        sub = df_out[df_out['Type'] == ttype]
        for s_idx, (_, row) in enumerate(sub.iterrows()):
            type_cell = (rf'\multirow{{{n_splits}}}{{*}}{{{ttype}}}' if s_idx == 0 else '')
            cells = [type_cell, row['Split'].capitalize()]
            for col in pct_cols:
                cells.append(str(row[col]))
            body_lines.append(' & '.join(cells) + r' \\')
        if t_idx < len(types) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return df_out


def build_party_presence_combined_table(raw_df, size='small', ir='HI',
                                         split_perc=(0.6, 0.2),
                                         out_dir='tables',
                                         out_name='party_presence_combined',
                                         csv_dir=None):
    """Combined comparable + system party presence table with Mode as outer grouping.

    Calls build_party_presence_table for both modes (also writing the individual
    tables), then merges the results into a single LaTeX table with Mode →
    Type → Split as the three-level row hierarchy.
    """
    out_dir = Path(out_dir)
    comp_df = build_party_presence_table(
        raw_df, size=size, ir=ir, eval_mode='comparable',
        split_perc=split_perc, out_dir=out_dir,
        out_name='party_presence_comparable', csv_dir=csv_dir,
    )
    sys_df = build_party_presence_table(
        raw_df, size=size, ir=ir, eval_mode='system',
        split_perc=split_perc, out_dir=out_dir,
        out_name='party_presence_system', csv_dir=csv_dir,
    )
    if comp_df is None or sys_df is None:
        return None

    comp_df = comp_df.copy()
    sys_df  = sys_df.copy()
    comp_df['Mode'] = 'Comparable'
    sys_df['Mode']  = 'System'
    combined = pd.concat([comp_df, sys_df], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir) if csv_dir else out_dir
    csv_path.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path / f'{out_name}.csv', index=False)

    pct_cols = ['Both (%)', 'From-only (%)', 'To-only (%)', 'Neither (%)']

    def _tex(s):
        return s.replace('%', r'\%')

    modes      = ['Comparable', 'System']
    types      = ['All', 'Illicit']
    n_type_rows = 3   # train / vali / test
    n_mode_rows = 6   # 2 types × 3 splits

    col_format = 'lll' + 'r' * len(pct_cols)
    header = 'Mode & Type & Split & ' + ' & '.join(_tex(c) for c in pct_cols) + r' \\'
    mid = str(3 + len(pct_cols))

    body_lines = []
    for m_idx, mode in enumerate(modes):
        mode_sub = combined[combined['Mode'] == mode]
        for t_idx, ttype in enumerate(types):
            type_sub = mode_sub[mode_sub['Type'] == ttype]
            for s_idx, (_, row) in enumerate(type_sub.iterrows()):
                mode_cell = (rf'\multirow{{{n_mode_rows}}}{{*}}{{{mode}}}'
                             if t_idx == 0 and s_idx == 0 else '')
                type_cell = (rf'\multirow{{{n_type_rows}}}{{*}}{{{ttype}}}'
                             if s_idx == 0 else '')
                cells = [mode_cell, type_cell, row['Split'].capitalize()]
                for col in pct_cols:
                    cells.append(str(row[col]))
                body_lines.append(' & '.join(cells) + r' \\')
            if t_idx < len(types) - 1:
                body_lines.append(rf'\cmidrule{{2-{mid}}}')
        if m_idx < len(modes) - 1:
            body_lines.append(r'\midrule')

    latex_lines = [
        rf'\begin{{tabular}}{{{col_format}}}',
        r'\toprule',
        header,
        r'\midrule',
    ] + body_lines + [
        r'\bottomrule',
        r'\end{tabular}',
    ]
    (out_dir / f'{out_name}.tex').write_text('\n'.join(latex_lines))
    return combined


def comparable_split_stats(split_df, split_name=''):
    """Transaction, account, and bank counts for a single comparable-filtered split.

    Returns a dict suitable for building a summary table or printing inline.
    Accounts = unique From Account ∪ To Account.
    Banks    = unique From Bank ∪ To Bank (includes non-630 counterparties).
    """
    n_txns = len(split_df)
    n_illicit = int((split_df['Is Laundering'] == 1).sum())
    from_acc = set(split_df['from_id'].dropna().unique())
    to_acc   = set(split_df['to_id'].dropna().unique())
    from_bnk = set(split_df['From Bank'].dropna().unique())
    to_bnk   = set(split_df['To Bank'].dropna().unique())
    return {
        'Split':         split_name,
        'Transactions':  n_txns,
        'Illicit (N)':   n_illicit,
        'Illicit (%)':   round(100 * n_illicit / n_txns, 3) if n_txns else 0.0,
        'Accounts (N)':  len(from_acc | to_acc),
        'Banks (N)':     len(from_bnk | to_bnk),
    }
    return df_out
