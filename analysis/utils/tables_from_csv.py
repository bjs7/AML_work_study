"""
Regenerate LaTeX tables from downloaded HPC CSVs — no HPC re-run needed.

Workflow:
  1. Run batch_analysis_hpc.py on the cluster → produces hpc_output/*.csv
  2. Run download_results.sh to sync hpc_output/ locally
  3. Set INPUT_DIR below to wherever hpc_output/ landed, then run this script

Column order can be freely changed by editing _COLS_A / _COLS_B / _PARTY_COLS
below.  Available column names are printed when the script runs.
"""

import os
import pandas as pd

# ==============================================================
# ======================== PATHS ===============================
# ==============================================================

# Where the downloaded hpc_output/ (or tables/party_visibility/) CSVs live.
INPUT_DIR = os.path.expanduser(
    '~/projects/AML_work_study/AML_work_study/scripts/hpc/analysis_hpc/hpc_output'
)

# Where to write the .tex files (default: same as notebook output).
OUTPUT_DIR = os.path.expanduser(
    '~/projects/AML_work_study/writing/Experimental-Protocol/tables/party_visibility'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# ============= COLUMN SELECTION — edit to rearrange ===========
# ==============================================================

# Table 1a: main coverage + neighbourhood size metrics.
# Must be a subset of the columns in cone_coverage_combined_a.csv.
_COLS_A = [
    'cone_from_cov',
    'cone_to_cov',
    'cone_union_cov',
    'cone_neither_frac',
    'cone_asymmetry',
    'cone_n_nodes',
    'cone_n_edges',
    'neigh_n_nodes',
    'neigh_n_edges',
]

# Table 1b: overlap / laundering enrichment metrics.
# Must be a subset of the columns in cone_coverage_combined_b.csv.
_COLS_B = [
    'cone_overlap_cov',
    'cone_unique_from',
    'cone_unique_to',
    'cone_nesting_idx',
    'cone_laund_frac',
    'cone_laund_frac_laund',
    'cone_laund_enrichment',
]

# Table 2: party-level batch coverage.
# Must be a subset of the columns in party_batch_coverage_summary.csv.
_PARTY_COLS = [
    'party_edge_fraction',
    'party_node_fraction',
    'party_seed_coverage',
    'n_party_edges',
    'n_party_nodes',
    'n_party_seed_edges',
]

# Which summary rows to include (subset of whatever the CSV has, e.g. mean/std).
_STATS = ['mean', 'std']


# ==============================================================
# ======================== LOAD CSVs ==========================
# ==============================================================

def _load_combined(fname):
    """Load a combined (multi-index group/stat) CSV back into group → DataFrame."""
    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path, index_col=[0, 1])
    groups = {}
    for grp in df.index.get_level_values(0).unique():
        sub = df.loc[grp]
        groups[grp] = sub.loc[[s for s in _STATS if s in sub.index]]
    return groups


def _load_simple(fname):
    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path, index_col=0)
    return df.loc[[s for s in _STATS if s in df.index]]


print(f"Loading CSVs from: {INPUT_DIR}")

groups_a = _load_combined('cone_coverage_combined_a.csv')
groups_b = _load_combined('cone_coverage_combined_b.csv')
party_df = _load_simple('party_batch_coverage_summary.csv')

# Print available columns so you know what you can put in _COLS_*.
first_a = next(iter(groups_a.values()))
first_b = next(iter(groups_b.values()))
print(f"\nAvailable columns — Table 1a:  {list(first_a.columns)}")
print(f"Available columns — Table 1b:  {list(first_b.columns)}")
print(f"Available columns — Table 2:   {list(party_df.columns)}")


# ==============================================================
# ===================== LaTeX helpers ==========================
# ==============================================================

def _grouped_latex(groups, cols):
    """Row-group LaTeX table (pattern_analysis style)."""
    labels = [c.replace('_', r'\_') for c in cols]
    n = len(cols)
    out = [r'\begin{tabular}{l' + 'r' * n + '}', r'\hline',
           '  & ' + ' & '.join(labels) + r' \\', r'\hline']
    for grp, df in groups.items():
        out.append(r'  \textit{' + grp + '} & ' + ' & '.join([''] * n) + r' \\')
        for stat in df.index:
            vals = []
            for c in cols:
                if c in df.columns:
                    v = df.loc[stat, c]
                    vals.append(f'{v:.4f}' if not pd.isna(v) else '---')
                else:
                    vals.append('---')
            out.append(r'  \quad ' + stat + ' & ' + ' & '.join(vals) + r' \\')
        out.append(r'  \hline')
    out.append(r'\end{tabular}')
    return '\n'.join(out)


def _simple_latex(df, cols):
    """Plain LaTeX table (no row groups) for Table 2."""
    cols_esc = [c.replace('_', r'\_') for c in cols]
    n = len(cols)
    out = [r'\begin{tabular}{l' + 'r' * n + '}', r'\hline',
           '  & ' + ' & '.join(cols_esc) + r' \\', r'\hline']
    for stat in df.index:
        vals = []
        for c in cols:
            if c in df.columns:
                v = df.loc[stat, c]
                vals.append(f'{v:.4f}' if not pd.isna(v) else '---')
            else:
                vals.append('---')
        out.append(f'  {stat} & ' + ' & '.join(vals) + r' \\')
    out.append(r'\hline')
    out.append(r'\end{tabular}')
    return '\n'.join(out)


# ==============================================================
# ==================== GENERATE TABLES ========================
# ==============================================================

out_a   = os.path.join(OUTPUT_DIR, 'cone_coverage_combined_a.tex')
out_b   = os.path.join(OUTPUT_DIR, 'cone_coverage_combined_b.tex')
out_p   = os.path.join(OUTPUT_DIR, 'party_batch_coverage_summary.tex')

with open(out_a, 'w') as f:
    f.write(_grouped_latex(groups_a, _COLS_A))

with open(out_b, 'w') as f:
    f.write(_grouped_latex(groups_b, _COLS_B))

with open(out_p, 'w') as f:
    f.write(_simple_latex(party_df, _PARTY_COLS))

print(f"\nSaved:")
print(f"  {out_a}")
print(f"  {out_b}")
print(f"  {out_p}")
