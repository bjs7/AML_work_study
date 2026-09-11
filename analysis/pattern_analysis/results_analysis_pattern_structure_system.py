# %%
"""
Pattern structure analysis — system evaluation, small HI dataset.

Two structural angles on laundering attempts, merged into one script because
they share the same data loading:
  1. Scale — how does attempt complexity (degree) vary across pattern types,
     and does it predict FL recall? (fast)
  2. Neighborhood — for each test transaction, how much of its k-hop local
     graph context spans bank boundaries, and does recall depend on that?
     (SLOW — a few minutes even at k=1 on small HI, longer at k=2)

Formerly results_analysis_pattern_scale_system.py and
results_analysis_neighborhood_system.py. The neighborhood section is kept as
its own `# %%` cell at the end specifically so it's still easy to skip when
running this interactively — running the whole file top-to-bottom (e.g.
`python results_analysis_pattern_structure_system.py`) will always pay the
slow k-hop cost, but re-running just the scale section in a notebook/VS Code
interactive session does not.

Degree meaning per pattern type:
  Fan-Out / Fan-In / Cycle / Random / Gather-Scatter — max degree from attempt
  header (e.g. 'Max 16-degree Fan-Out').
  Stack / Bipartite / Scatter-Gather — transaction count in the attempt
  (no degree specified in the header, so attempt size is used as a proxy).

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx).
V1 (FedGraphSimple/SplitFed) excluded — no system-eval results.
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import (
    assert_paths_exist,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    build_pattern_degree_distribution_wide,
    build_pattern_degree_recall_wide,
    compute_khop_cross_bank_fracs,
    neighborhood_recall_analysis,
    FIGS_DIR,
)
from scenarios import build_scenario_map

EVAL_MODE    = 'system'
LOCAL_TABLES = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables'
TABLES_BASE  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis'
TABLES_STATS  = TABLES_BASE + '/stats'
TABLES_RECALL = TABLES_BASE + '/recall'
CSV_DIR      = f'{LOCAL_TABLES}/pattern_analysis'

N_BINS = 4  # degree bins per pattern; adjust after seeing the distribution

# %% ========== Scenario definitions ==========

scenario_ids = ["S2", "F1", "P2"]  # V1 (FedGraphSimple/SplitFed) excluded — no system-eval results
scenarios = build_scenario_map(scenario_ids, eval_mode=EVAL_MODE)

assert_paths_exist(scenarios)

# %% ========== Load raw data ==========

print("Loading raw transaction data…")
raw_df = load_raw_df(size='small', ir='HI')
raw_df = enrich_raw_df_with_pattern_degree(raw_df, size='small', ir='HI')
print(f"  Raw transactions: {len(raw_df):,}")

test_raw_df, lv_offset = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
)
test_raw_df['is_cross_bank'] = test_raw_df['From Bank'] != test_raw_df['To Bank']
print(f"  Test split (system): {len(test_raw_df):,} transactions")
print(f"  laundering_values['indices'] range: [{lv_offset}, {lv_offset + len(test_raw_df) - 1}]")

n_with_degree = test_raw_df['pattern_degree'].notna().sum()
n_illicit = (test_raw_df['Is Laundering'] == 1).sum()
print(f"  Illicit in test: {n_illicit:,}, with degree info: {n_with_degree:,}")

# %% ========== 1. Scale: degree distribution (wide) ==========
# Rows: Q1..Q4 | Cols: pattern | Cell: "lo–hi (pct%)"

dist_df = build_pattern_degree_distribution_wide(
    test_raw_df,
    out_dir=TABLES_STATS,
    out_name='pattern_degree_distribution_system',
    n_bins=N_BINS,
    csv_dir=CSV_DIR,
)
print("\nDegree distribution per pattern (wide):")
print(dist_df.to_string(index=False))

# %% ========== 1. Scale: recall per (pattern × degree bin), wide ==========
# Rows: scenario × Q1..Q4, grouped by scenario | Cols: pattern | Cell: recall

recall_df = build_pattern_degree_recall_wide(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL,
    out_name='pattern_degree_recall_system',
    n_bins=N_BINS,
    csv_dir=CSV_DIR,
)
if recall_df is not None:
    print("\nRecall per (pattern × degree), wide:")
    print(recall_df.to_string(index=False))

# %% ========== 2. Neighborhood: k-hop cross-bank fraction (SLOW) ==========
# For each test transaction, how much of its local graph context spans bank
# boundaries? Recall as a function of this fraction shows whether FL's blind
# spot is structural.
#
# NOTE: This is slow for large graphs (BFS on all test edges). Set k=1 for a
# quick run. For small HI it takes a few minutes at k=2. Not currently
# referenced by mmain_doc.tex.

K = 1  # number of hops; increase to 2 for deeper analysis (slower)

print(f"\nComputing {K}-hop neighborhood cross-bank fractions…")
cb_fracs = compute_khop_cross_bank_fracs(raw_df, test_raw_df, k=K)

print(f"  Cross-bank fraction distribution (over all test edges):")
print(f"    mean={cb_fracs.mean():.3f}  median={cb_fracs.median():.3f}  "
      f"max={cb_fracs.max():.3f}  fraction==0: {(cb_fracs==0).mean():.2%}")

pivot_nhb, agg_nhb = neighborhood_recall_analysis(
    scenarios, scenario_ids, cb_fracs, test_raw_df,
    n_bins=5,
    out_dir=TABLES_RECALL,
    out_name=f'neighborhood_cb_recall_k{K}_system',
    out_fig=FIGS_DIR / 'pattern_analysis' / f'neighborhood_cb_recall_k{K}_system.pdf',
)
print(f"\nRecall by {K}-hop cross-bank fraction bin:")
print(pivot_nhb.to_string(index=False))

# %%
