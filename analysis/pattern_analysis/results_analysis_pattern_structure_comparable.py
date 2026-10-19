# %%
"""
Pattern structure analysis — comparable evaluation, small HI dataset.

Covers two structural angles on laundering attempts:
  1. Cross-bank analysis — recall by bank type, attempt transaction classes,
     bank span, txn-class recall, cross-bank fraction per pattern.
     (matches the Cross-bank Analysis section in mmain_doc.tex)
  2. Scale — degree distribution and degree-stratified recall per pattern.

Degree meaning per pattern type:
  Fan-Out / Fan-In / Cycle / Random / Gather-Scatter — max degree from attempt
  header (e.g. 'Max 16-degree Fan-Out').
  Stack / Bipartite / Scatter-Gather — transaction count in the attempt
  (no degree specified in the header, so attempt size is used as a proxy).

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx), V1 (SplitFed).
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import (
    assert_paths_exist,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    load_comparable_banks,
    # cross-bank analysis
    cross_bank_recall_analysis,
    build_attempt_transaction_class_table,
    build_attempt_bank_span_table,
    txn_class_recall_analysis,
    pattern_cross_bank_profile,
    views_recall_by_pattern,
    # scale
    build_pattern_degree_distribution_wide,
    build_pattern_degree_recall_wide,
)
from scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS

EVAL_MODE    = 'comparable'
LOCAL_TABLES = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables'
TABLES_BASE  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis'
TABLES_STATS  = TABLES_BASE + '/stats'
TABLES_VIS    = TABLES_BASE + '/visibility'
TABLES_RECALL = TABLES_BASE + '/recall'
TABLES_FRAG   = TABLES_BASE + '/fragmentation'
CSV_DIR      = f'{LOCAL_TABLES}/pattern_analysis'

N_BINS = 4  # degree bins per pattern; adjust after seeing the distribution

# %% ========== Scenario definitions ==========

scenario_ids = DEFAULT_SCENARIO_IDS  # ["S2", "F1", "P2", "V1"]
scenarios = build_scenario_map(scenario_ids, eval_mode=EVAL_MODE)

assert_paths_exist(scenarios)

# %% ========== Load raw data ==========

print("Loading raw transaction data…")
raw_df = load_raw_df(size='small', ir='HI')
raw_df = enrich_raw_df_with_pattern_degree(raw_df, size='small', ir='HI')
print(f"  Raw transactions: {len(raw_df):,}")

test_raw_df, lv_offset = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=True, size='small', ir='HI'
)
test_raw_df['is_cross_bank'] = test_raw_df['From Bank'] != test_raw_df['To Bank']
print(f"  Test split (comparable): {len(test_raw_df):,} transactions")
print(f"  laundering_values['indices'] range: [{lv_offset}, {lv_offset + len(test_raw_df) - 1}]")

n_with_degree = test_raw_df['pattern_degree'].notna().sum()
n_illicit = (test_raw_df['Is Laundering'] == 1).sum()
print(f"  Illicit in test: {n_illicit:,}, with degree info: {n_with_degree:,}")

system_test_raw_df, _ = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
)
comparable_banks = load_comparable_banks(size='small', ir='HI', split_perc=(0.6, 0.2))


# %% ========== 1. Cross-bank analysis ==========

pivot_cb, _ = cross_bank_recall_analysis(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL, out_name='cross_bank_recall_S2_F1_P2_comparable',
)
print("\nRecall by bank-type (within-bank vs cross-bank illicit transactions):")
print(pivot_cb.to_string(index=False))


# %%

txn_class_df = build_attempt_transaction_class_table(
    test_raw_df, raw_df=raw_df,
    out_dir=TABLES_FRAG,
    out_name='attempt_txn_class_comparable',
    csv_dir=CSV_DIR,
)
if txn_class_df is not None:
    print("\nTransaction class breakdown per pattern:")
    print(txn_class_df.to_string(index=False))


# %%

span_df = build_attempt_bank_span_table(
    test_raw_df, raw_df=raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_span_comparable',
    csv_dir=CSV_DIR,
    ref_test_raw_df=system_test_raw_df,
)
if span_df is not None:
    print("\nAttempt bank span per pattern:")
    print(span_df.to_string(index=False))


# %%

pivot_txn_class, _ = txn_class_recall_analysis(
    scenarios, scenario_ids, test_raw_df, raw_df=raw_df,
    out_dir=TABLES_RECALL,
    out_name='txn_class_recall_comparable',
)
if pivot_txn_class is not None:
    print("\nRecall by transaction class (WB-single / WB-multi / CB):")
    print(pivot_txn_class.to_string(index=False))


# %% ========== 2. Scale: degree distribution (wide) ==========
# Rows: Q1..Q4 | Cols: pattern | Cell: "lo–hi (pct%)"

dist_df = build_pattern_degree_distribution_wide(
    test_raw_df,
    out_dir=TABLES_STATS,
    out_name='pattern_degree_distribution_comparable',
    n_bins=N_BINS,
    csv_dir=CSV_DIR,
)
print("\nDegree distribution per pattern (wide):")
print(dist_df.to_string(index=False))

# %% ========== 2. Scale: recall per (pattern × degree bin), wide ==========
# Rows: scenario × Q1..Q4, grouped by scenario | Cols: pattern | Cell: recall

recall_df = build_pattern_degree_recall_wide(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL,
    out_name='pattern_degree_recall_comparable',
    n_bins=N_BINS,
    csv_dir=CSV_DIR,
)
if recall_df is not None:
    print("\nRecall per (pattern × degree), wide:")
    print(recall_df.to_string(index=False))
    

# %%

cb_profile = pattern_cross_bank_profile(
    test_raw_df,
    out_dir=TABLES_STATS,
    out_name='pattern_cross_bank_profile_comparable',
)
print("\nCross-bank fraction per laundering pattern:")
print(cb_profile.to_string(index=False))

views_rec = views_recall_by_pattern(
    scenarios, ["S2", "V1"], test_raw_df,
    comparable_banks=comparable_banks,
    out_dir=TABLES_RECALL,
    out_name='views_recall_by_pattern_comparable_fragmented',
    csv_dir=CSV_DIR,
)
if views_rec is not None:
    print("\nRecall by number of party views per pattern (S2 vs V1):")
    print(views_rec.to_string(index=False))
