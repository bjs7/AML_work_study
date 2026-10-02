# %%
"""
Pattern analysis — comparable evaluation, small HI dataset, NON-FRAGMENTED attempts only.

Restricts the test-split analysis to laundering attempts whose transactions fall
entirely within the test split (no train/vali leakage), so recall denominators and
visibility measurements are self-consistent.

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx), V1 (SplitFed).
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis')

from lib.analysis_functions import (
    assert_paths_exist,
    build_pattern_stats_wide,
    build_pattern_recall_comparison,
    pattern_recall_delta_vs_baseline,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    reconstruct_train_raw_df,
    build_attempt_visibility_table,
    build_attempt_bank_coverage_table,
    build_attempt_visibility_recall,
    build_attempt_visibility_recall_combined,
    build_attempt_size_recall,
    build_attempt_size_visibility_crosstab,
    load_comparable_banks,
    df_to_latex_table,
    build_pattern_recall_precision_combined,
    FIGS_DIR,
)
from lib.scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS

EVAL_MODE  = 'comparable'
TABLES_BASE  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis'
TABLES_STATS  = TABLES_BASE + '/stats'
TABLES_VIS    = TABLES_BASE + '/visibility'
TABLES_RECALL = TABLES_BASE + '/recall'
TABLES_FRAG   = TABLES_BASE + '/fragmentation'
CSV_DIR    = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables/pattern_analysis'


# %%

# ==============================================================
# ==================== SCENARIO DEFINITIONS ====================
# ==============================================================

scenario_ids = DEFAULT_SCENARIO_IDS  # ["S2", "F1", "P2", "V1"]
scenarios = build_scenario_map(scenario_ids, eval_mode=EVAL_MODE)

assert_paths_exist(scenarios)


# %%

# =============================================================
# ======================= LOAD RAW DATA =======================
# =============================================================

print("Loading raw transaction data…")
raw_df = load_raw_df(size='small', ir='HI')
raw_df = enrich_raw_df_with_pattern_degree(raw_df, size='small', ir='HI')
print(f"  Raw transactions: {len(raw_df):,}")

test_raw_df, lv_offset = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=True, size='small', ir='HI'
)
test_raw_df['is_cross_bank'] = test_raw_df['From Bank'] != test_raw_df['To Bank']
print(f"  Test split (comparable): {len(test_raw_df):,} transactions")

system_test_raw_df, _ = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
)
print(f"  Test split (system, reference): {len(system_test_raw_df):,} transactions")

comparable_banks = load_comparable_banks(size='small', ir='HI', split_perc=(0.6, 0.2))


# %%

# =============================================================
# ============ IDENTIFY NON-FRAGMENTED TEST ATTEMPTS ==========
# =============================================================

# Derive split membership from index sets (no positions variable here).
_train_idx = set(reconstruct_train_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
).index)
_test_idx  = set(system_test_raw_df.index)
_vali_idx  = set(raw_df.index) - _train_idx - _test_idx
split_labels = {**{i: 'train' for i in _train_idx},
                **{i: 'vali'  for i in _vali_idx},
                **{i: 'test'  for i in _test_idx}}

illicit_mask = (raw_df['Is Laundering'] == 1) & (raw_df['AttemptID'] >= 0)
illicit_df_full = raw_df[illicit_mask][['AttemptID', 'Pattern']].copy()
illicit_df_full['split'] = illicit_df_full.index.map(split_labels)

test_attempt_ids = set(illicit_df_full[illicit_df_full['split'] == 'test']['AttemptID'].unique())
test_att_info = illicit_df_full[illicit_df_full['AttemptID'].isin(test_attempt_ids)].groupby('AttemptID').apply(
    lambda g: pd.Series({
        'n_train': int((g['split'] == 'train').sum()),
        'n_vali':  int((g['split'] == 'vali').sum()),
        'n_test':  int((g['split'] == 'test').sum()),
    })
).reset_index()
test_att_info['is_fragmented'] = (test_att_info['n_train'] + test_att_info['n_vali']) > 0
non_fragmented_ids = set(test_att_info[~test_att_info['is_fragmented']]['AttemptID'])

n_nf    = len(non_fragmented_ids)
n_total = len(test_attempt_ids)
print(f"\nNon-fragmented test attempts: {n_nf} / {n_total} ({100*n_nf/n_total:.1f}%)")


def _keep_nf(df):
    """Keep all legitimate rows; for illicit, keep non-fragmented attempts and pattern 9.

    Pattern 9 (AttemptID < 0) transactions are not part of multi-transaction attempts,
    so the fragmentation concept does not apply to them.
    """
    illicit = df['Is Laundering'] == 1
    unknown  = illicit & (df['AttemptID'] < 0)
    known_nf = illicit & df['AttemptID'].isin(non_fragmented_ids)
    return df[~illicit | unknown | known_nf]


# Filtered test DataFrames — used for all analysis below.
test_nf        = _keep_nf(test_raw_df)
system_test_nf = _keep_nf(system_test_raw_df)

# Index values of non-fragmented illicit test transactions — matches lv['indices']
# in laundering_values; passed to build_pattern_recall_comparison for filtering.
nf_illicit_indices = set(test_nf.loc[test_nf['Is Laundering'] == 1].index)


# %%

# ==============================================================
# ================== PATTERN REFERENCE TABLE ==================
# ==============================================================

stats_wide_df = build_pattern_stats_wide(
    test_nf, out_dir=TABLES_STATS,
    out_name='pattern_stats_wide_comparable_test_nonfragmented',
)
print(stats_wide_df.to_string(index=False))


# %%

# ==============================================================
# ================== PATTERN RECALL BREAKDOWN ==================
# ==============================================================

pivot_patterns, agg_patterns = build_pattern_recall_comparison(
    scenarios,
    scenario_ids=scenario_ids,
    top_k=9,
    top_by="support_baseline",
    baseline_id="S2",
    plot=True,
    out_dir=TABLES_RECALL,
    out_fig=FIGS_DIR / 'pattern_analysis' / 'pattern_recall_S2_F1_P2_comparable_nonfragmented.pdf',
    out_name="pattern_recall_S2_F1_P2_comparable_nonfragmented",
    filter_indices=nf_illicit_indices,
)
print("\nPattern recall (non-fragmented, recall = TP/P per pattern):")
print(pivot_patterns.to_string(index=False))

delta_patterns = pattern_recall_delta_vs_baseline(
    pivot_patterns,
    scenario_ids=scenario_ids,
    baseline_id="S2",
    out_dir=TABLES_RECALL,
    out_name="pattern_recall_delta_vs_S2_comparable_nonfragmented",
)
print("\nRecall deficit vs S2 (negative = FL worse):")
print(delta_patterns.to_string(index=False))

# %%

combined_df, _ = build_pattern_recall_precision_combined(
    scenarios,
    scenario_ids=scenario_ids,
    top_k=9,
    top_by="support_baseline",
    baseline_id="S2",
    out_dir=TABLES_RECALL,
    out_name="pattern_recall_precision_comparable_nonfragmented",
    filter_indices=nf_illicit_indices,
)


# %%

# ===============================================================
# ==================== VISIBILITY & COVERAGE ====================
# ===============================================================

vis_table, vis_df = build_attempt_visibility_table(
    test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_visibility_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
)
if vis_table is not None:
    print("\nMax single-bank visibility per pattern (non-fragmented):")
    print(vis_table.to_string(index=False))


# %%

# =================================================================================
# ============ SANITY CHECK: why do some Fan-Out attempts have vis<100 ============
# =================================================================================
# Fan-Out hub bank is From Bank on every outgoing transaction → if it is in
# the 630 comparable banks, max_vis = 1.0.  Attempts with max_vis < 1.0 must
# have their hub bank outside the 630.  We verify that here and check why.

_fanout_ids = set(
    test_nf.loc[
        (test_nf['Is Laundering'] == 1) & (test_nf['Pattern'] == 1),
        'AttemptID',
    ].unique()
)
_fanout_df = raw_df[raw_df['AttemptID'].isin(_fanout_ids)]

_vis_check = []
for aid, grp in _fanout_df.groupby('AttemptID'):
    n = len(grp)
    hub_bank  = grp['From Bank'].value_counts().idxmax()
    banks_flt = (set(grp['From Bank']) | set(grp['To Bank'])) & comparable_banks
    bank_vis  = {b: int(((grp['From Bank'] == b) | (grp['To Bank'] == b)).sum()) for b in banks_flt}
    max_v     = max(bank_vis.values()) / n if bank_vis else float('nan')
    _vis_check.append({
        'AttemptID':     aid,
        'n_txns':        n,
        'hub_bank':      hub_bank,
        'hub_in_630':    hub_bank in comparable_banks,
        'max_vis':       round(max_v, 3),
    })

_vis_check_df = pd.DataFrame(_vis_check)
_low_vis      = _vis_check_df[_vis_check_df['max_vis'] < 1.0]
print(f"\nFan-Out non-fragmented attempts: {len(_vis_check_df)}, with vis<1.0: {len(_low_vis)}")

# For each excluded hub bank, show why it is not in the 630.
# Inclusion criterion: laundering in train AND laundering in vali AND any data in test.
_train_idx_set = set(reconstruct_train_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
).index)
_test_idx_set  = set(system_test_raw_df.index)
_vali_idx_set  = set(raw_df.index) - _train_idx_set - _test_idx_set
_split_map     = ({i: 'train' for i in _train_idx_set}
                  | {i: 'vali' for i in _vali_idx_set}
                  | {i: 'test' for i in _test_idx_set})
_raw_s         = raw_df.copy()
_raw_s['_split'] = _raw_s.index.map(_split_map)

print("\nHub banks excluded from 630 and why:")
for _, row in _low_vis.iterrows():
    b    = row['hub_bank']
    bdf  = _raw_s[(_raw_s['From Bank'] == b) | (_raw_s['To Bank'] == b)]
    by_s = bdf.groupby('_split').agg(n_txns=('Is Laundering', 'count'),
                                     n_illicit=('Is Laundering', 'sum'))
    has_ill_train = int(bdf[bdf['_split'] == 'train']['Is Laundering'].sum()) > 0
    has_ill_vali  = int(bdf[bdf['_split'] == 'vali']['Is Laundering'].sum()) > 0
    has_data_test = len(bdf[bdf['_split'] == 'test']) > 0
    reason = (
        'no laundering in train'               if not has_ill_train else
        'no laundering in vali'                if not has_ill_vali  else
        'no data in test'                      if not has_data_test else
        'UNEXPECTED — should be in 630'
    )
    print(f"  AttemptID {int(row['AttemptID'])}, hub bank {b}:  {reason}")
    print(f"    {by_s[['n_txns','n_illicit']].to_dict()}")




# %%

cov_table = build_attempt_bank_coverage_table(
    test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_coverage_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
)
if cov_table is not None:
    print("\nBank coverage distribution per pattern (non-fragmented):")
    print(cov_table.to_string(index=False))


# %%

# ============================================================================
# ======================= RECALL BY VISIBILITY BUCKETS =======================
# ============================================================================

pivot_vis_recall, agg_vis_recall = build_attempt_visibility_recall_combined(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
)
if pivot_vis_recall is not None:
    print("\nRecall by visibility bucket (all thresholds, non-fragmented):")
    print(pivot_vis_recall.to_string(index=False))


# %%

pivot_vis_any, _ = build_attempt_visibility_recall(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_any_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    detection_threshold='any',
    party_banks=comparable_banks,
)
if pivot_vis_any is not None:
    print("\nRecall by visibility bucket (>=1 threshold, non-fragmented):")
    print(pivot_vis_any.to_string(index=False))


# %%

# ============================================================================
# ============ SIZE/VISIBILITY CONFOUND — Stack, Random, Bipartite ==========
# ============================================================================
# Feedback comment 23: FedAvg/FedProx recall is higher in the vis<50 bucket than
# the vis=100 bucket for Stack, Random and Bipartite. These three patterns have no
# structural degree floor (no "Max N-degree" header), so a size=1 attempt — a
# single transaction — is always vis=100 by construction (one transaction, one
# bank). Check whether the vis=100 bucket's low recall is really a visibility
# effect or just these size=1 singletons, which carry the least relational
# structure for a message-passing model, dragging the bucket down.

SIZE_CONFOUND_PATTERNS = [6, 7, 8]  # Stack, Random, Bipartite

pivot_size_recall, _ = build_attempt_size_recall(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_size_recall_any_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    detection_threshold='any',
    party_banks=comparable_banks,
    patterns=SIZE_CONFOUND_PATTERNS,
)
if pivot_size_recall is not None:
    print("\nRecall by attempt-size bucket (>=1 threshold, Stack/Random/Bipartite, non-fragmented):")
    print(pivot_size_recall.to_string(index=False))

# The >=1 threshold gives a larger attempt more chances to count as "detected"
# (only one of its transactions needs to be flagged), so a size effect there
# could just be bookkeeping rather than a real per-transaction detection
# difference. Txn-level recall (denominator = transactions, not attempts)
# removes that multiple-chances effect and checks whether the size effect
# survives at the level of an individual transaction.
pivot_size_recall_txn, _ = build_attempt_size_recall(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_size_recall_txn_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    detection_threshold='txn',
    party_banks=comparable_banks,
    patterns=SIZE_CONFOUND_PATTERNS,
)
if pivot_size_recall_txn is not None:
    print("\nRecall by attempt-size bucket (txn-level, Stack/Random/Bipartite, non-fragmented):")
    print(pivot_size_recall_txn.to_string(index=False))

size_vis_crosstab = build_attempt_size_visibility_crosstab(
    test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_size_visibility_crosstab_comparable_nonfragmented',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
    patterns=SIZE_CONFOUND_PATTERNS,
)
if size_vis_crosstab is not None:
    print("\nAttempt counts by (size bucket x visibility bucket), Stack/Random/Bipartite:")
    print(size_vis_crosstab.to_string(index=False))


# %%

# ============================================================================
# ================================= APPENDIX =================================
# ============================================================================

# ============================================================================
# ========= VISIBILITY & COVERAGE — CORRECTED DENOMINATOR (OPTION 3) =========
# ============================================================================
# Visibility denominator = system (unfiltered) test split, so comparable-
# filtered-out transactions are included in the denominator.  Recall tables
# above stay on the comparable denominator (option A) for internal consistency.

vis_table_corr, _ = build_attempt_visibility_table(
    test_nf,
    raw_df=system_test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_visibility_comparable_nonfragmented_corrected',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
)
if vis_table_corr is not None:
    print("\nMax single-bank visibility per pattern (non-fragmented, corrected denominator):")
    print(vis_table_corr.to_string(index=False))


# %%

cov_table_corr = build_attempt_bank_coverage_table(
    test_nf,
    raw_df=system_test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_coverage_comparable_nonfragmented_corrected',
    csv_dir=CSV_DIR,
    party_banks=comparable_banks,
)
if cov_table_corr is not None:
    print("\nBank coverage distribution per pattern (non-fragmented, corrected denominator):")
    print(cov_table_corr.to_string(index=False))


