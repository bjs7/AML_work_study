# %%
"""
Pattern analysis — system evaluation, small HI dataset, NON-FRAGMENTED attempts only.

All banks participate (no comparable filter), so visibility is structural with no
denominator correction needed. Restricts to laundering attempts whose transactions
fall entirely within the test split.

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx), V1 (SplitFed).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/')

from lib.analysis_functions import (
    assert_paths_exist,
    build_pattern_stats_wide,
    build_attempt_bank_span_table,
    build_attempt_transaction_class_table,
    txn_class_recall_analysis,
    attempt_level_recall_analysis,
    build_pattern_recall_comparison,
    pattern_recall_delta_vs_baseline,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    reconstruct_train_raw_df,
    cross_bank_recall_analysis,
    pattern_cross_bank_profile,
    build_attempt_visibility_table,
    build_attempt_bank_coverage_table,
    build_attempt_visibility_recall,
    build_attempt_visibility_recall_combined,
    FIGS_DIR,
)
from lib.scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS

EVAL_MODE  = 'system'
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
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
)
test_raw_df['is_cross_bank'] = test_raw_df['From Bank'] != test_raw_df['To Bank']
print(f"  Test split (system): {len(test_raw_df):,} transactions")
n_illicit = (test_raw_df['Is Laundering'] == 1).sum()
n_cross_bank_illicit = (test_raw_df.loc[test_raw_df['Is Laundering'] == 1, 'is_cross_bank']).sum()
print(f"  Illicit in test set: {n_illicit} ({100*n_cross_bank_illicit/n_illicit:.1f}% cross-bank)")


# %%

# =============================================================
# ============ IDENTIFY NON-FRAGMENTED TEST ATTEMPTS ==========
# =============================================================

_train_idx = set(reconstruct_train_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
).index)
_test_idx  = set(test_raw_df.index)
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
    """Keep all legitimate rows; for illicit, keep non-fragmented attempts and pattern 9."""
    illicit = df['Is Laundering'] == 1
    unknown  = illicit & (df['AttemptID'] < 0)
    known_nf = illicit & df['AttemptID'].isin(non_fragmented_ids)
    return df[~illicit | unknown | known_nf]


test_nf = _keep_nf(test_raw_df)

# Index values of non-fragmented illicit test transactions — matches lv['indices']
# in system-mode laundering_values.
nf_illicit_indices = set(test_nf.loc[test_nf['Is Laundering'] == 1].index)


# %%

# ==============================================================
# ================== PATTERN REFERENCE TABLE ==================
# ==============================================================

stats_wide_df = build_pattern_stats_wide(
    test_nf, out_dir=TABLES_STATS,
    out_name='pattern_stats_wide_system_test_nonfragmented',
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
    out_fig=FIGS_DIR / 'pattern_analysis' / 'pattern_recall_S2_F1_P2_system_nonfragmented.pdf',
    out_name="pattern_recall_S2_F1_P2_system_nonfragmented",
    filter_indices=nf_illicit_indices,
)
print("\nPattern recall (system, non-fragmented, recall = TP/P per pattern):")
print(pivot_patterns.to_string(index=False))

delta_patterns = pattern_recall_delta_vs_baseline(
    pivot_patterns,
    scenario_ids=scenario_ids,
    baseline_id="S2",
    out_dir=TABLES_RECALL,
    out_name="pattern_recall_delta_vs_S2_system_nonfragmented",
)
print("\nRecall deficit vs S2 (negative = FL worse):")
print(delta_patterns.to_string(index=False))


# %%

# ===============================================================
# ==================== VISIBILITY & COVERAGE ====================
# ===============================================================

vis_table, vis_df = build_attempt_visibility_table(
    test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_visibility_system_nonfragmented',
    csv_dir=CSV_DIR,
)
if vis_table is not None:
    print("\nMax single-bank visibility per pattern (system, non-fragmented):")
    print(vis_table.to_string(index=False))


# %%

cov_table = build_attempt_bank_coverage_table(
    test_nf,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_coverage_system_nonfragmented',
    csv_dir=CSV_DIR,
)
if cov_table is not None:
    print("\nBank coverage distribution per pattern (system, non-fragmented):")
    print(cov_table.to_string(index=False))


# %%

# ============================================================================
# ======================= RECALL BY VISIBILITY BUCKETS =======================
# ============================================================================

pivot_vis_recall, agg_vis_recall = build_attempt_visibility_recall_combined(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_system_nonfragmented',
    csv_dir=CSV_DIR,
)
if pivot_vis_recall is not None:
    print("\nRecall by visibility bucket (all thresholds, system, non-fragmented):")
    print(pivot_vis_recall.to_string(index=False))


# %%

pivot_vis_any, _ = build_attempt_visibility_recall(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_any_system_nonfragmented',
    csv_dir=CSV_DIR,
    detection_threshold='any',
)
if pivot_vis_any is not None:
    print("\nRecall by visibility bucket (>=1 threshold, system, non-fragmented):")
    print(pivot_vis_any.to_string(index=False))


# %%

# ===================================================================================
# ===================== TABLE UP UNTIL THIS POINT ARE INCLUDED. =====================
# ===================== THE FOLLOWING MAY BE ADDED LATER ============================
# ===================================================================================
