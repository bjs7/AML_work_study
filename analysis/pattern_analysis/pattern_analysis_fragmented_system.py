# %%
"""
Pattern analysis — system evaluation, small HI dataset, ALL test attempts
(including fragmented attempts whose transactions span multiple splits).

All banks participate (no comparable filter), so visibility is structural with no
denominator correction needed. Note: recall denominators and visibility measurements
are based only on the test-split portion of each attempt. For fragmented attempts
this can inflate recall and under-report visibility relative to the full attempt.
See attempt_fragmentation.py for a quantification of this effect, and
pattern_analysis_nonfragmented_system.py for a clean non-fragmented view.

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx), V1 (SplitFed).
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import (
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
    cross_bank_recall_analysis,
    pattern_cross_bank_profile,
    build_attempt_visibility_table,
    build_attempt_bank_coverage_table,
    build_attempt_visibility_recall,
    build_attempt_visibility_recall_combined,
    FIGS_DIR,
)
from scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS

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

# ==============================================================
# ================== PATTERN REFERENCE TABLE ==================
# ==============================================================

stats_wide_df = build_pattern_stats_wide(
    test_raw_df, out_dir=TABLES_STATS,
    out_name='pattern_stats_wide_system_test',
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
    out_fig=FIGS_DIR / 'pattern_analysis' / 'pattern_recall_S2_F1_P2_system.pdf',
    out_name="pattern_recall_S2_F1_P2_system",
)
print("\nPattern recall (system, recall = TP/P per pattern):")
print(pivot_patterns.to_string(index=False))

delta_patterns = pattern_recall_delta_vs_baseline(
    pivot_patterns,
    scenario_ids=scenario_ids,
    baseline_id="S2",
    out_dir=TABLES_RECALL,
    out_name="pattern_recall_delta_vs_S2_system",
)
print("\nRecall deficit vs S2 (negative = FL worse):")
print(delta_patterns.to_string(index=False))


# %%

# ===============================================================
# ==================== VISIBILITY & COVERAGE ====================
# ===============================================================

vis_table, vis_df = build_attempt_visibility_table(
    test_raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_visibility_system_fragmented',
    csv_dir=CSV_DIR,
)
if vis_table is not None:
    print("\nMax single-bank visibility per pattern (system):")
    print(vis_table.to_string(index=False))


# %%

cov_table = build_attempt_bank_coverage_table(
    test_raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_coverage_system_fragmented',
    csv_dir=CSV_DIR,
)
if cov_table is not None:
    print("\nBank coverage distribution per pattern (system):")
    print(cov_table.to_string(index=False))


# %%

# ============================================================================
# ======================= RECALL BY VISIBILITY BUCKETS =======================
# ============================================================================

pivot_vis_recall, agg_vis_recall = build_attempt_visibility_recall_combined(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_system_fragmented',
    csv_dir=CSV_DIR,
)
if pivot_vis_recall is not None:
    print("\nRecall by visibility bucket (all thresholds, system):")
    print(pivot_vis_recall.to_string(index=False))


# %%

pivot_vis_any, _ = build_attempt_visibility_recall(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL,
    out_name='attempt_visibility_recall_any_system_fragmented',
    csv_dir=CSV_DIR,
    detection_threshold='any',
)
if pivot_vis_any is not None:
    print("\nRecall by visibility bucket (>=1 threshold, system):")
    print(pivot_vis_any.to_string(index=False))


# %%

# ===================================================================
# ====================== MORE ATTEMPT ANALYSIS ======================
# ===================================================================
# Moved here from the now-deleted results_analysis_pattern_system.py, which
# was the sole source of these three tables (attempt_bank_span_system,
# attempt_txn_class_system, txn_class_recall_system) — all three are `\input`
# in mmain_doc.tex.

span_df = build_attempt_bank_span_table(
    test_raw_df, raw_df=raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_span_system',
    csv_dir=CSV_DIR,
)
if span_df is not None:
    print("\nAttempt bank span per pattern:")
    print(span_df.to_string(index=False))


# %% ========== Attempt transaction classification ==========
# Classify each illicit transaction (test split) as WB-single, WB-multi, or CB.
# Bank span per attempt computed from full raw_df for accuracy.

txn_class_df = build_attempt_transaction_class_table(
    test_raw_df, raw_df=raw_df,
    out_dir=TABLES_FRAG,
    out_name='attempt_txn_class_system',
    csv_dir=CSV_DIR,
)
if txn_class_df is not None:
    print("\nTransaction class breakdown per pattern:")
    print(txn_class_df.to_string(index=False))


# %% ========== Transaction-class recall ==========
# Recall per scenario broken down by WB-single / WB-multi / CB.

pivot_txn_class, agg_txn_class = txn_class_recall_analysis(
    scenarios, scenario_ids, test_raw_df, raw_df=raw_df,
    out_dir=TABLES_RECALL,
    out_name='txn_class_recall_system',
)
if pivot_txn_class is not None:
    print("\nRecall by transaction class (WB-single / WB-multi / CB):")
    print(pivot_txn_class.to_string(index=False))


# %%

# ===================================================================================
# ===================== TABLE UP UNTIL THIS POINT ARE INCLUDED. =====================
# ===================== THE FOLLOWING MAY BE ADDED LATER ============================
# ===================================================================================




# %%

# ============================================================================
# ======= VISIBILITY & COVERAGE — FULL ATTEMPT DENOMINATOR (APPENDIX) ========
# ============================================================================
# Visibility denominator = all transactions of the attempt across all splits,
# so train/vali transactions are included. Shows how much of the full laundering
# structure any one bank observes — the structural difficulty of detection.
# Recall tables above stay on the test-split denominator for internal consistency.

vis_table_full, _ = build_attempt_visibility_table(
    test_raw_df,
    raw_df=raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_visibility_system_fragmented_full',
    csv_dir=CSV_DIR,
)
if vis_table_full is not None:
    print("\nMax single-bank visibility per pattern (full-attempt denominator, system):")
    print(vis_table_full.to_string(index=False))


# %%

cov_table_full = build_attempt_bank_coverage_table(
    test_raw_df,
    raw_df=raw_df,
    out_dir=TABLES_VIS,
    out_name='attempt_bank_coverage_system_fragmented_full',
    csv_dir=CSV_DIR,
)
if cov_table_full is not None:
    print("\nBank coverage distribution per pattern (full-attempt denominator, system):")
    print(cov_table_full.to_string(index=False))
