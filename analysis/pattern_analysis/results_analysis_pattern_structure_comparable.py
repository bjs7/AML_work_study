# %%
"""
Pattern structure analysis — comparable evaluation, small HI dataset.

Two structural angles on laundering attempts, merged into one script because
they share the same data loading:
  1. Scale — how does attempt complexity (degree) vary across pattern types,
     and does it predict FL recall? (fast)
  2. Neighborhood — for each test transaction, how much of its k-hop local
     graph context spans bank boundaries, and does recall depend on that?
     (SLOW — a few minutes even at k=1 on small HI, longer at k=2)

Formerly results_analysis_pattern_scale_comparable.py and
results_analysis_neighborhood_comparable.py. The neighborhood section is kept
as its own `# %%` cell at the end specifically so it's still easy to skip when
running this interactively — running the whole file top-to-bottom (e.g.
`python results_analysis_pattern_structure_comparable.py`) will always pay the
slow k-hop cost, but re-running just the scale section in a notebook/VS Code
interactive session does not.

Degree meaning per pattern type:
  Fan-Out / Fan-In / Cycle / Random / Gather-Scatter — max degree from attempt
  header (e.g. 'Max 16-degree Fan-Out').
  Stack / Bipartite / Scatter-Gather — transaction count in the attempt
  (no degree specified in the header, so attempt size is used as a proxy).

Scenarios: S2 (full-info oracle), F1 (FedAvg), P2 (FedProx), V1 (SplitFed).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import (
    assert_paths_exist,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    reconstruct_train_raw_df,
    build_pattern_degree_distribution_wide,
    build_pattern_degree_recall_wide,
    compute_khop_cross_bank_fracs,
    neighborhood_recall_analysis,
    cross_bank_recall_analysis,
    pattern_cross_bank_profile,
    views_recall_by_pattern,
    build_pattern_recall_comparison,
    pattern_recall_delta_vs_baseline,
    load_comparable_banks,
    FIGS_DIR,
)
from scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS

EVAL_MODE    = 'comparable'
LOCAL_TABLES = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables'
TABLES_BASE  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis'
TABLES_STATS  = TABLES_BASE + '/stats'
TABLES_RECALL = TABLES_BASE + '/recall'
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

# %% ========== 1. Scale: degree distribution (wide) ==========
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

# %% ========== 1. Scale: recall per (pattern × degree bin), wide ==========
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
    out_name=f'neighborhood_cb_recall_k{K}_comparable',
    out_fig=FIGS_DIR / 'pattern_analysis' / f'neighborhood_cb_recall_k{K}_comparable.pdf',
)
print(f"\nRecall by {K}-hop cross-bank fraction bin:")
print(pivot_nhb.to_string(index=False))


# %% ========== 3. Non-fragmented filter ==========
# Restricts all cross-bank sections below to attempts whose illicit transactions
# fall entirely within the test split (no train/vali leakage).

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
    """Keep all legitimate rows; for illicit keep non-fragmented attempts and pattern 9."""
    illicit = df['Is Laundering'] == 1
    unknown  = illicit & (df['AttemptID'] < 0)
    known_nf = illicit & df['AttemptID'].isin(non_fragmented_ids)
    return df[~illicit | unknown | known_nf]


test_nf = _keep_nf(test_raw_df)
nf_illicit_indices = set(test_nf.loc[test_nf['Is Laundering'] == 1].index)


# %% ========== 3. Cross-bank structure — fragmented (all test attempts) ==========

# Compute recall delta vs S2 (reuse saved table if already generated).
_pivot_frag, _ = build_pattern_recall_comparison(
    scenarios, scenario_ids=scenario_ids,
    top_k=9, top_by="support_baseline", baseline_id="S2",
    plot=False, out_dir=TABLES_RECALL, out_name="pattern_recall_S2_F1_P2_comparable",
)
delta_patterns_frag = pattern_recall_delta_vs_baseline(
    _pivot_frag, scenario_ids=scenario_ids, baseline_id="S2",
    out_dir=TABLES_RECALL, out_name="pattern_recall_delta_vs_S2_comparable",
)

pivot_cb, agg_cb = cross_bank_recall_analysis(
    scenarios, scenario_ids, test_raw_df,
    out_dir=TABLES_RECALL, out_name='cross_bank_recall_S2_F1_P2_comparable',
)
print("\nRecall by bank-type (within-bank vs cross-bank illicit transactions):")
print(pivot_cb.to_string(index=False))


# %%

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


# %%

cb_profile = pattern_cross_bank_profile(
    test_raw_df,
    out_dir=TABLES_STATS,
    out_name='pattern_cross_bank_profile_comparable',
)
print("\nCross-bank fraction per laundering pattern:")
print(cb_profile.to_string(index=False))

if cb_profile is not None and delta_patterns_frag is not None:
    delta_col = "F1"
    if delta_col in delta_patterns_frag.columns and "Pattern" in delta_patterns_frag.columns:
        merged = delta_patterns_frag.merge(
            cb_profile[['Pattern_name', 'cross_bank_pct']].rename(columns={'Pattern_name': 'Pattern'}),
            on='Pattern', how='left'
        )
        fig_dir = FIGS_DIR / 'pattern_analysis'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in merged.iterrows():
            ax.scatter(row['cross_bank_pct'], row[delta_col], s=60, zorder=3)
            ax.annotate(row.get('Pattern_name', str(row['Pattern'])),
                        (row['cross_bank_pct'], row[delta_col]),
                        fontsize=7, ha='left', va='bottom')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel("Cross-bank fraction of illicit transactions (pattern)")
        ax.set_ylabel("Recall deficit: F1 − S2 (negative = FL worse)")
        ax.set_title("Does cross-bank structure explain FL recall deficit?")
        plt.tight_layout()
        plt.savefig(fig_dir / 'recall_deficit_vs_cross_bank_pattern_comparable.pdf')
        plt.close()
        print("  Saved scatter plot (fragmented).")


# %% ========== 3. Cross-bank structure — non-fragmented ==========

_pivot_nf, _ = build_pattern_recall_comparison(
    scenarios, scenario_ids=scenario_ids,
    top_k=9, top_by="support_baseline", baseline_id="S2",
    plot=False, out_dir=TABLES_RECALL,
    out_name="pattern_recall_S2_F1_P2_comparable_nonfragmented",
    filter_indices=nf_illicit_indices,
)
delta_patterns_nf = pattern_recall_delta_vs_baseline(
    _pivot_nf, scenario_ids=scenario_ids, baseline_id="S2",
    out_dir=TABLES_RECALL, out_name="pattern_recall_delta_vs_S2_comparable_nonfragmented",
)

pivot_cb_nf, _ = cross_bank_recall_analysis(
    scenarios, scenario_ids, test_nf,
    out_dir=TABLES_RECALL, out_name='cross_bank_recall_S2_F1_P2_comparable_nonfragmented',
)
print("\nRecall by bank-type (non-fragmented):")
print(pivot_cb_nf.to_string(index=False))


# %%

views_rec_nf = views_recall_by_pattern(
    scenarios, ["S2", "V1"], test_nf,
    comparable_banks=comparable_banks,
    filter_indices=nf_illicit_indices,
    out_dir=TABLES_RECALL,
    out_name='views_recall_by_pattern_comparable_nonfragmented',
    csv_dir=CSV_DIR,
)
if views_rec_nf is not None:
    print("\nRecall by number of party views per pattern (S2 vs V1, non-fragmented):")
    print(views_rec_nf.to_string(index=False))


# %%

cb_profile_nf = pattern_cross_bank_profile(
    test_nf,
    out_dir=TABLES_STATS,
    out_name='pattern_cross_bank_profile_comparable_nonfragmented',
)
print("\nCross-bank fraction per laundering pattern (non-fragmented):")
print(cb_profile_nf.to_string(index=False))

if cb_profile_nf is not None and delta_patterns_nf is not None:
    delta_col = "F1"
    if delta_col in delta_patterns_nf.columns and "Pattern" in delta_patterns_nf.columns:
        merged_nf = delta_patterns_nf.merge(
            cb_profile_nf[['Pattern_name', 'cross_bank_pct']].rename(columns={'Pattern_name': 'Pattern'}),
            on='Pattern', how='left'
        )
        fig_dir = FIGS_DIR / 'pattern_analysis'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in merged_nf.iterrows():
            ax.scatter(row['cross_bank_pct'], row[delta_col], s=60, zorder=3)
            ax.annotate(row.get('Pattern_name', str(row['Pattern'])),
                        (row['cross_bank_pct'], row[delta_col]),
                        fontsize=7, ha='left', va='bottom')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel("Cross-bank fraction of illicit transactions (pattern)")
        ax.set_ylabel("Recall deficit: F1 − S2 (negative = FL worse)")
        ax.set_title("Recall deficit vs cross-bank structure (non-fragmented)")
        plt.tight_layout()
        plt.savefig(fig_dir / 'recall_deficit_vs_cross_bank_pattern_comparable_nonfragmented.pdf')
        plt.close()
        print("  Saved scatter plot (non-fragmented).")

# %%
