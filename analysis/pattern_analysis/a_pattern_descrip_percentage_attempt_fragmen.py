# %%

"""

Attempt fragmentation analysis — small HI dataset.

Table with description of the patterns and table with their share of illicit transactions
Characterises how laundering attempts span the train/vali/test temporal splits
(based on the full, unfiltered dataset). The final section then shows the
impact of the comparable bank filter on the test split, for both non-fragmented
and all (including fragmented) test attempts.

OUTPUT GOES INTO THE EXPERIMENTAL SETUP SECTION. MORE SPECIFICALLY THE SUBSECTION 'ANALYSIS OF LAUNDERING PATTERNS'

"""

import sys
import pandas as pd
from pathlib import Path

sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/')

from lib.analysis_functions import (
    df_to_latex_table,
    load_raw_df,
    enrich_raw_df_with_pattern_degree,
    reconstruct_test_raw_df,
    reconstruct_train_raw_df,
    build_comparable_filter_impact_wide,
    build_pattern_overview_table,
    PATTERN_NAMES,
)

TABLES_BASE  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis'
TABLES_STATS  = TABLES_BASE + '/stats'
TABLES_VIS    = TABLES_BASE + '/visibility'
TABLES_RECALL = TABLES_BASE + '/recall'
TABLES_FRAG   = TABLES_BASE + '/fragmentation'
CSV_DIR    = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables/pattern_analysis'


# %%

# =============================================================
# ======================= LOAD RAW DATA =======================
# =============================================================

print("Loading raw transaction data…")
raw_df = load_raw_df(size='small', ir='HI')
raw_df = enrich_raw_df_with_pattern_degree(raw_df, size='small', ir='HI')
print(f"  Raw transactions: {len(raw_df):,}")

system_test_raw_df, _ = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
)
print(f"  Test split: {len(system_test_raw_df):,} transactions")


# %%

# =============================================================
# ================== PATTERN REFERENCE TABLE ==================
# =============================================================

# Table with description of the patterns and table with their share of illicit transactions

desc_df, stats_df = build_pattern_overview_table(
    system_test_raw_df, out_dir=TABLES_STATS,
    out_name_desc='pattern_descriptions',
    out_name_stats='pattern_stats_test',
)
print("\nPattern descriptions:")
print(desc_df.to_string(index=False))
print("\nPattern counts (test split):")
print(stats_df.to_string(index=False))


# %%

# =======================================================================
# ================== ATTEMPTS SPANNING MULTIPLE SPLITS ==================
# =======================================================================

# Build split membership from unfiltered (system) split DataFrames.
_train_idx = set(reconstruct_train_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=False, size='small', ir='HI'
).index)
_test_idx  = set(system_test_raw_df.index)
_vali_idx  = set(raw_df.index) - _train_idx - _test_idx
split_labels = {**{i: 'train' for i in _train_idx},
                **{i: 'vali'  for i in _vali_idx},
                **{i: 'test'  for i in _test_idx}}

illicit_mask = (raw_df['Is Laundering'] == 1) & (raw_df['AttemptID'] >= 0)
illicit_df = raw_df[illicit_mask][['AttemptID', 'Pattern']].copy()
illicit_df['split'] = illicit_df.index.map(split_labels)

attempt_splits = illicit_df.groupby('AttemptID')['split'].nunique()
split_attempts = (attempt_splits > 1).sum()
total_attempts = len(attempt_splits)

print(f"\n=== Split-attempt integrity ===")
print(f"  Total attempts (patterns 1-8):  {total_attempts}")
print(f"  Attempts spanning >1 split:     {split_attempts}  ({100*split_attempts/total_attempts:.1f}%)")


# %%  Table 0: full-dataset illicit composition (known patterns vs unknown)

total_illicit_full   = int((raw_df['Is Laundering'] == 1).sum())
known_illicit_full   = int(illicit_mask.sum())
unknown_illicit_full = total_illicit_full - known_illicit_full

comp_rows = [
    {'Category': 'Known patterns (1–8)', 'Illicit txns (N)': known_illicit_full,
     'Illicit txns (\\%)': round(100 * known_illicit_full / total_illicit_full, 1),
     'Attempts (N)': total_attempts},
    {'Category': 'Unknown (pattern 9)',  'Illicit txns (N)': unknown_illicit_full,
     'Illicit txns (\\%)': round(100 * unknown_illicit_full / total_illicit_full, 1),
     'Attempts (N)': 'N/A'},
    {'Category': 'Total',                'Illicit txns (N)': total_illicit_full,
     'Illicit txns (\\%)': 100.0,
     'Attempts (N)': total_attempts},
]
comp_df = pd.DataFrame(comp_rows)
df_to_latex_table(comp_df, Path(TABLES_STATS) / 'illicit_pattern_composition.tex')
print("\n  Illicit transaction composition (full dataset, all splits):")
print(comp_df.to_string(index=False))


# %%  Tables 1 & 2: per-pattern fragmentation and span breakdown (all attempts)

if split_attempts > 0:
    broken = attempt_splits[attempt_splits > 1].index
    broken_df = illicit_df[illicit_df['AttemptID'].isin(broken)]
    per_pattern = broken_df.groupby('Pattern')['AttemptID'].nunique()

    # Table 1: per-pattern fragmentation counts (transposed, patterns as columns)
    pat_rows = []
    for pat, count in per_pattern.items():
        total_pat = illicit_df[illicit_df['Pattern'] == pat]['AttemptID'].nunique()
        pat_rows.append({
            'Pattern':          PATTERN_NAMES[pat],
            'Fragmented (N)':   count,
            'Total (N)':        total_pat,
            'Fragmented (\\%)': round(100 * count / total_pat, 1),
        })
    pat_rows.append({
        'Pattern':          'Total',
        'Fragmented (N)':   int(split_attempts),
        'Total (N)':        total_attempts,
        'Fragmented (\\%)': round(100 * split_attempts / total_attempts, 1),
    })
    pat_frag_df = pd.DataFrame(pat_rows)
    pat_frag_T = (pat_frag_df.set_index('Pattern').T
                              .reset_index()
                              .rename(columns={'index': 'Metric'}))
    # Transpose upcasts int columns to float64; convert N rows back to int so
    # df_to_latex_table doesn't format them with decimal places.
    # Must convert columns to object dtype first — assigning int into float64
    # columns causes pandas to silently upcast back to float.
    val_cols = [c for c in pat_frag_T.columns if c != 'Metric']
    for col in val_cols:
        pat_frag_T[col] = pat_frag_T[col].astype(object)
    n_mask = pat_frag_T['Metric'].str.endswith('(N)')
    pat_frag_T.loc[n_mask, val_cols] = (
        pat_frag_T.loc[n_mask, val_cols].astype(float).astype(int)
    )
    df_to_latex_table(pat_frag_T, Path(TABLES_FRAG) / 'attempt_fragmentation_per_pattern.tex')
    print("\n  Fragmented attempts per pattern (transposed):")
    print(pat_frag_T.to_string(index=False))

    # Table 2: span breakdown for ALL multi-split attempts
    attempt_info = illicit_df.groupby('AttemptID').apply(
        lambda g: pd.Series({
            'n_train': int((g['split'] == 'train').sum()),
            'n_vali':  int((g['split'] == 'vali').sum()),
            'n_test':  int((g['split'] == 'test').sum()),
            'n_total': len(g),
        })
    ).reset_index()

    def _span_label(row):
        return '+'.join(s for s in ('train', 'vali', 'test') if row[f'n_{s}'] > 0)

    attempt_info['span'] = attempt_info.apply(_span_label, axis=1)
    multi = attempt_info[attempt_info['span'].str.contains(r'\+')]

    span_rows = []
    for span, grp in multi.groupby('span'):
        span_rows.append({
            'Span':             span,
            'Attempts (N)':     len(grp),
            'Avg txns (train)': round(grp['n_train'].mean(), 1),
            'Avg txns (vali)':  round(grp['n_vali'].mean(), 1),
            'Avg txns (test)':  round(grp['n_test'].mean(), 1),
            'Avg txns (total)': round(grp['n_total'].mean(), 1),
            '\\% txns in test': round(100 * grp['n_test'].sum() / grp['n_total'].sum(), 1),
        })
    span_df = pd.DataFrame(span_rows)
    df_to_latex_table(span_df, Path(TABLES_FRAG) / 'attempt_split_span_all.tex')
    print("\n  Multi-split span breakdown (all fragmented attempts):")
    print(span_df.to_string(index=False))


# %%

# =======================================================================================
# ================== ATTEMPTS SPANNING MULTIPLE SPLITS — TEST SPECIFIC ==================
# =======================================================================================

test_attempt_ids = set(illicit_df[illicit_df['split'] == 'test']['AttemptID'].unique())
n_test_attempts  = len(test_attempt_ids)

test_att_info = illicit_df[illicit_df['AttemptID'].isin(test_attempt_ids)].groupby('AttemptID').apply(
    lambda g: pd.Series({
        'n_train': int((g['split'] == 'train').sum()),
        'n_vali':  int((g['split'] == 'vali').sum()),
        'n_test':  int((g['split'] == 'test').sum()),
        'n_total': len(g),
    })
).reset_index()

test_att_info['is_fragmented'] = (test_att_info['n_train'] + test_att_info['n_vali']) > 0
n_fragmented = test_att_info['is_fragmented'].sum()

print(f"\n=== Test-split attempt fragmentation ===")
print(f"  Attempts with >=1 txn in test split:              {n_test_attempts}")
print(f"  Of these, fragmented (also have train/vali txns): {n_fragmented}  ({100*n_fragmented/n_test_attempts:.1f}%)")

if n_fragmented > 0:
    frag = test_att_info[test_att_info['is_fragmented']].copy()
    frag['span'] = frag.apply(
        lambda r: '+'.join(s for s in ('train', 'vali', 'test') if r[f'n_{s}'] > 0), axis=1
    )

    # Table 3: span breakdown for test-appearing fragmented attempts
    frag_rows = []
    for span, grp in frag.groupby('span'):
        frag_rows.append({
            'Span':             span,
            'Attempts (N)':     len(grp),
            'Avg txns (train)': round(grp['n_train'].mean(), 1),
            'Avg txns (vali)':  round(grp['n_vali'].mean(), 1),
            'Avg txns (test)':  round(grp['n_test'].mean(), 1),
            'Avg txns (total)': round(grp['n_total'].mean(), 1),
            '\\% txns in test': round(100 * grp['n_test'].sum() / grp['n_total'].sum(), 1),
        })
    frag_df = pd.DataFrame(frag_rows)
    df_to_latex_table(frag_df, Path(TABLES_FRAG) / 'attempt_split_span_test.tex')
    print("\n  Fragmented test-appearing attempt breakdown by span:")
    print(frag_df.to_string(index=False))


# %%

# ============================================================================
# ================== COMPARABLE FILTER IMPACT ================================
# ============================================================================

# Load the comparable-filtered test split here — only needed for filter impact.
comparable_test_raw_df, _ = reconstruct_test_raw_df(
    raw_df, split_perc=(0.6, 0.2), comparable=True, size='small', ir='HI'
)
print(f"\n  Test split (comparable-filtered): {len(comparable_test_raw_df):,} transactions")

non_fragmented_ids = set(test_att_info[~test_att_info['is_fragmented']]['AttemptID'])


def _keep_nf(df):
    """Keep all legitimate rows; for illicit, keep non-fragmented attempts and pattern 9.

    Pattern 9 (AttemptID < 0) transactions are not part of multi-transaction attempts,
    so the fragmentation concept does not apply to them.
    """
    illicit = df['Is Laundering'] == 1
    unknown  = illicit & (df['AttemptID'] < 0)
    known_nf = illicit & df['AttemptID'].isin(non_fragmented_ids)
    return df[~illicit | unknown | known_nf]


system_test_nf    = _keep_nf(system_test_raw_df)
comparable_test_nf = _keep_nf(comparable_test_raw_df)

filter_impact_nf_df = build_comparable_filter_impact_wide(
    system_test_nf, comparable_test_nf,
    out_dir=TABLES_RECALL,
    out_name='comparable_filter_impact_wide_nonfragmented',
    csv_dir=CSV_DIR,
)
print("\nComparable filter impact (non-fragmented attempts only):")
print(filter_impact_nf_df.to_string(index=False))


# %%

filter_impact_df = build_comparable_filter_impact_wide(
    system_test_raw_df, comparable_test_raw_df,
    out_dir=TABLES_RECALL,
    out_name='comparable_filter_impact_wide',
    csv_dir=CSV_DIR,
)
print("\nComparable filter impact per pattern (all attempts, including fragmented):")
print(filter_impact_df.to_string(index=False))
