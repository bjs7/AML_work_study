# %%

"""
Inter-bank vs. intra-bank analysis on the FULL dataset — small HI.

Same statistics and figures as section 4 of bank_analysis_system.py /
bank_analysis_comparable.py, but over all transactions and all banks: no
train/vali/test split and no FL party filter. Adds the illicit-rate ratio
(inter-bank / intra-bank), i.e. how many times higher the illicit rate is on
transactions that cross a bank boundary.

Also reports, per laundering pattern and overall, how many banks a laundering
attempt spans (the full-dataset counterpart of attempt_bank_span_{comparable,
system}.tex, which only cover attempts touching the test split).

Two bank populations are reported, because the median bank in the full dataset
has only a handful of transactions and those banks dominate any per-bank view:
  - all banks
  - banks with >= MIN_EDGES transactions

Outputs carry a '_full' suffix so nothing from the split-based scripts is
overwritten.
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/data/heterogeneity')

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from analysis_functions import (
    load_raw_df, df_to_latex_table, PATTERN_NAMES, PATTERN_ORDER_NO_UNKNOWN,
)

import stats as het_stats
from quantity_skew import MIN_EDGES
from inter_intra_bank import build_inter_intra_table
from plotting import FONTSIZE, savefig, CSV_DIR, TABLES_DIR

SIZE, IR = 'small', 'HI'
# The bank-span table sits next to its test-split counterparts, not in heterogeneity/.
SPAN_TABLES_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/pattern_analysis/visibility')
SPAN_CSV_DIR = Path('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables/pattern_analysis')
BAR_COLOR = '#2a78d6'
INTER_COLOR = '#eb6834'
GRID_KW = dict(axis='y', alpha=0.3, linewidth=0.6)


def ratio_str(inter, intra):
    return rf"{inter / intra:.1f}$\times$" if intra > 0 else '--'


# %% ========== Data Loading ==========

raw_df = load_raw_df(size=SIZE, ir=IR)
all_banks = sorted(set(raw_df['From Bank']) | set(raw_df['To Bank']))
print(f"Full dataset: {len(raw_df):,} transactions, {len(all_banks):,} banks")


# %% ========== 1. Transaction-level (pooled) ==========
# Every transaction counted exactly once, so this is independent of how many
# banks there are or how small they are.

is_intra = raw_df['From Bank'] == raw_df['To Bank']
n_total = len(raw_df)
n_fraud_total = int(raw_df['Is Laundering'].sum())

pooled = {}
for name, mask in [('Intra-bank', is_intra), ('Inter-bank', ~is_intra)]:
    n = int(mask.sum())
    n_fraud = int(raw_df.loc[mask, 'Is Laundering'].sum())
    pooled[name] = {'n': n, 'n_fraud': n_fraud, 'rate': 100 * n_fraud / n}

pooled_rows = [{
    'Type':               name,
    'Transactions':       f"{p['n']:,}",
    'Share (%)':          f"{100 * p['n'] / n_total:.2f}",
    'Illicit (N)':          f"{p['n_fraud']:,}",
    'Share of illicit (%)': f"{100 * p['n_fraud'] / n_fraud_total:.2f}",
    'Illicit rate (%)':     f"{p['rate']:.3f}",
} for name, p in pooled.items()]

intra_p, inter_p = pooled['Intra-bank'], pooled['Inter-bank']
pooled_rows.append({
    'Type':               'Ratio (inter / intra)',
    'Transactions':       ratio_str(inter_p['n'], intra_p['n']),
    'Share (%)':          '',
    'Illicit (N)':          ratio_str(inter_p['n_fraud'], intra_p['n_fraud']),
    'Share of illicit (%)': '',
    'Illicit rate (%)':     ratio_str(inter_p['rate'], intra_p['rate']),
})

pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(CSV_DIR / 'inter_intra_pooled_full.csv', index=False)
df_to_latex_table(pooled_df, TABLES_DIR / 'inter_intra_pooled_full.tex', row_group_sizes=[2, 1])
print("\nTransaction-level inter- vs intra-bank (full dataset):")
print(pooled_df.to_string(index=False))


# %% ========== 2. Per-bank ==========

ii_df = het_stats.compute_inter_intra_stats(raw_df, all_banks)
ii_df_filtered = ii_df[ii_df['n_total'] >= MIN_EDGES].copy()
print(f"\nBanks with >= {MIN_EDGES:,} transactions: {len(ii_df_filtered):,} / {len(ii_df):,}")

# How much of the data do the banks above the size cutoff cover? A transaction is
# covered when its sender or receiver bank (or both) is one of them.
big_banks = set(ii_df_filtered['bank_id'])
from_big = raw_df['From Bank'].isin(big_banks)
to_big = raw_df['To Bank'].isin(big_banks)
covered = from_big | to_big
is_fraud = raw_df['Is Laundering'] == 1
print(f"  Coverage of banks >= {MIN_EDGES:,} transactions: "
      f"{100 * covered.mean():.2f}% of all transactions ({int(covered.sum()):,} / {n_total:,}), "
      f"{100 * (covered & is_fraud).sum() / n_fraud_total:.2f}% of illicit transactions; "
      f"both endpoints at such a bank: {100 * (from_big & to_big).mean():.2f}%")

# Is the low overall intra-bank share hiding banks that are mostly intra-bank?
mostly_intra = ii_df[ii_df['intra_pct'] >= 50]
print(f"  Intra-bank share among banks >= {MIN_EDGES:,} transactions: "
      f"min {ii_df_filtered['intra_pct'].min():.1f}%, median {ii_df_filtered['intra_pct'].median():.1f}%, "
      f"max {ii_df_filtered['intra_pct'].max():.1f}%")
print(f"  Banks with >= 50% intra-bank transactions: {len(mostly_intra):,} "
      f"(largest has {mostly_intra['n_total'].max():,} transactions, "
      f"median {mostly_intra['n_total'].median():g})")

populations = [
    ('all banks',                            ii_df,          '_full'),
    (rf'banks $\geq$ {MIN_EDGES:,} edges', ii_df_filtered, f'_full_min{MIN_EDGES}'),
]

for label, df, suffix in populations:
    table = build_inter_intra_table(df, out_name=f'inter_intra_summary{suffix}')
    print(f"\nInter-bank vs intra-bank summary — {label} (N={len(df):,}):")
    print(table.to_string(index=False))


# %% ========== 3. Fraud-rate ratio (inter / intra) ==========
# Pooled = one rate over all transactions. Per-bank mean = each bank's own
# intra/inter rate averaged across banks (banks with no intra- or no inter-bank
# transactions are left out of the respective mean).

ratio_rows = [{
    'Basis':                     'Pooled (all transactions)',
    'Intra-bank illicit rate (%)': f"{intra_p['rate']:.3f}",
    'Inter-bank illicit rate (%)': f"{inter_p['rate']:.3f}",
    'Ratio':                     ratio_str(inter_p['rate'], intra_p['rate']),
}]
for label, df, _ in populations:
    intra_mean = df['intra_fraud_rate'].mean()
    inter_mean = df['inter_fraud_rate'].mean()
    ratio_rows.append({
        'Basis':                     f'Per-bank mean, {label}',
        'Intra-bank illicit rate (%)': f"{intra_mean:.3f}",
        'Inter-bank illicit rate (%)': f"{inter_mean:.3f}",
        'Ratio':                     ratio_str(inter_mean, intra_mean),
    })

ratio_df = pd.DataFrame(ratio_rows)
ratio_df.to_csv(CSV_DIR / 'inter_intra_fraud_ratio_full.csv', index=False)
df_to_latex_table(ratio_df, TABLES_DIR / 'inter_intra_fraud_ratio_full.tex')
print("\nIllicit-rate ratio, inter- vs intra-bank (full dataset):")
print(ratio_df.to_string(index=False))


# %% ========== 4. Bank span per laundering attempt ==========
# Distinct banks (From Bank ∪ To Bank) over all transactions of an attempt.
# Illicit transactions without an AttemptID (pattern 'Unknown') belong to no
# attempt and are left out.

attempt_txns = raw_df[(raw_df['Is Laundering'] == 1) & (raw_df['AttemptID'] >= 0)]
n_unassigned = n_fraud_total - len(attempt_txns)

attempt_banks_long = pd.concat([
    attempt_txns[['AttemptID', 'From Bank']].rename(columns={'From Bank': 'bank_id'}),
    attempt_txns[['AttemptID', 'To Bank']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)

attempt_df = pd.DataFrame({
    'Pattern': attempt_txns.groupby('AttemptID')['Pattern'].first(),
    'n_txns':  attempt_txns.groupby('AttemptID').size(),
    'n_banks': attempt_banks_long.groupby('AttemptID')['bank_id'].nunique(),
})


def span_row(name, sub):
    n = len(sub)
    return {
        'Pattern':       name,
        'Attempts':      f"{n:,}",
        'Mean txns':     f"{sub['n_txns'].mean():.1f}",
        'Median txns':   f"{sub['n_txns'].median():g}",
        'Mean banks':    f"{sub['n_banks'].mean():.2f}",
        'Median banks':  f"{sub['n_banks'].median():g}",
        '1-bank (%)':    f"{100 * (sub['n_banks'] == 1).sum() / n:.1f}",
        '2-bank (%)':    f"{100 * (sub['n_banks'] == 2).sum() / n:.1f}",
        '3-bank (%)':    f"{100 * (sub['n_banks'] == 3).sum() / n:.1f}",
        '4+-bank (%)':   f"{100 * (sub['n_banks'] >= 4).sum() / n:.1f}",
    }


span_rows = [span_row(PATTERN_NAMES[pid], attempt_df[attempt_df['Pattern'] == pid])
             for pid in PATTERN_ORDER_NO_UNKNOWN if (attempt_df['Pattern'] == pid).any()]
span_rows.append(span_row('Overall', attempt_df))

span_df = pd.DataFrame(span_rows)
SPAN_CSV_DIR.mkdir(parents=True, exist_ok=True)
span_df.to_csv(SPAN_CSV_DIR / 'attempt_bank_span_full.csv', index=False)
df_to_latex_table(span_df, SPAN_TABLES_DIR / 'attempt_bank_span_full.tex',
                  row_group_sizes=[len(span_rows) - 1, 1])
print(f"\nBank span per laundering attempt (full dataset, {len(attempt_df):,} attempts; "
      f"{n_unassigned:,} illicit transactions belong to no attempt):")
print(span_df.to_string(index=False))


# %% ========== 5. Figures ==========

for label, df, suffix in populations:

    # Bank size vs. intra-bank proportion. Log x: bank size spans ~5-6 orders
    # of magnitude once the tiny banks are included.
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df['n_total'], df['intra_pct'], color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
    ax.set_ylabel('Intra-bank proportion (%)', fontsize=FONTSIZE)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(f'intra_proportion_scatter{suffix}.pdf')

    # Intra-/inter-bank edge-share distribution across banks.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, xlabel in [(axes[0], 'intra_pct', 'Intra-bank edge proportion (%)'),
                            (axes[1], 'inter_pct', 'Inter-bank edge proportion (%)')]:
        ax.hist(df[col], bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
        ax.grid(**GRID_KW)
        ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(f'intra_inter_proportion_hist{suffix}.pdf')

    # Inter-bank edge share on its own (the intra-bank panel is its mirror image).
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df['inter_pct'], bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Inter-bank edge proportion (%)', fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    plt.tight_layout()
    savefig(f'inter_proportion_hist{suffix}.pdf')

    # Fraud rate: intra- vs inter-bank.
    vals_intra = df['intra_fraud_rate'].dropna()
    vals_inter = df['inter_fraud_rate'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(vals_intra, bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5, alpha=0.7, label='Intra-bank')
    axes[0].hist(vals_inter, bins=50, color=INTER_COLOR, edgecolor='white', linewidth=0.5, alpha=0.7, label='Inter-bank')
    axes[0].set_xlabel('Illicit rate (%)', fontsize=FONTSIZE)
    axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
    axes[0].legend()
    axes[0].set_title('Illicit rate distribution', fontsize=FONTSIZE)
    axes[0].grid(**GRID_KW)
    axes[0].set_axisbelow(True)

    axes[1].scatter(df['intra_fraud_rate'], df['inter_fraud_rate'], color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
    axes[1].set_xlabel('Intra-bank illicit rate (%)', fontsize=FONTSIZE)
    axes[1].set_ylabel('Inter-bank illicit rate (%)', fontsize=FONTSIZE)
    axes[1].set_title('Intra vs inter illicit rate per bank', fontsize=FONTSIZE)
    axes[1].grid(alpha=0.3, linewidth=0.6)
    axes[1].set_axisbelow(True)
    plt.tight_layout()
    savefig(f'intra_inter_fraud_comparison{suffix}.pdf')

# %%
