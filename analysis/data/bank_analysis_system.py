# %%

"""
Data heterogeneity analysis — how unevenly are the FL parties' data distributed?

Rebuilt step by step to avoid the full FL Manager/Party pipeline
(Manager.setup_parties(...)), which was both slow and — via
party.data[...]['df'].edge_attr — reading the wrong columns for Pattern/
Currency/Payment/Amount (edge_attr never contains Pattern at all, and its
other column positions depend on the ibm_fe flag). Everything here instead
reads directly from the raw transactions CSV (raw_df) plus the
relevant_banks.json bank list, which is enough to reproduce every statistic
in this analysis and is far lighter/more robust.

Mirrors the four subsections of the "Detection of Heterogeneity - System
Evaluation" section in data_heterogeneity_holder.tex (writing repo):
  1. Quantity Skew                                -> done below
  2. Label Distribution Skew                       -> TODO, not yet ported to this approach
  3. Laundering-pattern & Feature Covariate Shift  -> TODO, not yet ported to this approach
  4. Inter-bank vs. Intra-bank                     -> TODO (ii_df below already has what's needed)

Only eval_mode='system' is supported (matches this analysis's own heading).
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/data/heterogeneity')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from configs.configs import split_perc
from data.relevant_banks import load_relevant_banks
from analysis_functions import load_raw_df, reconstruct_train_raw_df, reconstruct_test_raw_df, PATTERN_NAMES

import stats as het_stats
from quantity_skew import build_quantity_skew_table
from label_skew import build_label_skew_table
from pattern_covariate_shift import build_pattern_covariate_table, _bank_entropy
from inter_intra_bank import build_inter_intra_table
from plotting import FONTSIZE, thousands_fmt, savefig, plot_proportion_heatmap


# %% ========== Data Loading ==========

SIZE, IR = 'small', 'HI'
EVAL_MODE = 'system'
FL_ALGO = 'FedAvg'
comparable = (EVAL_MODE == 'comparable')

raw_df = load_raw_df(size=SIZE, ir=IR)

train_raw_df = reconstruct_train_raw_df(raw_df, split_perc=split_perc, comparable=comparable, size=SIZE, ir=IR)
test_raw_df, lv_offset = reconstruct_test_raw_df(raw_df, split_perc=split_perc, comparable=comparable, size=SIZE, ir=IR)
# No reconstruct_vali_raw_df exists — the split is contiguous day-blocks over
# timestamp-sorted raw_df, so vali is exactly the gap between train and test.
n_train = len(train_raw_df)
vali_raw_df = raw_df.iloc[n_train:lv_offset].reset_index(drop=True)

print(f"Split sizes: train={len(train_raw_df):,} vali={len(vali_raw_df):,} test={len(test_raw_df):,} "
      f"(raw_df total={len(raw_df):,})")

# Party/bank list — just the relevant_banks.json lookup, no FL setup needed.
parsers = utils.parser_all()
relevant_banks = load_relevant_banks(parsers['data_parser']).get(FL_ALGO)
# dict.fromkeys (not set()) — preserves relevant_banks.json's list order while
# still deduping, so bank ordering here matches what Manager/Party would have
# produced (utils.add_banks_to_manager just does `for bank in banks`, i.e. the
# JSON list's order) rather than Python's arbitrary set hash-bucket order.
#party_banks = set(relevant_banks['train_banks'])
party_banks = dict.fromkeys(relevant_banks['train_banks']).keys()
print(f"Party banks ({FL_ALGO} train_banks): {len(party_banks)}")


# %% ========== 1. Quantity Skew ==========
# How unevenly are nodes/edges distributed across banks?

# n_edges per bank: reuses compute_inter_intra_stats (n_total = n_intra + n_inter,
# i.e. edges where the bank participates, counted once even for intra-bank edges).
ii_df = het_stats.compute_inter_intra_stats(train_raw_df, party_banks)

# n_nodes per bank: unique accounts (from_id ∪ to_id) touched by edges where the
# bank participates — the node set of that bank's local subgraph, including
# counterparty accounts hosted at other banks. Vectorized via a "long" table
# (each transaction contributes its from_id/to_id to both the sending and
# receiving bank) + groupby-nunique, rather than an O(n_banks * n_rows) filter
# loop over the 3M+-row train_raw_df.
long_df = pd.concat([
    train_raw_df[['From Bank', 'from_id']].rename(columns={'From Bank': 'bank_id', 'from_id': 'node_id'}),
    train_raw_df[['From Bank', 'to_id']].rename(columns={'From Bank': 'bank_id', 'to_id': 'node_id'}),
    train_raw_df[['To Bank', 'from_id']].rename(columns={'To Bank': 'bank_id', 'from_id': 'node_id'}),
    train_raw_df[['To Bank', 'to_id']].rename(columns={'To Bank': 'bank_id', 'to_id': 'node_id'}),
], ignore_index=True)
n_nodes_series = long_df.groupby('bank_id')['node_id'].nunique()

quantity_df = ii_df[['bank_id', 'n_total']].rename(columns={'n_total': 'n_edges'}).copy()
quantity_df['n_nodes'] = quantity_df['bank_id'].map(n_nodes_series).fillna(0).astype(int)

quantity_table = build_quantity_skew_table(quantity_df)
print("\nQuantity skew summary:")
print(quantity_table.to_string(index=False))

# One combined figure: distribution shape (top row, log-x histograms) +
# rank-ordered decay (bottom two rows, log-y sorted bars, stacked so edges and
# nodes can be compared bank-by-bank at the same x-position). Log scale because
# n_edges/n_nodes span ~4 orders of magnitude (15 to 272,000) — linear axes
# would cram almost everything into the first bin/first few bars.
BAR_COLOR = '#2a78d6'
GRID_KW = dict(axis='y', alpha=0.3, linewidth=0.6)
sorted_df = quantity_df.sort_values('n_edges', ascending=False)

fig, axd = plt.subplot_mosaic(
    [['nodes_hist', 'edges_hist'],
     ['edges_sorted', 'edges_sorted'],
     ['nodes_sorted', 'nodes_sorted']],
    figsize=(7, 8),
)

for key, col, xlabel in [('nodes_hist', 'n_nodes', 'Number of nodes'), ('edges_hist', 'n_edges', 'Number of edges')]:
    ax = axd[key]
    vals = quantity_df[col]
    bins = np.logspace(np.log10(max(vals.min(), 1)), np.log10(vals.max()), 30)
    ax.hist(vals, bins=bins, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)

axd['edges_sorted'].bar(range(len(sorted_df)), sorted_df['n_edges'], width=1.0, color=BAR_COLOR)
axd['edges_sorted'].set_yscale('log')
axd['edges_sorted'].set_ylabel('Number of edges', fontsize=FONTSIZE)
axd['edges_sorted'].grid(**GRID_KW)
axd['edges_sorted'].set_axisbelow(True)
axd['edges_sorted'].tick_params(labelbottom=False)

axd['nodes_sorted'].bar(range(len(sorted_df)), sorted_df['n_nodes'], width=1.0, color=BAR_COLOR)
axd['nodes_sorted'].set_yscale('log')
axd['nodes_sorted'].set_ylabel('Number of nodes', fontsize=FONTSIZE)
axd['nodes_sorted'].set_xlabel('Banks (sorted by edge count)', fontsize=FONTSIZE)
axd['nodes_sorted'].grid(**GRID_KW)
axd['nodes_sorted'].set_axisbelow(True)

plt.tight_layout()
savefig('quantity_skew_overview.pdf')


# %% ========== 2. Label Distribution Skew ==========
# How unevenly is fraud (Is Laundering) distributed across banks? n_fraud/
# fraud_rate per bank come straight out of ii_df (compute_inter_intra_stats
# already sums Is Laundering per bank alongside the edge counts).

ii_indexed = ii_df.set_index('bank_id')
label_df = quantity_df.copy()
label_df['n_fraud'] = label_df['bank_id'].map(ii_indexed['n_fraud'])
label_df['fraud_rate'] = label_df['bank_id'].map(ii_indexed['fraud_rate'])

label_table = build_label_skew_table(label_df)
print("\nLabel skew summary (fraud count / fraud rate across banks):")
print(label_table.to_string(index=False))

# Fraud count / fraud rate distribution (same log-scale + BAR_COLOR style as
# Quantity Skew — n_fraud spans ~2.5 orders of magnitude, fraud_rate ~2.8).
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, col, xlabel in [(axes[0], 'n_fraud', 'Number of fraud transactions'), (axes[1], 'fraud_rate', 'Fraud rate (%)')]:
    vals = label_df[col]
    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 30)
    ax.hist(vals, bins=bins, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
plt.tight_layout()
savefig('fraud_hist.pdf')

# Sorted bars: edge count, fraud count, fraud rate (descending by edge count) —
# lets a reader compare all three for the same bank at the same x-position.
sorted_label_df = label_df.sort_values('n_edges', ascending=False)
fig, axes = plt.subplots(3, 1, figsize=(7, 5.5))
for ax, col, ylabel in [(axes[0], 'n_edges', 'Number of edges'), (axes[1], 'n_fraud', 'Fraud cases'), (axes[2], 'fraud_rate', 'Fraud rate (%)')]:
    ax.bar(range(len(sorted_label_df)), sorted_label_df[col], width=1.0, color=BAR_COLOR)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
axes[-1].set_xlabel('Banks (sorted by edge count)', fontsize=FONTSIZE)
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
plt.tight_layout()
savefig('fraud_sorted_bars.pdf')

# Fraud rate vs. bank size (bubble = fraud count) — both axes are heavy-tailed,
# so log-log makes the relationship legible instead of bunching everything
# into the bottom-left corner.
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(label_df['n_edges'], label_df['fraud_rate'], s=label_df['n_fraud'],
           color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
ax.set_title('Bubble size = fraud count', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('fraud_scatter.pdf')


# %% ========== 3. Laundering-pattern Covariate Shift (pattern mix only for now) ==========
# Do banks see different mixes of laundering pattern types? Pattern counts per
# bank use the same intra/inter split as compute_inter_intra_stats above (each
# edge counted once per participating bank), just broken out by Pattern type
# instead of collapsed to a single total. Currency/payment-format/amount
# covariate shift are deferred to a later pass.

pattern_cols = [name for i, name in PATTERN_NAMES.items() if i != 0]

intra_mask = train_raw_df['From Bank'] == train_raw_df['To Bank']
intra_df = train_raw_df[intra_mask]
inter_df = train_raw_df[~intra_mask]

intra_pattern = intra_df.groupby(['From Bank', 'Pattern']).size().unstack(fill_value=0)
inter_pattern_long = pd.concat([
    inter_df[['From Bank', 'Pattern']].rename(columns={'From Bank': 'bank_id'}),
    inter_df[['To Bank', 'Pattern']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)
inter_pattern = inter_pattern_long.groupby(['bank_id', 'Pattern']).size().unstack(fill_value=0)

pattern_counts = intra_pattern.add(inter_pattern, fill_value=0)
pattern_counts = pattern_counts.reindex(columns=list(PATTERN_NAMES.keys()), fill_value=0)
pattern_counts = pattern_counts.rename(columns=PATTERN_NAMES)

pattern_df = quantity_df[['bank_id', 'n_edges']].copy()
for col in pattern_cols:
    pattern_df[col] = pattern_df['bank_id'].map(pattern_counts[col]).fillna(0)

pattern_table = build_pattern_covariate_table(pattern_df, pattern_cols, [], [])
print("\nPattern covariate-shift summary (CV across banks):")
print(pattern_table.to_string(index=False))

# Proportion of each pattern type within a bank's fraud cases (each row sums
# to 1 over pattern_cols, i.e. normalized by n_fraud since Pattern=NONE is
# excluded from pattern_cols) — reuses the existing generic heatmap helper.
fig = plot_proportion_heatmap(pattern_df, pattern_cols, 'Pattern type')
savefig('pattern_heatmap.pdf')

# Pattern-mix entropy vs. bank size — higher entropy means a more uniform
# spread across pattern types; low entropy (e.g. the large outlier bank at
# ~270k edges) means that bank's fraud is dominated by one pattern type.
bank_entropy = _bank_entropy(pattern_df, pattern_cols)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(pattern_df['n_edges'], bank_entropy, color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Pattern entropy', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('pattern_entropy_scatter.pdf')

# Amount received / timestamp: mean/std per bank, over the same "involved
# edges" set used throughout (each edge counted once per participating bank).
intra_amt = intra_df[['From Bank', 'Amount Received', 'Timestamp']].rename(columns={'From Bank': 'bank_id'})
inter_amt = pd.concat([
    inter_df[['From Bank', 'Amount Received', 'Timestamp']].rename(columns={'From Bank': 'bank_id'}),
    inter_df[['To Bank', 'Amount Received', 'Timestamp']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)
amount_long = pd.concat([intra_amt, inter_amt], ignore_index=True)
amount_long = amount_long[amount_long['bank_id'].isin(party_banks)]
amount_stats = amount_long.groupby('bank_id').agg(
    amount_mean=('Amount Received', 'mean'), amount_std=('Amount Received', 'std'),
    timestamp_mean=('Timestamp', 'mean'),
)

amount_df = quantity_df[['bank_id']].copy()
amount_df['amount_received_mean'] = amount_df['bank_id'].map(amount_stats['amount_mean'])
amount_df['amount_received_std'] = amount_df['bank_id'].map(amount_stats['amount_std'])
amount_df['timestamp_mean'] = amount_df['bank_id'].map(amount_stats['timestamp_mean'])

# Both span ~7-8 orders of magnitude across banks — log scale is essentially
# required here (linear axes collapse to a single spike at 0, as in the
# original figure).
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, col, xlabel in [(axes[0], 'amount_received_mean', 'Mean amount received'), (axes[1], 'amount_received_std', 'Std of amount received')]:
    vals = amount_df[col].dropna()
    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 30)
    ax.hist(vals, bins=bins, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
plt.tight_layout()
savefig('amount_received_hist.pdf')

# Mean amount received vs. mean timestamp per bank — does a bank's typical
# transaction size relate to when it's active (e.g. joining/becoming active
# later in the observation window)?
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(amount_df['amount_received_mean'], amount_df['timestamp_mean'],
           color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xscale('log')
ax.set_xlabel('Mean amount received', fontsize=FONTSIZE)
ax.set_ylabel('Mean timestamp', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('amount_vs_timestamp.pdf')

# Currency mix per bank — same intra/inter participation-counted approach as
# the pattern-type heatmap, but over ALL edges (every transaction has a
# currency, unlike Pattern which is only set for fraud cases), so proportions
# here are relative to n_edges rather than n_fraud.
currency_cols = list(het_stats.CURRENCY_NAMES.values())

intra_currency = intra_df.groupby(['From Bank', 'Received Currency']).size().unstack(fill_value=0)
inter_currency_long = pd.concat([
    inter_df[['From Bank', 'Received Currency']].rename(columns={'From Bank': 'bank_id'}),
    inter_df[['To Bank', 'Received Currency']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)
inter_currency = inter_currency_long.groupby(['bank_id', 'Received Currency']).size().unstack(fill_value=0)

currency_counts = intra_currency.add(inter_currency, fill_value=0)
currency_counts = currency_counts.reindex(columns=list(het_stats.CURRENCY_NAMES.keys()), fill_value=0)
currency_counts = currency_counts.rename(columns=het_stats.CURRENCY_NAMES)

currency_df = quantity_df[['bank_id', 'n_edges']].copy()
for col in currency_cols:
    currency_df[col] = currency_df['bank_id'].map(currency_counts[col]).fillna(0)

fig = plot_proportion_heatmap(currency_df, currency_cols, 'Received currency type')
savefig('currency_heatmap.pdf')

# Payment format mix per bank — identical approach to the currency heatmap.
payment_cols = list(het_stats.PAYMENT_NAMES.values())

intra_payment = intra_df.groupby(['From Bank', 'Payment Format']).size().unstack(fill_value=0)
inter_payment_long = pd.concat([
    inter_df[['From Bank', 'Payment Format']].rename(columns={'From Bank': 'bank_id'}),
    inter_df[['To Bank', 'Payment Format']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)
inter_payment = inter_payment_long.groupby(['bank_id', 'Payment Format']).size().unstack(fill_value=0)

payment_counts = intra_payment.add(inter_payment, fill_value=0)
payment_counts = payment_counts.reindex(columns=list(het_stats.PAYMENT_NAMES.keys()), fill_value=0)
payment_counts = payment_counts.rename(columns=het_stats.PAYMENT_NAMES)

payment_df = quantity_df[['bank_id', 'n_edges']].copy()
for col in payment_cols:
    payment_df[col] = payment_df['bank_id'].map(payment_counts[col]).fillna(0)

fig = plot_proportion_heatmap(payment_df, payment_cols, 'Payment format type')
savefig('payment_heatmap.pdf')

# Per-currency amount-received summary (mean/std/max), over the full train
# split directly — each transaction counted once (unlike the per-bank stats
# above, this isn't split by participating bank, so there's no intra/inter
# double-counting concern for inter-bank edges).
currency_amount_summary = train_raw_df.groupby('Received Currency')['Amount Received'].agg(['mean', 'std', 'max'])
currency_amount_summary = currency_amount_summary.reindex(list(het_stats.CURRENCY_NAMES.keys()))
currency_amount_summary.index = currency_amount_summary.index.map(het_stats.CURRENCY_NAMES)

fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))
x = np.arange(len(currency_amount_summary))
for ax, col, title in zip(axes, ['mean', 'std', 'max'], ['Mean', 'Std', 'Max']):
    ax.bar(x, currency_amount_summary[col], color=BAR_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(currency_amount_summary.index, rotation=90, ha='center', fontsize=FONTSIZE - 3)
    ax.set_ylabel('Amount received', fontsize=FONTSIZE - 1)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
plt.tight_layout()
savefig('currency_amount_summary.pdf')


# %% ========== 4. Inter-bank vs. Intra-bank ==========
# What fraction of each bank's transactions cross a bank boundary, and does a
# bank's size predict how much of its traffic stays within itself? ii_df
# already has everything needed (intra_pct/inter_pct/intra_fraud_rate/
# inter_fraud_rate), computed once back in the Quantity Skew section.

ii_table = build_inter_intra_table(ii_df)
print("\nInter-bank vs intra-bank summary:")
print(ii_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(ii_df['n_total'], ii_df['intra_pct'], color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Intra-bank proportion (%)', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('intra_proportion_scatter.pdf')

# Intra-/inter-bank edge-share distribution — backs up the "most parties have
# an intra-bank proportion of roughly 5%-20%" claim with an actual histogram
# of that distribution's shape, rather than just the edges-vs-proportion
# scatter above.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, xlabel in [(axes[0], 'intra_pct', 'Intra-bank edge proportion (%)'), (axes[1], 'inter_pct', 'Inter-bank edge proportion (%)')]:
    ax.hist(ii_df[col], bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
plt.tight_layout()
savefig('intra_inter_proportion_hist.pdf')

# Fraud rate: intra- vs inter-bank. Most banks have zero intra-bank fraud
# cases at all (intra edges are a small slice of an already-rare event), so
# these stay on a linear scale rather than the log treatment used elsewhere.
vals_intra = ii_df['intra_fraud_rate'].dropna()
vals_inter = ii_df['inter_fraud_rate'].dropna()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(vals_intra, bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5, alpha=0.7, label='Intra-bank')
axes[0].hist(vals_inter, bins=50, color='#eb6834', edgecolor='white', linewidth=0.5, alpha=0.7, label='Inter-bank')
axes[0].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
axes[0].legend()
axes[0].set_title('Fraud rate distribution', fontsize=FONTSIZE)
axes[0].grid(**GRID_KW)
axes[0].set_axisbelow(True)

axes[1].scatter(ii_df['intra_fraud_rate'], ii_df['inter_fraud_rate'], color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
axes[1].set_xlabel('Intra-bank fraud rate (%)', fontsize=FONTSIZE)
axes[1].set_ylabel('Inter-bank fraud rate (%)', fontsize=FONTSIZE)
axes[1].set_title('Intra vs inter fraud rate per bank', fontsize=FONTSIZE)
axes[1].grid(alpha=0.3, linewidth=0.6)
axes[1].set_axisbelow(True)
plt.tight_layout()
savefig('intra_inter_fraud_comparison.pdf')


# %% ========== 5. Test Split — Summary Tables Only ==========
# Same population as Sections 1-4 (system eval, FedAvg train_banks) but their
# test-period transactions instead of train — checks whether the skew already
# characterized above persists into the held-out period. Tables only, no
# figures, mirroring how the paper already treats the validation split
# ("similar patterns hold", reported as text rather than a full figure set).

ii_df_test = het_stats.compute_inter_intra_stats(test_raw_df, party_banks)

long_df_test = pd.concat([
    test_raw_df[['From Bank', 'from_id']].rename(columns={'From Bank': 'bank_id', 'from_id': 'node_id'}),
    test_raw_df[['From Bank', 'to_id']].rename(columns={'From Bank': 'bank_id', 'to_id': 'node_id'}),
    test_raw_df[['To Bank', 'from_id']].rename(columns={'To Bank': 'bank_id', 'from_id': 'node_id'}),
    test_raw_df[['To Bank', 'to_id']].rename(columns={'To Bank': 'bank_id', 'to_id': 'node_id'}),
], ignore_index=True)
n_nodes_series_test = long_df_test.groupby('bank_id')['node_id'].nunique()

quantity_df_test = ii_df_test[['bank_id', 'n_total']].rename(columns={'n_total': 'n_edges'}).copy()
quantity_df_test['n_nodes'] = quantity_df_test['bank_id'].map(n_nodes_series_test).fillna(0).astype(int)

quantity_table_test = build_quantity_skew_table(quantity_df_test, out_name='quantity_skew_summary_test')
print("\n[TEST] Quantity skew summary:")
print(quantity_table_test.to_string(index=False))

ii_indexed_test = ii_df_test.set_index('bank_id')
label_df_test = quantity_df_test.copy()
label_df_test['n_fraud'] = label_df_test['bank_id'].map(ii_indexed_test['n_fraud'])
label_df_test['fraud_rate'] = label_df_test['bank_id'].map(ii_indexed_test['fraud_rate'])

label_table_test = build_label_skew_table(label_df_test, out_name='label_skew_summary_test')
print("\n[TEST] Label skew summary:")
print(label_table_test.to_string(index=False))

intra_mask_test = test_raw_df['From Bank'] == test_raw_df['To Bank']
intra_df_test = test_raw_df[intra_mask_test]
inter_df_test = test_raw_df[~intra_mask_test]

intra_pattern_test = intra_df_test.groupby(['From Bank', 'Pattern']).size().unstack(fill_value=0)
inter_pattern_long_test = pd.concat([
    inter_df_test[['From Bank', 'Pattern']].rename(columns={'From Bank': 'bank_id'}),
    inter_df_test[['To Bank', 'Pattern']].rename(columns={'To Bank': 'bank_id'}),
], ignore_index=True)
inter_pattern_test = inter_pattern_long_test.groupby(['bank_id', 'Pattern']).size().unstack(fill_value=0)

pattern_counts_test = intra_pattern_test.add(inter_pattern_test, fill_value=0)
pattern_counts_test = pattern_counts_test.reindex(columns=list(PATTERN_NAMES.keys()), fill_value=0)
pattern_counts_test = pattern_counts_test.rename(columns=PATTERN_NAMES)

pattern_df_test = quantity_df_test[['bank_id', 'n_edges']].copy()
for col in pattern_cols:
    pattern_df_test[col] = pattern_df_test['bank_id'].map(pattern_counts_test[col]).fillna(0)

pattern_table_test = build_pattern_covariate_table(pattern_df_test, pattern_cols, [], [], out_name='pattern_covariate_summary_test')
print("\n[TEST] Pattern covariate-shift summary:")
print(pattern_table_test.to_string(index=False))

ii_table_test = build_inter_intra_table(ii_df_test, out_name='inter_intra_summary_test')
print("\n[TEST] Inter-bank vs intra-bank summary:")
print(ii_table_test.to_string(index=False))

# %%
