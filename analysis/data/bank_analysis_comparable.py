# %%

"""
Data heterogeneity analysis — comparable evaluation.

Same approach and same four subsections as bank_analysis.py (system
evaluation), just over the "comparable" population instead: the 630
individual-scenario banks common across all FL scenarios, used for
apples-to-apples comparison, rather than the 1093 FedAvg train_banks used in
system evaluation. See bank_analysis.py's docstring for why this is raw_df +
groupby based rather than going through the Manager/Party pipeline.

Mirrors the "Detection of Heterogeneity - Comparable Evaluation" section in
data_heterogeneity_holder.tex (writing repo) — currently just a placeholder
heading there, since this script (and its figures/tables) didn't exist yet.

Figure/table filenames all get a "_comparable" suffix so they don't collide
with the system-evaluation versions bank_analysis.py writes to the same
figs/heterogeneity and tables/heterogeneity directories.
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/data/heterogeneity')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.configs import split_perc
from configs.paths import get_data_path
from analysis_functions import load_raw_df, reconstruct_train_raw_df, reconstruct_test_raw_df, PATTERN_NAMES

import stats as het_stats
from quantity_skew import build_quantity_skew_table
from label_skew import build_label_skew_table
from pattern_covariate_shift import build_pattern_covariate_table, _bank_entropy
from inter_intra_bank import build_inter_intra_table
from plotting import FONTSIZE, thousands_fmt, savefig, plot_proportion_heatmap


# %% ========== Data Loading ==========

SIZE, IR = 'small', 'HI'
EVAL_MODE = 'comparable'
comparable = True

raw_df = load_raw_df(size=SIZE, ir=IR)

# comparable=True makes both reconstructions apply the same individual-bank
# (630-bank) row filter used during training — train_raw_df/test_raw_df here
# are already restricted to that population, not just party_banks below.
train_raw_df = reconstruct_train_raw_df(raw_df, split_perc=split_perc, comparable=comparable, size=SIZE, ir=IR)
test_raw_df, lv_offset = reconstruct_test_raw_df(raw_df, split_perc=split_perc, comparable=comparable, size=SIZE, ir=IR)
n_train = len(train_raw_df)
vali_raw_df = raw_df.iloc[n_train:lv_offset].reset_index(drop=True)

print(f"Split sizes: train={len(train_raw_df):,} vali={len(vali_raw_df):,} test={len(test_raw_df):,} "
      f"(raw_df total={len(raw_df):,})")

# Party/bank list — the 630 "individual" scenario banks, read directly (not
# via analysis_functions.load_comparable_banks, which returns a plain set)
# so list order is preserved — dict.fromkeys (not set()) keeps the JSON's
# order, matching what Manager/Party would have produced. See bank_analysis.py
# for why this ordering turned out to matter (it affects heatmap row order).
rb_path = (f"{get_data_path()}/AML_work_study/experiments/relevant_banks/"
           f"{SIZE}_{IR}__split_{split_perc[0]}_{split_perc[1]}.json")
with open(rb_path) as f:
    relevant_banks_json = json.load(f)
party_banks = dict.fromkeys(relevant_banks_json['individual']['banks']).keys()
print(f"Party banks (individual/comparable): {len(party_banks)}")


# %% ========== 1. Quantity Skew ==========
# How unevenly are nodes/edges distributed across banks?

ii_df = het_stats.compute_inter_intra_stats(train_raw_df, party_banks)

long_df = pd.concat([
    train_raw_df[['From Bank', 'from_id']].rename(columns={'From Bank': 'bank_id', 'from_id': 'node_id'}),
    train_raw_df[['From Bank', 'to_id']].rename(columns={'From Bank': 'bank_id', 'to_id': 'node_id'}),
    train_raw_df[['To Bank', 'from_id']].rename(columns={'To Bank': 'bank_id', 'from_id': 'node_id'}),
    train_raw_df[['To Bank', 'to_id']].rename(columns={'To Bank': 'bank_id', 'to_id': 'node_id'}),
], ignore_index=True)
n_nodes_series = long_df.groupby('bank_id')['node_id'].nunique()

quantity_df = ii_df[['bank_id', 'n_total']].rename(columns={'n_total': 'n_edges'}).copy()
quantity_df['n_nodes'] = quantity_df['bank_id'].map(n_nodes_series).fillna(0).astype(int)

quantity_table = build_quantity_skew_table(quantity_df, out_name='quantity_skew_summary_comparable')
print("\nQuantity skew summary:")
print(quantity_table.to_string(index=False))

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
savefig('quantity_skew_overview_comparable.pdf')


# %% ========== 2. Label Distribution Skew ==========

ii_indexed = ii_df.set_index('bank_id')
label_df = quantity_df.copy()
label_df['n_fraud'] = label_df['bank_id'].map(ii_indexed['n_fraud'])
label_df['fraud_rate'] = label_df['bank_id'].map(ii_indexed['fraud_rate'])

label_table = build_label_skew_table(label_df, out_name='label_skew_summary_comparable')
print("\nLabel skew summary (fraud count / fraud rate across banks):")
print(label_table.to_string(index=False))

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
savefig('fraud_hist_comparable.pdf')

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
savefig('fraud_sorted_bars_comparable.pdf')

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
savefig('fraud_scatter_comparable.pdf')


# %% ========== 3. Laundering-pattern Covariate Shift (pattern mix only for now) ==========

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

pattern_table = build_pattern_covariate_table(pattern_df, pattern_cols, [], [], out_name='pattern_covariate_summary_comparable')
print("\nPattern covariate-shift summary (CV across banks):")
print(pattern_table.to_string(index=False))

fig = plot_proportion_heatmap(pattern_df, pattern_cols, 'Pattern type')
savefig('pattern_heatmap_comparable.pdf')

bank_entropy = _bank_entropy(pattern_df, pattern_cols)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(pattern_df['n_edges'], bank_entropy, color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Pattern entropy', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('pattern_entropy_scatter_comparable.pdf')

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
savefig('amount_received_hist_comparable.pdf')

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(amount_df['amount_received_mean'], amount_df['timestamp_mean'],
           color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xscale('log')
ax.set_xlabel('Mean amount received', fontsize=FONTSIZE)
ax.set_ylabel('Mean timestamp', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('amount_vs_timestamp_comparable.pdf')

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
savefig('currency_heatmap_comparable.pdf')

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
savefig('payment_heatmap_comparable.pdf')

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
savefig('currency_amount_summary_comparable.pdf')


# %% ========== 4. Inter-bank vs. Intra-bank ==========

ii_table = build_inter_intra_table(ii_df, out_name='inter_intra_summary_comparable')
print("\nInter-bank vs intra-bank summary:")
print(ii_table.to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(ii_df['n_total'], ii_df['intra_pct'], color=BAR_COLOR, alpha=0.5, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Intra-bank proportion (%)', fontsize=FONTSIZE)
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
savefig('intra_proportion_scatter_comparable.pdf')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, xlabel in [(axes[0], 'intra_pct', 'Intra-bank edge proportion (%)'), (axes[1], 'inter_pct', 'Inter-bank edge proportion (%)')]:
    ax.hist(ii_df[col], bins=50, color=BAR_COLOR, edgecolor='white', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel('Number of banks', fontsize=FONTSIZE)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
plt.tight_layout()
savefig('intra_inter_proportion_hist_comparable.pdf')

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
savefig('intra_inter_fraud_comparison_comparable.pdf')

# %%
