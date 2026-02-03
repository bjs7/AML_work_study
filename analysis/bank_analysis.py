# %% 

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')

# packages
import pandas as pd
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
from data.get_indices_type_data import get_indices_bdt
import copy
import data.data_functions as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
import models.gnn_models
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos
import data.feature_engi as fe
from results.save_results import save_results
from data.relevant_banks import get_relevant_banks

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter
from scipy.stats import entropy


def plot_proportion_heatmap(df, cols, xlabel, bank_id_col='bank_id', figsize=(7, 5), fontsize=11):
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='lightgrey')

    prop_df = df[cols].div(df[cols].sum(axis=1), axis=0)
    prop_df.index = df[bank_id_col]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(prop_df.values, aspect='auto', cmap=cmap, interpolation='nearest')
    for x in range(len(cols) - 1):
        ax.axvline(x + 0.5, color='white', linewidth=1.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=fontsize - 1)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Bank', fontsize=fontsize)
    fig.colorbar(ax.images[0], label='Proportion')
    plt.tight_layout()
    return fig


def plot_proportion_stacked_bar(df, cols, bank_id_col='bank_id', sort_by='n_edges', N=50, legend_title='Category', figsize=(7, 4), fontsize=11):
    sorted_df = df.sort_values(sort_by, ascending=False).head(N)
    proportions = sorted_df[cols].fillna(0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xticklabels(sorted_df[bank_id_col].values, rotation=45, fontsize=fontsize - 2)
    ax.set_xlabel('Bank ID (sorted by edge count)', fontsize=fontsize)
    ax.set_ylabel('Proportion', fontsize=fontsize)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), fontsize=fontsize - 2)
    plt.tight_layout()
    return fig


# %% ========== Data Loading ==========


utils.logger_setup()
parsers = utils.parser_all()
utils.set_seed(parsers['data_parser'].seed, True)
# -------------

#parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
#parsers['data_parser'].train_for_final = True
parsers['fl_parser'].fl_algo = 'FedAvg'
#parsers['fl_parser'].fl_algo = 'full_info'
#parsers['data_parser'].scenario = 'individual_banks' if parsers['fl_parser'].fl_algo != 'full_info' else 'full_info'


# Get data ---------------------------------------------------------------------------------------
#df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_withpatterns_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")


# %%

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))
manager = Manager.get_algo_class(parsers)
self = manager
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)


# %%

train_indices = []
vali_indices = []
test_indices = []
for bank_id, party in self.parties.items():
    
    data = party.data['train_data']['df']
    train_indices += party.indices['train_indices']
    vali_indices += party.indices['vali_indices']
    test_indices += party.indices['test_indices']


len(set(test_indices))
list(set(test_indices))

sum(df['regular_data']['train_data']['x'].loc[list(set(train_indices)),'Is Laundering'])
sum(df['regular_data']['vali_data']['x'].loc[list(set(vali_indices)),'Is Laundering'])
sum(df['regular_data']['test_data']['x'].loc[list(set(test_indices)),'Is Laundering'])

sum(df['regular_data']['train_data']['x'].loc[:,'Is Laundering'])
sum(df['regular_data']['vali_data']['x'].loc[:,'Is Laundering'])
sum(df['regular_data']['test_data']['x'].loc[:,'Is Laundering'])


def load_relevant_banks(parsers):

    save_direc = config.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    str_folder = f"{parsers['data_parser'].size}_{parsers['data_parser'].ir}__split_{config.split_perc[0:2][0]}_{config.split_perc[0:2][1]}.json"
    
    file_location = os.path.join(save_direc, str_folder)

    with open(file_location, 'r') as file:
        relevant_banks = json.load(file)

    return relevant_banks


import configs.configs as config
import json
relevant_banks = load_relevant_banks(parsers).get(parsers['fl_parser'].fl_algo)

len(relevant_banks['train_banks'])
len(self.parties)
len(self.vali_parties)
len(self.test_parties)


# %% ========== Stat Calculations ==========


# ===========================================================================================
# ==================================== STAT CALCULATIONS ====================================
# ===========================================================================================



PATTERN_LABELS = {
    f'Pattern_{i}': name for i, name in enumerate([
        'None', 'Fan-out', 'Fan-in', 'Cycle', 'Gather-scatter',
        'Scatter-gather', 'Stack', 'Random', 'Bipartite', 'Unknown'
    ])
}
CURRENCY_LABELS = {
    f'Received_currency_{i}': name for i, name in enumerate([
        'US Dollar', 'Bitcoin', 'Euro', 'AUD', 'Yuan', 'Rupee', 'Yen',
        'MXN Peso', 'UK Pound', 'Ruble', 'CAD', 'CHF', 'BRL', 'SAR', 'ILS'
    ])
}
PAYMENT_LABELS = {
    f'Payment_format_{i}': name for i, name in enumerate([
        'Reinvestment', 'Cheque', 'Credit Card', 'ACH', 'Cash', 'Wire', 'Bitcoin'
    ])
}



# The finding that fraud rates are relatively similar but absolute fraud counts vary a lot is important for FedAvg. 
# In FedAvg, if you weight updates equally across parties, the banks with 50 fraud cases get the same influence as banks with 5000. 
# That's a form of quantity skew that can make training unstable - the small banks will produce noisier gradients.


# Maybe look at what happens if banks with less than 1000 edges are removed?

#data_str = 'test_data'
#parties = self.test_parties
data_str = 'train_data'
parties = self.parties

n_laundering_train = np.sum(df['regular_data'][data_str]['x']['Is Laundering'])

stats = []
for bank_id, party in parties.items():

    data = party.data[data_str]['df']
    edge_attr = party.data[data_str]['df'].edge_attr.numpy()
    currency_dist = Counter(edge_attr[:, 2].tolist())
    payment_dist = Counter(edge_attr[:, 3].tolist())

    laundering_patterns = dict(Counter(data.edge_attr[:,4].tolist()))

    stats.append({
        'bank_id': bank_id,
        'n_nodes': data.x.shape[0],
        'n_edges': data.y.shape[0],
        'n_fraud': np.sum(data.y.tolist()),
        'fraud_rate': np.mean(data.y.tolist()) * 100,
        'total_fraud_rate': np.sum(data.y.tolist()) / n_laundering_train * 100,
        **{f'Pattern_{i}': laundering_patterns.get(i) for i in range(10)},
        'amount_received_mean': edge_attr[:, 1].mean(),
        'amount_received_std': edge_attr[:, 1].std(),
        'amount_received_median': np.median(edge_attr[:, 1]),
        'timestamp_mean': edge_attr[:, 0].mean(),
        'timestamp_std': edge_attr[:, 0].std(),
        **{f'Received_currency_{i}': currency_dist.get(i) for i in range(15)},
        **{f'Payment_format_{i}': payment_dist.get(i) for i in range(7)}
    })


stats_df = pd.DataFrame(stats)
stats_df = stats_df.rename(columns={**PATTERN_LABELS, **CURRENCY_LABELS, **PAYMENT_LABELS})

pattern_cols = [PATTERN_LABELS[f'Pattern_{i}'] for i in range(1, 10)]
currency_cols = list(CURRENCY_LABELS.values())
payment_cols = list(PAYMENT_LABELS.values())



# %% ========== Edge & Node Distributions ==========


# ===========================================================================================
# ==================== EDGE, NODES AND LAUNDERING DSITRIBUTIONS ANALYSIS ====================
# ===========================================================================================


FONTSIZE = 11

import os
SAVE_DIR = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/figures'
os.makedirs(SAVE_DIR, exist_ok=True)

from matplotlib.ticker import FuncFormatter
thousands_fmt = FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}')

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].hist(stats_df['n_nodes'], bins=50, edgecolor='black')
axes[0].set_xlabel('Number of nodes', fontsize=FONTSIZE)
axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
axes[0].xaxis.set_major_formatter(thousands_fmt)

axes[1].hist(stats_df['n_edges'], bins=50, edgecolor='black')
axes[1].set_xlabel('Number of edges', fontsize=FONTSIZE)
axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
axes[1].xaxis.set_major_formatter(thousands_fmt)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'node_edge_hist.pdf'), bbox_inches='tight')


# %% ========== Edge & Node Distributions ==========

# Bar charts sorted by number of edges in descending order

fig, axes = plt.subplots(2, 1, figsize=(7, 4))
sorted_df = stats_df.sort_values('n_edges', ascending=False)
axes[0].bar(range(len(sorted_df)), sorted_df['n_edges'], width=1.0)
axes[0].set_ylabel('Number of edges', fontsize=FONTSIZE)
axes[0].yaxis.set_major_formatter(thousands_fmt)

axes[1].bar(range(len(sorted_df)), sorted_df['n_nodes'], width=1.0)
axes[1].set_ylabel('Number of nodes', fontsize=FONTSIZE)
axes[1].set_xlabel('Banks (sorted by edge count)', fontsize=FONTSIZE)
axes[1].yaxis.set_major_formatter(thousands_fmt)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'edge_node_sorted_bars.pdf'), bbox_inches='tight')


# %% ========== Fraud Rate Analysis ==========

# Histograms of fraud count and fraud rate across banks

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].hist(stats_df['n_fraud'], bins=50, edgecolor='black')
axes[0].set_xlabel('Number of fraud transactions', fontsize=FONTSIZE)
axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)

axes[1].hist(stats_df['fraud_rate'], bins=50, edgecolor='black')
axes[1].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fraud_hist.pdf'), bbox_inches='tight')


# %% ========== Fraud Rate Analysis ==========

# Banks sorted by number of edges: edge count, fraud count and fraud rate

fig, axes = plt.subplots(3, 1, figsize=(7, 5.5))
sorted_df = stats_df.sort_values('n_edges', ascending=False)
axes[0].bar(range(len(sorted_df)), sorted_df['n_edges'], width=1.0)
axes[0].set_ylabel('Number of edges', fontsize=FONTSIZE)
axes[0].yaxis.set_major_formatter(thousands_fmt)

axes[1].bar(range(len(sorted_df)), sorted_df['n_fraud'], width=1.0)
axes[1].set_ylabel('Fraud cases', fontsize=FONTSIZE)

axes[2].bar(range(len(sorted_df)), sorted_df['fraud_rate'], width=1.0)
axes[2].set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
axes[2].set_xlabel('Banks (sorted by edge count)', fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fraud_sorted_bars.pdf'), bbox_inches='tight')


# %% ========== Fraud Rate Analysis ==========

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.scatter(stats_df['n_edges'], stats_df['fraud_rate'],
           s=stats_df['n_fraud'], alpha=0.6)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
ax.set_title('Bubble size = fraud count', fontsize=FONTSIZE)
ax.xaxis.set_major_formatter(thousands_fmt)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fraud_scatter.pdf'), bbox_inches='tight')


# %% ========== Fraud Rate Analysis ==========

# CV values

for col in ['n_edges', 'n_fraud', 'fraud_rate']:
    cv = stats_df[col].mean() / stats_df[col].std()
    print(f"{col}: CV = {cv:.2f}")



# %% ========== Laundering Patterns ==========

# ==========================================================================================
# ================================== ANALYSIS OF PATTERNS ==================================
# ==========================================================================================

# Now to look at the distribution of the laudering patterns
fig = plot_proportion_heatmap(stats_df, pattern_cols, 'Pattern type')
plt.savefig(os.path.join(SAVE_DIR, 'pattern_heatmap.pdf'), bbox_inches='tight')


# %%

fig = plot_proportion_stacked_bar(stats_df, pattern_cols, legend_title='Pattern', N = 50)
plt.savefig(os.path.join(SAVE_DIR, 'pattern_stacked_bar.pdf'), bbox_inches='tight')




# %% ========== Laundering Patterns ==========

pattern_counts = stats_df[pattern_cols].fillna(0)

# CV per pattern across banks (how unevenly distributed is each pattern?)
for col in pattern_cols:
    cv = pattern_counts[col].std() / pattern_counts[col].mean() if pattern_counts[col].mean() > 0 else 0
    print(f"{col}: CV = {cv:.2f}")


# Entropy per bank (how diverse is each bank's fraud patterns?)
# Higher entropy = more diverse patterns, lower = dominated by one pattern
proportions = pattern_counts.div(pattern_counts.sum(axis=1), axis=0).fillna(0)
#for idx, row in proportions.iterrows():
#    bank_id = stats_df.loc[idx, 'bank_id']
#    h = entropy(row)
#    print(f"Bank {bank_id}: entropy = {h:.3f}")


bank_entropy = [entropy(row) for _, row in proportions.iterrows()]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(stats_df['n_edges'], bank_entropy, alpha=0.5)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Pattern entropy', fontsize=FONTSIZE)
ax.xaxis.set_major_formatter(thousands_fmt)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'pattern_entropy_scatter.pdf'), bbox_inches='tight')




# %% ========== Feature Distributions ==========


# ==========================================================================================
# ================================== ANALYSIS OF FEATURES ==================================
# ==========================================================================================


fig, axes = plt.subplots(1, 2, figsize=(7, 3))

axes[0].hist(stats_df['amount_received_mean'], bins=50, edgecolor='black')
axes[0].set_xlabel('Mean amount received', fontsize=FONTSIZE)
axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)

axes[1].hist(stats_df['amount_received_std'], bins=50, edgecolor='black')
axes[1].set_xlabel('Std of amount received', fontsize=FONTSIZE)
axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'amount_received_hist.pdf'), bbox_inches='tight')


# %%

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(stats_df['amount_received_mean'], stats_df['timestamp_mean'], alpha=0.6)
ax.set_xlabel('Mean amount received', fontsize=FONTSIZE)
ax.set_ylabel('Mean timestamp', fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'amount_vs_timestamp.pdf'), bbox_inches='tight')



# %%

# Currency distribution across banks

fig = plot_proportion_heatmap(stats_df, currency_cols, 'Received currency type')
plt.savefig(os.path.join(SAVE_DIR, 'currency_heatmap.pdf'), bbox_inches='tight')

# %%

fig = plot_proportion_stacked_bar(stats_df, currency_cols, legend_title='Currency')
plt.savefig(os.path.join(SAVE_DIR, 'currency_stacked_bar.pdf'), bbox_inches='tight')


# %%

# Payment format distribution across banks

fig = plot_proportion_heatmap(stats_df, payment_cols, 'Payment format type')
plt.savefig(os.path.join(SAVE_DIR, 'payment_heatmap.pdf'), bbox_inches='tight')

# %%

fig = plot_proportion_stacked_bar(stats_df, payment_cols, legend_title='Payment format')
plt.savefig(os.path.join(SAVE_DIR, 'payment_stacked_bar.pdf'), bbox_inches='tight')


# %% ========== Amount by Currency ==========

# Collect per-currency amounts from the full dataset and per-party means

currency_names = list(CURRENCY_LABELS.values())

all_amounts_by_currency = {name: [] for name in currency_names}
party_currency_means = []

for bank_id, party in self.parties.items():
    edge_attr = party.data['train_data']['df'].edge_attr.numpy()
    row = {'bank_id': bank_id}
    for idx, name in enumerate(currency_names):
        mask = edge_attr[:, 2] == idx
        if mask.any():
            all_amounts_by_currency[name].extend(edge_attr[mask, 1].tolist())
            row[name] = edge_attr[mask, 1].mean()
        else:
            row[name] = np.nan
    party_currency_means.append(row)

currency_mean_df = pd.DataFrame(party_currency_means)


# %% ========== Amount by Currency ==========

# Summary statistics per currency (full dataset)

summary_rows = []
for currency in currency_names:
    vals = all_amounts_by_currency[currency]
    if len(vals) == 0:
        continue
    arr = np.array(vals)
    summary_rows.append({
        'currency': currency, 'n': len(arr),
        'mean': arr.mean(), 'std': arr.std(), 'max': arr.max(),
    })

summary_df = pd.DataFrame(summary_rows).set_index('currency')
print(summary_df.to_string())


# %% ========== Amount by Currency ==========

# Visual comparison of summary stats across currencies

fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))

x = np.arange(len(summary_df))
for ax, col, title in zip(axes, ['mean', 'std', 'max'], ['Mean', 'Std', 'Max']):
    ax.bar(x, summary_df[col])
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index, rotation=90, ha='center', fontsize=FONTSIZE - 3)
    ax.set_ylabel('Amount received', fontsize=FONTSIZE - 1)
    ax.set_title(title, fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'currency_amount_summary.pdf'), bbox_inches='tight')


# %% ========== Amount by Currency ==========

# Full dataset: histogram of amounts per currency (outliers above p99 removed)

UPPER_PERCENTILE = 99

currencies_with_data = [c for c in currency_names if len(all_amounts_by_currency[c]) > 0]
n = len(currencies_with_data)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * nrows))
axes = axes.flatten()

for i, currency in enumerate(currencies_with_data):
    arr = np.array(all_amounts_by_currency[currency])
    cutoff = np.percentile(arr, UPPER_PERCENTILE)
    filtered = arr[arr <= cutoff]
    axes[i].hist(filtered, bins=50, edgecolor='black')
    axes[i].set_title(f'{currency} (n={len(arr):,})', fontsize=FONTSIZE - 2)
    axes[i].set_xlabel('Amount received', fontsize=FONTSIZE - 2)
    axes[i].set_ylabel('Count', fontsize=FONTSIZE - 2)
    axes[i].tick_params(labelsize=FONTSIZE - 3)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f'Amount received by currency (full dataset, <p{UPPER_PERCENTILE})', fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'currency_amount_full_hist.pdf'), bbox_inches='tight')


# %% ========== Amount by Currency ==========

# Per party: distribution of mean amounts across banks (outliers above p99 removed)

fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * nrows))
axes = axes.flatten()

for i, currency in enumerate(currencies_with_data):
    vals = currency_mean_df[currency].dropna().values
    cutoff = np.percentile(vals, UPPER_PERCENTILE)
    filtered = vals[vals <= cutoff]
    axes[i].hist(filtered, bins=50, edgecolor='black')
    axes[i].set_title(f'{currency} (n={len(vals)} banks)', fontsize=FONTSIZE - 2)
    axes[i].set_xlabel('Mean amount received', fontsize=FONTSIZE - 2)
    axes[i].set_ylabel('Number of banks', fontsize=FONTSIZE - 2)
    axes[i].tick_params(labelsize=FONTSIZE - 3)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f'Per-bank mean amount by currency (<p{UPPER_PERCENTILE})', fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'currency_amount_perbank_hist.pdf'), bbox_inches='tight')


# %% ========== Inter-bank vs Intra-bank ==========

# ---------------------------------------------------------------------------------------
# Inter-bank vs intra-bank transaction analysis ------------------------------------------
# ---------------------------------------------------------------------------------------

train_df = df['regular_data']['train_data']['x']
#train_df = df['regular_data']['vali_data']['x']
#train_df = df['regular_data']['test_data']['x']
party_banks = set(parties.keys())

inter_intra_stats = []
for bank_id in party_banks:
    bank_edges = train_df[(train_df['From Bank'] == bank_id) | (train_df['To Bank'] == bank_id)]
    n_total = len(bank_edges)
    if n_total == 0:
        continue

    intra = bank_edges[(bank_edges['From Bank'] == bank_id) & (bank_edges['To Bank'] == bank_id)]
    inter = bank_edges[~((bank_edges['From Bank'] == bank_id) & (bank_edges['To Bank'] == bank_id))]

    n_intra = len(intra)
    n_inter = len(inter)

    intra_fraud = intra['Is Laundering'].sum()
    inter_fraud = inter['Is Laundering'].sum()

    inter_intra_stats.append({
        'bank_id': bank_id,
        'n_total': n_total,
        'n_intra': n_intra,
        'n_inter': n_inter,
        'intra_pct': n_intra / n_total * 100,
        'inter_pct': n_inter / n_total * 100,
        'intra_fraud_rate': intra_fraud / n_intra * 100 if n_intra > 0 else np.nan,
        'inter_fraud_rate': inter_fraud / n_inter * 100 if n_inter > 0 else np.nan,
    })

ii_df = pd.DataFrame(inter_intra_stats)

print(f"Mean intra-bank %: {ii_df['intra_pct'].mean():.1f}")
print(f"Mean inter-bank %: {ii_df['inter_pct'].mean():.1f}")


# %% ========== Inter-bank vs Intra-bank ==========

# Distribution of intra-bank proportion across banks

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(ii_df['intra_pct'], bins=50, edgecolor='black')
axes[0].set_xlabel('Intra-bank edge proportion (%)')
axes[0].set_ylabel('Number of banks')

axes[1].hist(ii_df['inter_pct'], bins=50, edgecolor='black')
axes[1].set_xlabel('Inter-bank edge proportion (%)')
axes[1].set_ylabel('Number of banks')

plt.tight_layout()


# %% ========== Inter-bank vs Intra-bank ==========

# Scatter: bank size vs intra-bank proportion

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(ii_df['n_total'], ii_df['intra_pct'], alpha=0.5)
ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
ax.set_ylabel('Intra-bank proportion (%)', fontsize=FONTSIZE)
ax.xaxis.set_major_formatter(thousands_fmt)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'intra_proportion_scatter.pdf'), bbox_inches='tight')


# %% ========== Inter-bank vs Intra-bank ==========

# Fraud rate comparison: intra-bank vs inter-bank

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

vals_intra = ii_df['intra_fraud_rate'].dropna()
vals_inter = ii_df['inter_fraud_rate'].dropna()

axes[0].hist(vals_intra, bins=50, edgecolor='black', alpha=0.7, label='Intra-bank')
axes[0].hist(vals_inter, bins=50, edgecolor='black', alpha=0.7, label='Inter-bank')
axes[0].set_xlabel('Fraud rate (%)')
axes[0].set_ylabel('Number of banks')
axes[0].legend()
axes[0].set_title('Fraud rate distribution')

axes[1].scatter(ii_df['intra_fraud_rate'], ii_df['inter_fraud_rate'], alpha=0.5)
axes[1].plot([0, axes[1].get_xlim()[1]], [0, axes[1].get_xlim()[1]], 'r--', alpha=0.5)
axes[1].set_xlabel('Intra-bank fraud rate (%)')
axes[1].set_ylabel('Inter-bank fraud rate (%)')
axes[1].set_title('Intra vs inter fraud rate per bank')

plt.tight_layout()


# %% ========== Filtered Analysis: Excluding Small Banks ==========

# ---------------------------------------------------------------------------------------
# Repeat key plots after removing small banks (few edges/transactions)
# ---------------------------------------------------------------------------------------

# How many banks survive at different thresholds?
for threshold in [1000, 2500, 5000]:
    n_keep = (stats_df['n_edges'] >= threshold).sum()
    n_total = len(stats_df)
    print(f"MIN_EDGES >= {threshold}: {n_keep}/{n_total} banks remain ({n_keep/n_total*100:.1f}%)")


# %%

MIN_EDGES = 1000

filtered_stats_df = stats_df[stats_df['n_edges'] >= MIN_EDGES].copy()
filtered_ii_df = ii_df[ii_df['n_total'] >= MIN_EDGES].copy()

print(f"Threshold: {MIN_EDGES} edges")
print(f"Banks kept: {len(filtered_stats_df)} / {len(stats_df)}")
print(f"Banks kept (inter/intra): {len(filtered_ii_df)} / {len(ii_df)}")


# %% ========== Filtered: Fraud Rate Scatter ==========

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

axes[0].scatter(stats_df['n_edges'], stats_df['fraud_rate'],
                s=stats_df['n_fraud'], alpha=0.6)
axes[0].set_xlabel('Number of edges')
axes[0].set_ylabel('Fraud rate (%)')
axes[0].set_title('All banks')

axes[1].scatter(filtered_stats_df['n_edges'], filtered_stats_df['fraud_rate'],
                s=filtered_stats_df['n_fraud'], alpha=0.6)
axes[1].set_xlabel('Number of edges')
axes[1].set_ylabel('Fraud rate (%)')
axes[1].set_title(f'Banks with >= {MIN_EDGES} edges')

plt.suptitle('Edges vs Fraud Rate (bubble size = fraud count)')
plt.tight_layout()


# %% ========== Filtered: Pattern Heatmap ==========

plot_proportion_heatmap(filtered_stats_df, pattern_cols, 'Pattern type')
plt.title(f'Pattern proportions (banks with >= {MIN_EDGES} edges)')


# %% ========== Filtered: Inter-bank vs Intra-bank ==========

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top row: intra-bank proportion
axes[0, 0].hist(ii_df['intra_pct'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Intra-bank proportion (%)')
axes[0, 0].set_ylabel('Number of banks')
axes[0, 0].set_title('All banks')

axes[0, 1].hist(filtered_ii_df['intra_pct'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Intra-bank proportion (%)')
axes[0, 1].set_ylabel('Number of banks')
axes[0, 1].set_title(f'>= {MIN_EDGES} edges')

# Bottom row: fraud rate comparison
vals_intra_all = ii_df['intra_fraud_rate'].dropna()
vals_inter_all = ii_df['inter_fraud_rate'].dropna()
axes[1, 0].hist(vals_intra_all, bins=50, edgecolor='black', alpha=0.7, label='Intra')
axes[1, 0].hist(vals_inter_all, bins=50, edgecolor='black', alpha=0.7, label='Inter')
axes[1, 0].set_xlabel('Fraud rate (%)')
axes[1, 0].set_ylabel('Number of banks')
axes[1, 0].legend()
axes[1, 0].set_title('All banks')

vals_intra_f = filtered_ii_df['intra_fraud_rate'].dropna()
vals_inter_f = filtered_ii_df['inter_fraud_rate'].dropna()
axes[1, 1].hist(vals_intra_f, bins=50, edgecolor='black', alpha=0.7, label='Intra')
axes[1, 1].hist(vals_inter_f, bins=50, edgecolor='black', alpha=0.7, label='Inter')
axes[1, 1].set_xlabel('Fraud rate (%)')
axes[1, 1].set_ylabel('Number of banks')
axes[1, 1].legend()
axes[1, 1].set_title(f'>= {MIN_EDGES} edges')

plt.suptitle('Inter-bank vs Intra-bank: All vs Filtered')
plt.tight_layout()


# %% ========== Filtered: Intra vs Inter Fraud Scatter ==========

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(ii_df['intra_fraud_rate'], ii_df['inter_fraud_rate'], alpha=0.5)
lim = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
axes[0].plot([0, lim], [0, lim], 'r--', alpha=0.5)
axes[0].set_xlabel('Intra-bank fraud rate (%)')
axes[0].set_ylabel('Inter-bank fraud rate (%)')
axes[0].set_title('All banks')

axes[1].scatter(filtered_ii_df['intra_fraud_rate'], filtered_ii_df['inter_fraud_rate'], alpha=0.5)
lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
axes[1].set_xlabel('Intra-bank fraud rate (%)')
axes[1].set_ylabel('Inter-bank fraud rate (%)')
axes[1].set_title(f'>= {MIN_EDGES} edges')

plt.suptitle('Intra vs Inter Fraud Rate per Bank')
plt.tight_layout()


# %%

feature_stats = []
for bank_id, party in self.parties.items():

    edge_attr = party.data['train_data']['df'].edge_attr.numpy()
    currency_dist = Counter(edge_attr[:, 2].tolist())
    payment_dist = Counter(edge_attr[:, 3].tolist())
    
    feature_stats.append({
        'bank_id': bank_id,
        'amount_received_mean': edge_attr[:, 1].mean(),
        'amount_received_std': edge_attr[:, 1].std(),
        'amount_received_median': np.median(edge_attr[:, 1]),
        'timestamp_mean': edge_attr[:, 0].mean(),
        'timestamp_std': edge_attr[:, 0].std(),
        **{f'Received_currency_{i}': currency_dist.get(i) for i in range(13)},
        **{f'Payment_format_{i}': payment_dist.get(i) for i in range(7)}
    })

feature_df = pd.DataFrame(feature_stats)




# %%






# %%



fig, axes = plt.subplots(2, 1, figsize=(20, 15))

axes[0].hist(feature_df['timestamp_mean'], bins=200, edgecolor='black', log = False)
axes[0].set_xlabel('Number of nodes in local network')
axes[0].set_ylabel('Number of banks')
#axes[0].set_title('Number of nodes in local network')

axes[1].hist(feature_df['timestamp_std'], bins=200, edgecolor='black', log = False)
axes[1].set_xlabel('Number of edges in local network')
axes[1].set_ylabel('Number of banks')

plt.tight_layout()
plt.show()


# %%


fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    feature_df['amount_received_mean'], 
    feature_df['timestamp_mean'], 
    #s=feature_df['n_fraud'],  # bubble size = fraud count
    alpha=0.6
)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Fraud rate (%)')
ax.set_title('Edges vs Fraud Rate (bubble size = fraud count)')



# %%
















# %%


Counter(df['regular_data']['train_data']['x']['Received Currency'])
Counter(df['regular_data']['train_data']['x']['Payment Format'])




for bank_id, party in self.parties.items():

    currency_dist = Counter(party.data['train_data']['df'].edge_attr[:, 2].tolist())
    payment_dist = Counter(party.data['train_data']['df'].edge_attr[:, 3].tolist())








stats = []
for bank_id, party in self.parties.items():

    data = party.data['train_data']['df']
    laundering_patterns = dict(Counter(data.edge_attr[:,4].tolist()))

    stats.append({
        'bank_id': bank_id,
        'n_nodes': data.x.shape[0],
        'n_edges': data.y.shape[0],
        'n_fraud': np.sum(data.y.tolist()),
        'fraud_rate': np.mean(data.y.tolist()) * 100,
        'total_fraud_rate': np.sum(data.y.tolist()) / n_laundering_train * 100,
        **{f'Pattern_{i}': laundering_patterns.get(i) for i in range(13)}
        **{f'Pattern_{i}': laundering_patterns.get(i) for i in range(7)}
    })
























# %%

len(self.parties)
len(self.parties)














# %%

num_nodes = []
num_edges = []
num_edges_train = []
num_edges_bank_id = {}
less_than_1k_train = []
less_than_1k_test = []

for bank_id, party in manager.parties.items():
    num_nodes.append(party.data['test_data']['df'].x.shape[0])
    num_edges.append(party.data['test_data']['df'].y.shape[0])
    num_edges_bank_id[bank_id] = party.data['test_data']['df'].y.shape[0]

    num_edges_train.append(party.data['train_data']['df'].y.shape[0])
    
    if party.data['test_data']['df'].y.shape[0] < 1000:
        less_than_1k_test.append(bank_id)

    if party.data['train_data']['df'].y.shape[0] < 1000:
        less_than_1k_train.append(bank_id)


# %%

top_30 = dict(sorted(num_edges_bank_id.items(), key=lambda x: x[1], reverse=True)[:30])
top_30.keys()


# %%



fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(num_nodes, bins=100, edgecolor='black')
#axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('num_nodes')

axes[1].hist(num_edges, bins=100, edgecolor='black')
#axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('num_edges')

plt.tight_layout()
plt.show()



# %%

np.where(num_edges < 1000)

whereis = [i for idx, i in enumerate(num_edges) if i < 1000]


llengths = []

for bank in less_than_1k_train:
    llengths.append(len(manager.parties[bank].data['train_data']['df'].y))


min(num_edges_train)


