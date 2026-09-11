# %%

"""

Data statistics  — small HI dataset.
Tables for how much data is in each split, by evaluation mode. 
Tables that show how much of the original data are covered by the banks participating in the individual scenario, 
the banks participating in FedAvg (only applies to system evaluation), 
and a table that reports how many percentage of transactions where both banks, or only the from/to bank, 
are participating in a given split, also by evaluation mode. For system evaluation mode it is 100% for all except train split.

OUTPUT GOES INTO THE EXPERIMENTAL SETUP SECTION. MORE SPECIFICALLY THE 'BANK FILTERING' AND 'EVALUATIONS FRAMEWORKS' SUBSECTIONS.

"""

import sys
import json
import itertools
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, '/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import df_to_latex_table, load_raw_df, build_party_presence_combined_table

TABLES_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/data')
RB_FILE = Path('/home/nam_07/projects/AML_work_study/experiments/relevant_banks/small_HI__split_0.6_0.2.json')

# %%  

# -------------------------------------------------------------------------------
# Table for section 5: The subsections 'Bank Filtering', 'Evaluation Frameworks'
# -------------------------------------------------------------------------------

# Full data split overview (before comparable filter)

raw_df = load_raw_df(size='small', ir='HI')

timestamps = torch.tensor(raw_df['Timestamp'].to_numpy(), dtype=torch.float32)
split_perc = (0.6, 0.2)

n_days_total = int(timestamps.max() / (3600 * 24) + 1)
daily_inds = []
for day in range(n_days_total):
    l, r = day * 24 * 3600, (day + 1) * 24 * 3600
    day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
    daily_inds.append(day_inds)

d_ts = np.array([len(d) for d in daily_inds])
test_perc = round(1 - sum(split_perc), 10)
split_perc_full = list(split_perc) + [test_perc]
split_scores = {}
for i, j in itertools.combinations(range(len(d_ts)), 2):
    totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
    s = sum(totals)
    props = [v / s for v in totals]
    split_scores[(i, j)] = max(abs(v - t) / t for v, t in zip(props, split_perc_full))
i_star, j_star = min(split_scores, key=split_scores.get)
split_days = [list(range(i_star)), list(range(i_star, j_star)), list(range(j_star, len(d_ts)))]
split_inds_map = {k: [daily_inds[day] for day in split_days[k]] for k in range(3)}
positions = [np.concatenate([t.numpy() for t in split_inds_map[k]]) for k in range(3)]

PRIMARY_DAYS = 10  # Sep 1–10: days with legitimate traffic

split_rows = []
for name, pos, days_list in zip(['Train', 'Vali', 'Test'], positions, split_days):
    df = raw_df.iloc[pos]
    n_txns = len(df)
    n_illicit = int((df['Is Laundering'] == 1).sum())
    n_banks = len(set(df['From Bank'].dropna().unique()) | set(df['To Bank'].dropna().unique()))
    n_accounts = len(set(df['from_id'].dropna().unique()) | set(df['to_id'].dropna().unique()))
    primary = sum(1 for d in days_list if d < PRIMARY_DAYS)
    total   = len(days_list)
    period_str = f'{primary} ({total})' if total > primary else str(total)
    split_rows.append({
        'Split':         name,
        'Transactions':  n_txns,
        'Illicit (N)':   n_illicit,
        'Illicit (\\%)': round(100 * n_illicit / n_txns, 3),
        'Banks (N)':     n_banks,
        'Accounts (N)':  n_accounts,
        'Period (days)': period_str,
    })

df_splits = pd.DataFrame(split_rows)
out_path = TABLES_DIR / 'data_split_overview.tex'
df_to_latex_table(df_splits, out_path)
print(f"Saved: {out_path}")
print(df_splits.to_string(index=False))


# %%  Comparable mode split overview (filtered to 630 individual banks)

with open(RB_FILE) as f:
    rb_comp = json.load(f)
individual_set = set(rb_comp['individual']['indices'])

comp_rows = []
for name, pos, days_list in zip(['Train', 'Vali', 'Test'], positions, split_days):
    comp_pos = pos[np.isin(pos, list(individual_set))]
    df = raw_df.iloc[comp_pos]
    n_txns = len(df)
    n_illicit = int((df['Is Laundering'] == 1).sum())
    n_banks = len(set(df['From Bank'].dropna().unique()) | set(df['To Bank'].dropna().unique()))
    n_accounts = len(set(df['from_id'].dropna().unique()) | set(df['to_id'].dropna().unique()))
    primary = sum(1 for d in days_list if d < PRIMARY_DAYS)
    total   = len(days_list)
    period_str = f'{primary} ({total})' if total > primary else str(total)
    comp_rows.append({
        'Split':         name,
        'Transactions':  n_txns,
        'Illicit (N)':   n_illicit,
        'Illicit (\\%)': round(100 * n_illicit / n_txns, 3) if n_txns else 0.0,
        'Banks (N)':     n_banks,
        'Accounts (N)':  n_accounts,
        'Period (days)': period_str,
    })

df_comp = pd.DataFrame(comp_rows)
out_path = TABLES_DIR / 'data_split_overview_comparable.tex'
df_to_latex_table(df_comp, out_path)
print(f"\nSaved: {out_path}")
print(df_comp.to_string(index=False))



# %% Individual bank coverage

# The table generated here, is the table that holds stats on how much of the original data the 
# 630 banks cover of the different splits.

with open(RB_FILE) as f:
    rb = json.load(f)
ind = rb['individual']

total_illicit_covered = ind['train_laundering'] + ind['vali_laundering'] + ind['test_laundering']

rows = [
    {
        'Split':              'Overall',
        'Obs. coverage (\\%)': round(rb['individual']['percentage'] * 100, 2),
        'Illicit cov. (\\%)': round(ind['total_laundering_pct'] * 100, 2),
        'Illicit (N)':        total_illicit_covered,
        'Total illicit (N)':  rb['total_laundering'],
    },
    {
        'Split':              'Train',
        'Obs. coverage (\\%)': round(ind['train_data_pct'] * 100, 2),
        'Illicit cov. (\\%)': round(ind['train_laundering_pct'] * 100, 2),
        'Illicit (N)':        ind['train_laundering'],
        'Total illicit (N)':  rb['train_laundering'],
    },
    {
        'Split':              'Validation',
        'Obs. coverage (\\%)': round(ind['vali_data_pct'] * 100, 2),
        'Illicit cov. (\\%)': round(ind['vali_laundering_pct'] * 100, 2),
        'Illicit (N)':        ind['vali_laundering'],
        'Total illicit (N)':  rb['vali_laundering'],
    },
    {
        'Split':              'Test',
        'Obs. coverage (\\%)': round(ind['test_data_pct'] * 100, 2),
        'Illicit cov. (\\%)': round(ind['test_laundering_pct'] * 100, 2),
        'Illicit (N)':        ind['test_laundering'],
        'Total illicit (N)':  rb['test_laundering'],
    },
]

df = pd.DataFrame(rows)
out_path = TABLES_DIR / 'individual_banks_coverage.tex'
df_to_latex_table(df, out_path)
print(f"Saved: {out_path}")
print(df.to_string(index=False))


# %%  FedAvg coverage table (same structure as individual)

# Table with stats of how much of the original data the banks participating in the different
# phases/splits in FedAvg cover

fa = rb['FedAvg']
fa_total_illicit = fa['train_laundering'] + fa['vali_laundering'] + fa['test_laundering']
fa_total_banks = len(set(fa['train_banks']) | set(fa['vali_banks']) | set(fa['test_banks']))

fa_rows = [
    {
        'Split':               'Overall',
        'Banks (N)':           fa_total_banks,
        'Obs. coverage (\\%)': round(fa['percentage'] * 100, 2),
        'Illicit cov. (\\%)':  round(fa['total_laundering_pct'] * 100, 2),
        'Illicit (N)':         fa_total_illicit,
        'Total illicit (N)':   rb['total_laundering'],
    },
    {
        'Split':               'Train',
        'Banks (N)':           len(fa['train_banks']),
        'Obs. coverage (\\%)': round(fa['train_data_pct'] * 100, 2),
        'Illicit cov. (\\%)':  round(fa['train_laundering_pct'] * 100, 2),
        'Illicit (N)':         fa['train_laundering'],
        'Total illicit (N)':   rb['train_laundering'],
    },
    {
        'Split':               'Validation',
        'Banks (N)':           len(fa['vali_banks']),
        'Obs. coverage (\\%)': round(fa['vali_data_pct'] * 100, 2),
        'Illicit cov. (\\%)':  round(fa['vali_laundering_pct'] * 100, 2),
        'Illicit (N)':         fa['vali_laundering'],
        'Total illicit (N)':   rb['vali_laundering'],
    },
    {
        'Split':               'Test',
        'Banks (N)':           len(fa['test_banks']),
        'Obs. coverage (\\%)': round(fa['test_data_pct'] * 100, 2),
        'Illicit cov. (\\%)':  round(fa['test_laundering_pct'] * 100, 2),
        'Illicit (N)':         fa['test_laundering'],
        'Total illicit (N)':   rb['test_laundering'],
    },
]

fa_df = pd.DataFrame(fa_rows)
out_path = TABLES_DIR / 'fedavg_banks_coverage.tex'
df_to_latex_table(fa_df, out_path)
print(f"\nSaved: {out_path}")
print(fa_df.to_string(index=False))


# %%  FedGraph bank group sizes
# FedGraph has no pre-computed coverage stats — participation groups are exclusive by split.
# vali/test groups are banks appearing for the first time in those splits (not seen in train).

fg = rb['FedGraph']

fg_rows = [
    {'Group': 'Train',      'Banks (N)': len(fg['train_banks'])},
    {'Group': 'Vali-only',  'Banks (N)': len(fg['vali_banks'])},
    {'Group': 'Test-only',  'Banks (N)': len(fg['test_banks'])},
    {'Group': 'Total',      'Banks (N)': len(fg['train_banks']) + len(fg['vali_banks']) + len(fg['test_banks'])},
]

fg_df = pd.DataFrame(fg_rows)
out_path = TABLES_DIR / 'fedgraph_bank_groups.tex'
df_to_latex_table(fg_df, out_path, row_group_sizes=[3, 1])
print(f"\nSaved: {out_path}")
print(fg_df.to_string(index=False))


# %%  Party presence: how many transactions per split have both/one/no party bank present
# Generates individual comparable/system tables plus a combined table.

presence_df = build_party_presence_combined_table(
    raw_df, size='small', ir='HI',
    split_perc=(0.6, 0.2),
    out_dir=TABLES_DIR,
    out_name='party_presence_combined',
)
if presence_df is not None:
    print("\nParty presence per split (combined):")
    print(presence_df.to_string(index=False))

# %%
