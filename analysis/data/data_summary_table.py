# %%

"""
Dataset summary-statistics table for the Data section — small HI.

One table, per temporal split and for the full dataset:
  - volume: transactions, accounts, banks, days
  - labels: illicit transactions, illicit rate, laundering attempts
  - bank size: transactions per bank (median, p90, max)
  - cross-bank: share of all and of illicit transactions that are cross-bank
  - motif counts: illicit transactions per laundering pattern

A bank is active in a split if one of its accounts sends or receives a
transaction in it. Transactions per bank counts every transaction the bank is
sender or receiver in (a cross-bank transaction counts for both banks), the same
definition as the heterogeneity analysis. Attempts are counted in every split
they have a transaction in, so the split columns sum to more than the full count.
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from pathlib import Path
import pandas as pd

from analysis_functions import (
    load_raw_df, reconstruct_train_raw_df, reconstruct_test_raw_df, df_to_latex_table,
    PATTERN_NAMES, PATTERN_ORDER,
)

SIZE, IR = 'small', 'HI'
SPLIT_PERC = (0.6, 0.2)
PRIMARY_DAYS = 10  # days 0-9 carry the regular traffic; later days only finish off laundering attempts

TABLES_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/data')
CSV_DIR = Path('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables/data')


# %% ========== Splits ==========

raw_df = load_raw_df(size=SIZE, ir=IR)

# The split is contiguous day-blocks over the timestamp-sorted raw_df, so the
# validation split is exactly the gap between train and test.
n_train = len(reconstruct_train_raw_df(raw_df, split_perc=SPLIT_PERC, comparable=False, size=SIZE, ir=IR))
_, test_start = reconstruct_test_raw_df(raw_df, split_perc=SPLIT_PERC, comparable=False, size=SIZE, ir=IR)

splits = {
    'Train':      raw_df.iloc[:n_train],
    'Validation': raw_df.iloc[n_train:test_start],
    'Test':       raw_df.iloc[test_start:],
    'Full':       raw_df,
}


# %% ========== Statistics ==========

def transactions_per_bank(df):
    is_cross = df['From Bank'] != df['To Bank']
    intra = df.loc[~is_cross, 'From Bank'].value_counts()
    cross = pd.concat([df.loc[is_cross, 'From Bank'], df.loc[is_cross, 'To Bank']]).value_counts()
    return intra.add(cross, fill_value=0)


def split_stats(df):
    is_illicit = df['Is Laundering'] == 1
    is_cross = df['From Bank'] != df['To Bank']
    per_bank = transactions_per_bank(df)
    days = df['Timestamp'].floordiv(86400)
    n_primary, n_days = days[days < PRIMARY_DAYS].nunique(), days.nunique()
    pattern_counts = df[is_illicit].groupby('Pattern').size()

    return {
        'Transactions':        f"{len(df):,}",
        'Accounts':            f"{len(set(df['from_id']) | set(df['to_id'])):,}",
        'Banks':               f"{len(set(df['From Bank']) | set(df['To Bank'])):,}",
        'Days':                f"{n_primary} ({n_days})" if n_days > n_primary else f"{n_days}",
        'Illicit transactions': f"{int(is_illicit.sum()):,}",
        'Illicit rate (%)':    f"{100 * is_illicit.mean():.2f}",
        'Laundering attempts': f"{df.loc[is_illicit & (df['AttemptID'] >= 0), 'AttemptID'].nunique():,}",
        'Transactions per bank, median': f"{per_bank.median():,.0f}",
        'Transactions per bank, p90':    f"{per_bank.quantile(0.9):,.0f}",
        'Transactions per bank, max':    f"{per_bank.max():,.0f}",
        'Cross-bank, of all transactions (%)':     f"{100 * is_cross.mean():.2f}",
        'Cross-bank, of illicit transactions (%)': f"{100 * is_cross[is_illicit].mean():.2f}",
        **{rf"\quad {PATTERN_NAMES[pid]}": f"{int(pattern_counts.get(pid, 0)):,}" for pid in PATTERN_ORDER},
    }


stats = {name: split_stats(df) for name, df in splits.items()}

# Rows are statistics, columns are splits. The pattern block gets a header row.
summary = pd.DataFrame(stats)
pattern_rows = [rf"\quad {PATTERN_NAMES[pid]}" for pid in PATTERN_ORDER]
header = pd.DataFrame({c: [''] for c in summary.columns}, index=['Illicit transactions by pattern'])
summary = pd.concat([summary.drop(index=pattern_rows), header, summary.loc[pattern_rows]])
summary = summary.reset_index().rename(columns={'index': ''})

group_sizes = [4, 3, 3, 2, 1 + len(PATTERN_ORDER)]
assert sum(group_sizes) == len(summary)


# %% ========== Output ==========

CSV_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(CSV_DIR / 'data_summary.csv', index=False)
df_to_latex_table(summary, TABLES_DIR / 'data_summary.tex', row_group_sizes=group_sizes)
print(summary.to_string(index=False))

# %%
