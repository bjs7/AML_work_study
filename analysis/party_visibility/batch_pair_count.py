# %%
"""Empirical count of unique (From Bank, To Bank) pairs per batch.

Answers: how many unique bank pairs appear in a typical lazy_link_neighbor batch?
This determines whether the per-pair GNN cost of the proper-merge experiment is
feasible under comparable and system evaluation settings.

Run each cell in order. Takes < 1 min locally on the small/HI dataset.
"""
import sys
sys.path.insert(0, '/home/nam_07/projects/AML_work_study/AML_work_study')

import numpy as np
import pandas as pd

CSV_PATH = '/home/nam_07/projects/AML_work_study/formatted_transactions_small_HI.csv'
BATCH_SIZE = 8192
N_SAMPLE_BATCHES = 200   # how many random batches to draw
SEED = 42

# %%
# Load raw transactions and do the 60/20/20 temporal split
df = pd.read_csv(CSV_PATH)
df = df.sort_values('Timestamp').reset_index(drop=True)

n = len(df)
train_end = int(n * 0.6)
vali_end  = int(n * 0.8)

train_df = df.iloc[:train_end]
print(f"Total transactions : {n:,}")
print(f"Train transactions : {len(train_df):,}")
print(f"Unique banks total : {pd.concat([df['From Bank'], df['To Bank']]).nunique():,}")
print(f"Unique banks train : {pd.concat([train_df['From Bank'], train_df['To Bank']]).nunique():,}")

# %%
# --- Comparable evaluation bank set ---
# Banks that have at least one laundering transaction in both the train AND
# test splits (mirrors the individual_banks selection in relevant_banks.py).

vali_df = df.iloc[train_end:vali_end]
test_df  = df.iloc[vali_end:]

def banks_with_both_labels(split):
    """Banks that have at least one laundering AND one non-laundering transaction."""
    pos = set(split[split['Is Laundering'] == 1]['From Bank']) | set(split[split['Is Laundering'] == 1]['To Bank'])
    neg = set(split[split['Is Laundering'] == 0]['From Bank']) | set(split[split['Is Laundering'] == 0]['To Bank'])
    return pos & neg

def banks_with_laundering(split):
    ldf = split[split['Is Laundering'] == 1]
    return set(ldf['From Bank'].unique()) | set(ldf['To Bank'].unique())

def banks_with_any_data(split):
    return set(split['From Bank'].unique()) | set(split['To Bank'].unique())

# Mirrors relevant_banks.py individual_banks condition exactly:
#   train: both labels present  |  vali: laundering present  |  test: any data
train_both  = banks_with_both_labels(train_df)
vali_laund  = banks_with_laundering(vali_df)
test_any    = banks_with_any_data(test_df)

comparable_banks = train_both & vali_laund & test_any
print(f"Comparable banks (mirrors individual_banks logic): {len(comparable_banks):,}")

# Also apply comparable's cross-bank filter:
# keep only transactions where BOTH banks are in comparable_banks
comp_train = train_df[
    train_df['From Bank'].isin(comparable_banks) &
    train_df['To Bank'].isin(comparable_banks)
]
print(f"Comparable train transactions (both banks in set): {len(comp_train):,}")

# %%
# --- System evaluation bank set ---
# Approximation: banks active in training (appear in at least one transaction).
system_banks = set(train_df['From Bank'].unique()) | set(train_df['To Bank'].unique())
sys_train = train_df  # all train transactions
print(f"System banks (any train transaction): {len(system_banks):,}")
print(f"System train transactions            : {len(sys_train):,}")

# %%
# --- Helper: sample N_SAMPLE_BATCHES random batches and count unique bank pairs ---
def sample_batch_pair_counts(data, batch_size, n_batches, seed=SEED):
    """Sample n_batches random batches of size batch_size from data.
    Returns array of unique (From Bank, To Bank) pair counts per batch."""
    rng = np.random.default_rng(seed)
    counts = []
    idx = data.index.to_numpy()
    for _ in range(n_batches):
        chosen = rng.choice(idx, size=min(batch_size, len(idx)), replace=False)
        batch = data.loc[chosen, ['From Bank', 'To Bank']]
        unique_pairs = batch.drop_duplicates().shape[0]
        counts.append(unique_pairs)
    return np.array(counts)

# %%
# --- Run for comparable ---
comp_counts = sample_batch_pair_counts(comp_train, BATCH_SIZE, N_SAMPLE_BATCHES)

print("=== Comparable evaluation ===")
print(f"  Banks in set          : {len(comparable_banks):,}")
print(f"  Possible bank pairs   : {len(comparable_banks)**2:,}")
print(f"  Batch size            : {BATCH_SIZE:,}")
print(f"  Batches sampled       : {N_SAMPLE_BATCHES}")
print(f"  Unique pairs / batch  : mean={comp_counts.mean():.0f}  "
      f"median={np.median(comp_counts):.0f}  "
      f"p5={np.percentile(comp_counts,5):.0f}  "
      f"p95={np.percentile(comp_counts,95):.0f}  "
      f"max={comp_counts.max()}")

# %%
# --- Run for system ---
sys_counts = sample_batch_pair_counts(sys_train, BATCH_SIZE, N_SAMPLE_BATCHES)

print("=== System evaluation ===")
print(f"  Banks in set          : {len(system_banks):,}")
print(f"  Possible bank pairs   : {len(system_banks)**2:,}")
print(f"  Batch size            : {BATCH_SIZE:,}")
print(f"  Batches sampled       : {N_SAMPLE_BATCHES}")
print(f"  Unique pairs / batch  : mean={sys_counts.mean():.0f}  "
      f"median={np.median(sys_counts):.0f}  "
      f"p5={np.percentile(sys_counts,5):.0f}  "
      f"p95={np.percentile(sys_counts,95):.0f}  "
      f"max={sys_counts.max()}")

# %%
# --- Break down: unique individual banks per batch (not pairs) ---
def sample_unique_bank_counts(data, batch_size, n_batches, seed=SEED):
    rng = np.random.default_rng(seed)
    from_counts, to_counts, either_counts = [], [], []
    idx = data.index.to_numpy()
    for _ in range(n_batches):
        chosen = rng.choice(idx, size=min(batch_size, len(idx)), replace=False)
        batch = data.loc[chosen, ['From Bank', 'To Bank']]
        from_counts.append(batch['From Bank'].nunique())
        to_counts.append(batch['To Bank'].nunique())
        either_counts.append(pd.concat([batch['From Bank'], batch['To Bank']]).nunique())
    return np.array(from_counts), np.array(to_counts), np.array(either_counts)

print("=== Comparable — unique individual banks per batch ===")
fc, tc, ec = sample_unique_bank_counts(comp_train, BATCH_SIZE, N_SAMPLE_BATCHES)
print(f"  Unique From Bank  : mean={fc.mean():.0f}  median={np.median(fc):.0f}  max={fc.max()}")
print(f"  Unique To Bank    : mean={tc.mean():.0f}  median={np.median(tc):.0f}  max={tc.max()}")
print(f"  Unique either     : mean={ec.mean():.0f}  median={np.median(ec):.0f}  max={ec.max()}")

print("=== System — unique individual banks per batch ===")
fc, tc, ec = sample_unique_bank_counts(sys_train, BATCH_SIZE, N_SAMPLE_BATCHES)
print(f"  Unique From Bank  : mean={fc.mean():.0f}  median={np.median(fc):.0f}  max={fc.max()}")
print(f"  Unique To Bank    : mean={tc.mean():.0f}  median={np.median(tc):.0f}  max={tc.max()}")
print(f"  Unique either     : mean={ec.mean():.0f}  median={np.median(ec):.0f}  max={ec.max()}")
