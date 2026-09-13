"""Bank-level and inter/intra-bank statistics — single source of truth.

Replaces the three duplicated (and drifted — range(10)/range(13)/range(15) for
the same currency dimension) stats-building loops that used to live directly in
bank_analysis.py. compute_bank_stats() and compute_inter_intra_stats() are each
called exactly once by the orchestrator; every heterogeneity subsection module
derives the columns it needs from the resulting DataFrames.

IMPORTANT — Pattern/Currency/Payment come from raw_df, not edge_attr:
The original (pre-refactor) bank_analysis.py read laundering pattern, currency,
and payment format straight out of party.data[data_str]['df'].edge_attr at
fixed column indices (2/3/4). That was wrong: edge_attr never contains Pattern
at all (including it would leak the label to the model), and with the default
ibm_fe=False config its actual 7-column layout is
[Timestamp, Amount Sent, Amount Received, Sent Currency, Received Currency,
Payment Format, is_currency_exchange] — none of which lines up with what the
original script assumed. The fix here reads Pattern/Currency/Payment/Amount/
Timestamp directly from the raw transactions CSV (raw_df), sliced per party via
party.indices[f'{split}_indices'] — the same row-position bookkeeping
analysis_functions.py's reconstruct_test_raw_df/reconstruct_train_raw_df and
enrich_lv_with_raw already rely on elsewhere in this codebase.
"""

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import PATTERN_NAMES

# Fixed, known vocabularies for this dataset's currency/payment-format codes
# (confirmed against scaler_encoders['encoder_currency'/'encoder_payment_format']
# categories_, which have exactly 15/7 entries) — not derived from data, since
# these are a small closed set baked into the synthetic AML dataset generator.
CURRENCY_NAMES = {
    i: name for i, name in enumerate([
        'US Dollar', 'Bitcoin', 'Euro', 'AUD', 'Yuan', 'Rupee', 'Yen',
        'MXN Peso', 'UK Pound', 'Ruble', 'CAD', 'CHF', 'BRL', 'SAR', 'ILS',
    ])
}
PAYMENT_NAMES = {
    i: name for i, name in enumerate([
        'Reinvestment', 'Cheque', 'Credit Card', 'ACH', 'Cash', 'Wire', 'Bitcoin',
    ])
}

_SPLIT_KEY = {'train_data': 'train', 'vali_data': 'vali', 'test_data': 'test'}


def compute_bank_stats(parties, df, raw_df, eval_mode, data_str='train_data'):
    """One row per bank: n_nodes, n_edges, n_fraud, fraud_rate, total_fraud_rate,
    per-pattern/currency/payment-format counts (raw code-indexed columns, e.g.
    'Pattern_0'), and amount/timestamp summary stats.

    raw_df: the full transactions CSV (analysis_functions.load_raw_df(size, ir)).
    eval_mode: only 'system' is supported — party.indices[...] are literal
    raw_df row positions in that mode. 'comparable' mode re-numbers indices per
    split and would need reconstruct_train_raw_df/reconstruct_test_raw_df
    instead of a direct raw_df.loc[...] lookup (not implemented here since
    this heterogeneity analysis is specifically the system-evaluation one —
    see data_heterogeneity_holder.tex in the writing repo).

    Returns:
        stats_df, pattern_cols, currency_cols, payment_cols
        (the last three are the human-readable column name lists, already
        present in stats_df after the rename below).
    """
    if eval_mode != 'system':
        raise NotImplementedError(
            "compute_bank_stats only supports eval_mode='system'. 'comparable' mode "
            "re-numbers party.indices per split and needs reconstruct_train_raw_df/"
            "reconstruct_test_raw_df (analysis_functions.py) instead of a direct "
            "raw_df.loc[...] lookup."
        )

    split = _SPLIT_KEY[data_str]
    n_laundering = np.sum(df['regular_data'][data_str]['x']['Is Laundering'])

    rows = []
    for bank_id, party in parties.items():
        data = party.data[data_str]['df']
        idx = party.indices[f'{split}_indices']
        party_raw = raw_df.loc[idx]

        y = data.y.numpy()
        # Sanity check: if this doesn't line up, the indices assumption above
        # is wrong and everything below would be silently mislabeled again —
        # fail loudly instead.
        if not np.array_equal(party_raw['Is Laundering'].to_numpy(), y):
            raise ValueError(
                f"Bank {bank_id}: raw_df.loc[party.indices[...]] does not line up with "
                f"party.data[{data_str!r}]['df'].y — the raw_df/party.indices join is wrong."
            )

        currency_dist = party_raw['Received Currency'].value_counts()
        payment_dist = party_raw['Payment Format'].value_counts()
        laundering_patterns = party_raw['Pattern'].value_counts()
        n_fraud = int(y.sum())

        rows.append({
            'bank_id': bank_id,
            'n_nodes': data.x.shape[0],
            'n_edges': data.y.shape[0],
            'n_fraud': n_fraud,
            'fraud_rate': y.mean() * 100,
            'total_fraud_rate': n_fraud / n_laundering * 100,
            **{f'Pattern_{i}': int(laundering_patterns.get(i, 0)) for i in PATTERN_NAMES},
            'amount_received_mean': party_raw['Amount Received'].mean(),
            'amount_received_std': party_raw['Amount Received'].std(),
            'amount_received_median': party_raw['Amount Received'].median(),
            'timestamp_mean': party_raw['Timestamp'].mean(),
            'timestamp_std': party_raw['Timestamp'].std(),
            **{f'Currency_{i}': int(currency_dist.get(i, 0)) for i in CURRENCY_NAMES},
            **{f'Payment_{i}': int(payment_dist.get(i, 0)) for i in PAYMENT_NAMES},
        })

    stats_df = pd.DataFrame(rows)

    pattern_rename = {f'Pattern_{i}': name for i, name in PATTERN_NAMES.items()}
    currency_rename = {f'Currency_{i}': name for i, name in CURRENCY_NAMES.items()}
    payment_rename = {f'Payment_{i}': name for i, name in PAYMENT_NAMES.items()}
    stats_df = stats_df.rename(columns={**pattern_rename, **currency_rename, **payment_rename})

    # Pattern 0 is 'NONE' (legitimate) — exclude from the laundering-pattern-mix columns.
    pattern_cols = [name for i, name in PATTERN_NAMES.items() if i != 0]
    currency_cols = list(CURRENCY_NAMES.values())
    payment_cols = list(PAYMENT_NAMES.values())

    return stats_df, pattern_cols, currency_cols, payment_cols


def compute_inter_intra_stats(train_df, party_banks):
    """One row per bank: intra-/inter-bank edge counts, proportions, fraud rates.

    Vectorized via groupby rather than filtering the full (multi-million-row)
    train_df once per bank — the original per-bank boolean-mask loop was
    O(n_banks * n_rows), which is the dominant cost at ~1000-bank scale.
    """
    intra_mask = train_df['From Bank'] == train_df['To Bank']
    intra_df = train_df[intra_mask]
    inter_df = train_df[~intra_mask]

    intra_counts = intra_df.groupby('From Bank').size()
    intra_fraud = intra_df.groupby('From Bank')['Is Laundering'].sum()

    # Each inter-bank edge touches two distinct banks — count it once for each.
    inter_long = pd.concat([
        inter_df[['From Bank', 'Is Laundering']].rename(columns={'From Bank': 'bank_id'}),
        inter_df[['To Bank', 'Is Laundering']].rename(columns={'To Bank': 'bank_id'}),
    ], ignore_index=True)
    inter_counts = inter_long.groupby('bank_id').size()
    inter_fraud = inter_long.groupby('bank_id')['Is Laundering'].sum()

    rows = []
    for bank_id in party_banks:
        n_intra = int(intra_counts.get(bank_id, 0))
        n_inter = int(inter_counts.get(bank_id, 0))
        n_total = n_intra + n_inter
        if n_total == 0:
            continue

        n_fraud = int(intra_fraud.get(bank_id, 0) + inter_fraud.get(bank_id, 0))

        rows.append({
            'bank_id': bank_id,
            'n_total': n_total,
            'n_intra': n_intra,
            'n_inter': n_inter,
            'intra_pct': n_intra / n_total * 100,
            'inter_pct': n_inter / n_total * 100,
            'n_fraud': n_fraud,
            'fraud_rate': n_fraud / n_total * 100,
            'intra_fraud_rate': intra_fraud.get(bank_id, 0) / n_intra * 100 if n_intra > 0 else np.nan,
            'inter_fraud_rate': inter_fraud.get(bank_id, 0) / n_inter * 100 if n_inter > 0 else np.nan,
        })

    return pd.DataFrame(rows)
