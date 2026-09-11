"""Bank-level and inter/intra-bank statistics — single source of truth.

Replaces the three duplicated (and drifted — range(10)/range(13)/range(15) for
the same currency dimension) stats-building loops that used to live directly in
bank_analysis.py. compute_bank_stats() and compute_inter_intra_stats() are each
called exactly once by the orchestrator; every heterogeneity subsection module
derives the columns it needs from the resulting DataFrames.
"""

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import PATTERN_NAMES


def derive_category_labels(scaler_encoders, encoder_key, observed_codes, prefix):
    """code -> label, derived from scaler_encoders[encoder_key].categories_[0].

    observed_codes: the full set of codes actually seen across ALL parties (not
    just one bank — a single bank may not cover the full code range).

    scaler_encoders is None when data_parser.ibm_fe=True (IBM's own feature
    engineering path, which doesn't build a OneHotEncoder) — falls back to
    generic f'{prefix}_{code}' labels in that case, and also if the encoder's
    category count doesn't match the codes actually observed
    (fail loud with a warning, never silently mislabel).
    """
    n_observed = (max(observed_codes) + 1) if observed_codes else 0

    if scaler_encoders is not None and encoder_key in scaler_encoders:
        categories = list(scaler_encoders[encoder_key].categories_[0])
        if len(categories) >= n_observed:
            return {i: str(categories[i]) for i in range(len(categories))}
        print(f"[WARN] {encoder_key}: {len(categories)} categories from the encoder but "
              f"codes up to {n_observed - 1} observed in the data — falling back to "
              f"generic '{prefix}_<code>' labels rather than risk a wrong mapping.")

    return {i: f'{prefix}_{i}' for i in range(n_observed)}


def _counts_by_code(col):
    """Vectorized replacement for Counter(col.tolist()) — np.unique on the raw
    float32 array is far faster than converting millions of elements to Python
    floats and hashing them one at a time (this was the actual bottleneck the
    original Counter(edge_attr[:, i].tolist()) pattern had at ~1000-party
    scale). Returns {code: count}."""
    codes, counts = np.unique(col.astype(np.int32), return_counts=True)
    return dict(zip(codes.tolist(), counts.tolist()))


def compute_bank_stats(parties, df, scaler_encoders, data_str='train_data'):
    """One row per bank: n_nodes, n_edges, n_fraud, fraud_rate, total_fraud_rate,
    per-pattern/currency/payment-format counts (raw code-indexed columns, e.g.
    'Pattern_0'), and amount/timestamp summary stats.

    Returns:
        stats_df, pattern_cols, currency_cols, payment_cols
        (the last three are the human-readable column name lists, already
        present in stats_df after the rename below).
    """
    n_laundering = np.sum(df['regular_data'][data_str]['x']['Is Laundering'])

    # Single pass: process one party's edge_attr at a time (never retaining more
    # than one in memory simultaneously — with ~1000+ parties, caching all of
    # them at once is a real memory cost, not just a style choice). What IS
    # retained across the loop is small: one {code: count} dict per bank per
    # column (a few dozen entries each), not the raw per-edge data.
    per_bank_dists = {}
    per_bank_edge_stats = {}
    max_currency_code = -1
    max_payment_code = -1
    for bank_id, party in parties.items():
        edge_attr = party.data[data_str]['df'].edge_attr.numpy()
        currency_dist = _counts_by_code(edge_attr[:, 2])
        payment_dist = _counts_by_code(edge_attr[:, 3])
        laundering_patterns = _counts_by_code(edge_attr[:, 4])
        per_bank_dists[bank_id] = (currency_dist, payment_dist, laundering_patterns)
        per_bank_edge_stats[bank_id] = {
            'amount_received_mean': edge_attr[:, 1].mean(),
            'amount_received_std': edge_attr[:, 1].std(),
            'amount_received_median': np.median(edge_attr[:, 1]),
            'timestamp_mean': edge_attr[:, 0].mean(),
            'timestamp_std': edge_attr[:, 0].std(),
        }
        if currency_dist:
            max_currency_code = max(max_currency_code, max(currency_dist))
        if payment_dist:
            max_payment_code = max(max_payment_code, max(payment_dist))

    currency_labels = derive_category_labels(
        scaler_encoders, 'encoder_currency', range(int(max_currency_code) + 1), 'Currency')
    payment_labels = derive_category_labels(
        scaler_encoders, 'encoder_payment_format', range(int(max_payment_code) + 1), 'Payment')

    rows = []
    for bank_id, party in parties.items():
        data = party.data[data_str]['df']
        currency_dist, payment_dist, laundering_patterns = per_bank_dists[bank_id]
        edge_stats = per_bank_edge_stats[bank_id]
        y = data.y.numpy()
        n_fraud = y.sum()

        rows.append({
            'bank_id': bank_id,
            'n_nodes': data.x.shape[0],
            'n_edges': data.y.shape[0],
            'n_fraud': n_fraud,
            'fraud_rate': y.mean() * 100,
            'total_fraud_rate': n_fraud / n_laundering * 100,
            **{f'Pattern_{i}': laundering_patterns.get(i, 0) for i in PATTERN_NAMES},
            **edge_stats,
            **{f'Currency_{i}': currency_dist.get(i, 0) for i in currency_labels},
            **{f'Payment_{i}': payment_dist.get(i, 0) for i in payment_labels},
        })

    stats_df = pd.DataFrame(rows)

    pattern_rename = {f'Pattern_{i}': name for i, name in PATTERN_NAMES.items()}
    currency_rename = {f'Currency_{i}': name for i, name in currency_labels.items()}
    payment_rename = {f'Payment_{i}': name for i, name in payment_labels.items()}
    stats_df = stats_df.rename(columns={**pattern_rename, **currency_rename, **payment_rename})

    # Pattern 0 is 'NONE' (legitimate) — exclude from the laundering-pattern-mix columns.
    pattern_cols = [name for i, name in PATTERN_NAMES.items() if i != 0]
    currency_cols = list(currency_labels.values())
    payment_cols = list(payment_labels.values())

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

        rows.append({
            'bank_id': bank_id,
            'n_total': n_total,
            'n_intra': n_intra,
            'n_inter': n_inter,
            'intra_pct': n_intra / n_total * 100,
            'inter_pct': n_inter / n_total * 100,
            'intra_fraud_rate': intra_fraud.get(bank_id, 0) / n_intra * 100 if n_intra > 0 else np.nan,
            'inter_fraud_rate': inter_fraud.get(bank_id, 0) / n_inter * 100 if n_inter > 0 else np.nan,
        })

    return pd.DataFrame(rows)
