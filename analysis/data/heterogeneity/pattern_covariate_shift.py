"""Laundering-pattern and Feature Covariate Shift — do banks see different
mixes of laundering patterns, currencies, and payment formats, and does the
amount/timestamp distribution shift between banks?

Matches the "Detection of Heterogeneity - System Evaluation: Laundering-pattern
and Feature Covariate Shift" subsection of data_heterogeneity_holder.tex
(writing repo).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import df_to_latex_table

from plotting import (
    FONTSIZE, savefig, CSV_DIR, TABLES_DIR,
    plot_proportion_heatmap, plot_proportion_stacked_bar,
)
from stats import CURRENCY_NAMES, _SPLIT_KEY

UPPER_PERCENTILE = 99  # outlier cutoff for the per-currency amount histograms


def build_pattern_covariate_table(stats_df, pattern_cols, currency_cols, payment_cols,
                                   out_name='pattern_covariate_summary'):
    """CV per pattern/currency/payment-format across banks (how unevenly is
    each category distributed?), plus mean pattern-mix entropy per bank —
    the numbers the 'Laundering-pattern and Feature Covariate Shift'
    paragraph needs to cite."""
    rows = []
    for label, cols in [('Pattern', pattern_cols), ('Currency', currency_cols), ('Payment format', payment_cols)]:
        counts = stats_df[cols].fillna(0)
        for col in cols:
            mean = counts[col].mean()
            cv = counts[col].std() / mean if mean > 0 else float('nan')
            rows.append({'Category': label, 'Value': col, 'CV': round(cv, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / f'{out_name}.csv', index=False)
    df_to_latex_table(df, TABLES_DIR / f'{out_name}.tex')
    return df


def _bank_entropy(stats_df, cols):
    counts = stats_df[cols].fillna(0)
    proportions = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return [entropy(row) for _, row in proportions.iterrows()]


def run(stats_df, pattern_cols, currency_cols, payment_cols, parties, raw_df, eval_mode,
        data_str='train_data', filtered_stats_df=None, min_edges=None):
    """filtered_stats_df: optional bank-size-filtered subset (see
    quantity_skew.build_filtered_views) — when given, also saves the pattern
    heatmap restricted to banks with >= min_edges edges, since small/thin
    banks can dominate the pattern mix in the unfiltered heatmap.

    raw_df/eval_mode: needed for the per-currency amount breakdown below,
    which (like compute_bank_stats) must read Currency/Amount from the raw
    transactions CSV via party.indices, not from edge_attr — see stats.py's
    module docstring for why. Only eval_mode='system' is supported."""
    if eval_mode != 'system':
        raise NotImplementedError("pattern_covariate_shift.run only supports eval_mode='system'.")
    split = _SPLIT_KEY[data_str]
    table = build_pattern_covariate_table(stats_df, pattern_cols, currency_cols, payment_cols)
    print("\nPattern/currency/payment covariate-shift summary (CV across banks):")
    print(table.to_string(index=False))

    # --- Laundering patterns ---
    fig = plot_proportion_heatmap(stats_df, pattern_cols, 'Pattern type')
    savefig('pattern_heatmap.pdf')

    if filtered_stats_df is not None:
        fig = plot_proportion_heatmap(filtered_stats_df, pattern_cols, 'Pattern type')
        fig.suptitle(f'Banks with >= {min_edges:,} edges', fontsize=FONTSIZE)
        savefig('pattern_heatmap_filtered.pdf')

    fig = plot_proportion_stacked_bar(stats_df, pattern_cols, legend_title='Pattern', N=50)
    savefig('pattern_stacked_bar.pdf')

    bank_entropy = _bank_entropy(stats_df, pattern_cols)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(stats_df['n_edges'], bank_entropy, alpha=0.5)
    ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
    ax.set_ylabel('Pattern entropy', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('pattern_entropy_scatter.pdf')

    # --- Amount / timestamp distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].hist(stats_df['amount_received_mean'], bins=50, edgecolor='black')
    axes[0].set_xlabel('Mean amount received', fontsize=FONTSIZE)
    axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
    axes[1].hist(stats_df['amount_received_std'], bins=50, edgecolor='black')
    axes[1].set_xlabel('Std of amount received', fontsize=FONTSIZE)
    axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('amount_received_hist.pdf')

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(stats_df['amount_received_mean'], stats_df['timestamp_mean'], alpha=0.6)
    ax.set_xlabel('Mean amount received', fontsize=FONTSIZE)
    ax.set_ylabel('Mean timestamp', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('amount_vs_timestamp.pdf')

    # --- Currency / payment format ---
    fig = plot_proportion_heatmap(stats_df, currency_cols, 'Received currency type')
    savefig('currency_heatmap.pdf')
    fig = plot_proportion_stacked_bar(stats_df, currency_cols, legend_title='Currency')
    savefig('currency_stacked_bar.pdf')

    fig = plot_proportion_heatmap(stats_df, payment_cols, 'Payment format type')
    savefig('payment_heatmap.pdf')
    fig = plot_proportion_stacked_bar(stats_df, payment_cols, legend_title='Payment format')
    savefig('payment_stacked_bar.pdf')

    # --- Amount by currency (needs each party's raw transactions, not just
    # the aggregated stats_df — read from raw_df via party.indices, same as
    # compute_bank_stats, not from edge_attr) ---
    all_amounts_by_currency = {name: [] for name in currency_cols}
    party_currency_means = []
    for bank_id, party in parties.items():
        party_raw = raw_df.loc[party.indices[f'{split}_indices']]
        row = {'bank_id': bank_id}
        for code, name in CURRENCY_NAMES.items():
            amounts = party_raw.loc[party_raw['Received Currency'] == code, 'Amount Received']
            if len(amounts) > 0:
                all_amounts_by_currency[name].extend(amounts.tolist())
                row[name] = amounts.mean()
            else:
                row[name] = np.nan
        party_currency_means.append(row)
    currency_mean_df = pd.DataFrame(party_currency_means)

    summary_rows = []
    for currency in currency_cols:
        vals = all_amounts_by_currency[currency]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        summary_rows.append({'currency': currency, 'n': len(arr), 'mean': arr.mean(),
                              'std': arr.std(), 'max': arr.max()})
    summary_df = pd.DataFrame(summary_rows).set_index('currency')
    print("\nPer-currency amount summary (full dataset):")
    print(summary_df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))
    x = np.arange(len(summary_df))
    for ax, col, title in zip(axes, ['mean', 'std', 'max'], ['Mean', 'Std', 'Max']):
        ax.bar(x, summary_df[col])
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df.index, rotation=90, ha='center', fontsize=FONTSIZE - 3)
        ax.set_ylabel('Amount received', fontsize=FONTSIZE - 1)
        ax.set_title(title, fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('currency_amount_summary.pdf')

    currencies_with_data = [c for c in currency_cols if len(all_amounts_by_currency[c]) > 0]
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
    savefig('currency_amount_full_hist.pdf')

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
    savefig('currency_amount_perbank_hist.pdf')

    return table
