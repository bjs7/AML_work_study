"""Inter-bank vs. Intra-bank — what fraction of each bank's transactions cross
a bank boundary, and does fraud rate differ between intra- and inter-bank
transactions?

Matches the "Detection of Heterogeneity - System Evaluation: Inter-bank vs.
Intra-bank" subsection of data_heterogeneity_holder.tex (writing repo).
"""

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import df_to_latex_table

from plotting import FONTSIZE, thousands_fmt, savefig, CSV_DIR, TABLES_DIR


def build_inter_intra_table(ii_df, out_name='inter_intra_summary'):
    """Mean intra-/inter-bank proportion and fraud rate across banks — the
    numbers the 'Inter-bank vs. Intra-bank' paragraph needs to cite."""
    rows = [{
        'Metric': 'Intra-bank edge share (%)',
        'Mean': round(ii_df['intra_pct'].mean(), 2), 'Median': round(ii_df['intra_pct'].median(), 2),
    }, {
        'Metric': 'Inter-bank edge share (%)',
        'Mean': round(ii_df['inter_pct'].mean(), 2), 'Median': round(ii_df['inter_pct'].median(), 2),
    }, {
        'Metric': 'Intra-bank fraud rate (%)',
        'Mean': round(ii_df['intra_fraud_rate'].mean(), 2), 'Median': round(ii_df['intra_fraud_rate'].median(), 2),
    }, {
        'Metric': 'Inter-bank fraud rate (%)',
        'Mean': round(ii_df['inter_fraud_rate'].mean(), 2), 'Median': round(ii_df['inter_fraud_rate'].median(), 2),
    }]
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / f'{out_name}.csv', index=False)
    df_to_latex_table(df, TABLES_DIR / f'{out_name}.tex')
    return df


def run(ii_df, filtered_ii_df=None, min_edges=None):
    """filtered_ii_df: optional bank-size-filtered subset (see
    quantity_skew.build_filtered_views) — when given, adds three "all banks
    vs. filtered" comparisons, since small/thin banks can dominate the
    unfiltered intra/inter proportion and fraud-rate plots."""
    table = build_inter_intra_table(ii_df)
    print("\nInter-bank vs intra-bank summary:")
    print(table.to_string(index=False))

    # Intra-/inter-bank edge-share distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ii_df['intra_pct'], bins=50, edgecolor='black')
    axes[0].set_xlabel('Intra-bank edge proportion (%)', fontsize=FONTSIZE)
    axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
    axes[1].hist(ii_df['inter_pct'], bins=50, edgecolor='black')
    axes[1].set_xlabel('Inter-bank edge proportion (%)', fontsize=FONTSIZE)
    axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('intra_inter_proportion_hist.pdf')

    # Bank size vs. intra-bank proportion
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(ii_df['n_total'], ii_df['intra_pct'], alpha=0.5)
    ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
    ax.set_ylabel('Intra-bank proportion (%)', fontsize=FONTSIZE)
    ax.xaxis.set_major_formatter(thousands_fmt)
    plt.tight_layout()
    savefig('intra_proportion_scatter.pdf')

    # Fraud rate: intra- vs inter-bank
    vals_intra = ii_df['intra_fraud_rate'].dropna()
    vals_inter = ii_df['inter_fraud_rate'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(vals_intra, bins=50, edgecolor='black', alpha=0.7, label='Intra-bank')
    axes[0].hist(vals_inter, bins=50, edgecolor='black', alpha=0.7, label='Inter-bank')
    axes[0].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
    axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
    axes[0].legend()
    axes[0].set_title('Fraud rate distribution', fontsize=FONTSIZE)

    axes[1].scatter(ii_df['intra_fraud_rate'], ii_df['inter_fraud_rate'], alpha=0.5)
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
    axes[1].set_xlabel('Intra-bank fraud rate (%)', fontsize=FONTSIZE)
    axes[1].set_ylabel('Inter-bank fraud rate (%)', fontsize=FONTSIZE)
    axes[1].set_title('Intra vs inter fraud rate per bank', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('intra_inter_fraud_comparison.pdf')

    # All banks vs. banks with >= min_edges edges
    if filtered_ii_df is not None:
        filt_label = f'Banks with >= {min_edges:,} edges'

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(ii_df['intra_pct'], bins=50, edgecolor='black')
        axes[0].set_xlabel('Intra-bank edge proportion (%)', fontsize=FONTSIZE)
        axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
        axes[0].set_title('All banks', fontsize=FONTSIZE)
        axes[1].hist(filtered_ii_df['intra_pct'], bins=50, edgecolor='black')
        axes[1].set_xlabel('Intra-bank edge proportion (%)', fontsize=FONTSIZE)
        axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
        axes[1].set_title(filt_label, fontsize=FONTSIZE)
        plt.tight_layout()
        savefig('intra_proportion_hist_filtered_comparison.pdf')

        filt_vals_intra = filtered_ii_df['intra_fraud_rate'].dropna()
        filt_vals_inter = filtered_ii_df['inter_fraud_rate'].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(vals_intra, bins=50, edgecolor='black', alpha=0.7, label='Intra-bank')
        axes[0].hist(vals_inter, bins=50, edgecolor='black', alpha=0.7, label='Inter-bank')
        axes[0].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
        axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
        axes[0].legend()
        axes[0].set_title('All banks', fontsize=FONTSIZE)
        axes[1].hist(filt_vals_intra, bins=50, edgecolor='black', alpha=0.7, label='Intra-bank')
        axes[1].hist(filt_vals_inter, bins=50, edgecolor='black', alpha=0.7, label='Inter-bank')
        axes[1].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
        axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
        axes[1].legend()
        axes[1].set_title(filt_label, fontsize=FONTSIZE)
        plt.suptitle('Fraud rate distribution: all vs. filtered', fontsize=FONTSIZE)
        plt.tight_layout()
        savefig('intra_inter_fraud_hist_filtered_comparison.pdf')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(ii_df['intra_fraud_rate'], ii_df['inter_fraud_rate'], alpha=0.5)
        lim = max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])
        axes[0].plot([0, lim], [0, lim], 'r--', alpha=0.5)
        axes[0].set_xlabel('Intra-bank fraud rate (%)', fontsize=FONTSIZE)
        axes[0].set_ylabel('Inter-bank fraud rate (%)', fontsize=FONTSIZE)
        axes[0].set_title('All banks', fontsize=FONTSIZE)
        axes[1].scatter(filtered_ii_df['intra_fraud_rate'], filtered_ii_df['inter_fraud_rate'], alpha=0.5)
        lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
        axes[1].plot([0, lim], [0, lim], 'r--', alpha=0.5)
        axes[1].set_xlabel('Intra-bank fraud rate (%)', fontsize=FONTSIZE)
        axes[1].set_ylabel('Inter-bank fraud rate (%)', fontsize=FONTSIZE)
        axes[1].set_title(filt_label, fontsize=FONTSIZE)
        plt.suptitle('Intra vs inter fraud rate per bank: all vs. filtered', fontsize=FONTSIZE)
        plt.tight_layout()
        savefig('intra_inter_fraud_scatter_filtered_comparison.pdf')

    return table
