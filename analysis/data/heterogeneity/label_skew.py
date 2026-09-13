"""Label Distribution Skew — how unevenly is fraud (Is Laundering) distributed
across banks? Fraud counts vary far more than fraud rates: a FedAvg scheme
that weights updates equally per bank gives banks with 50 fraud cases the
same influence as banks with 5000, a form of quantity skew compounding label
skew that can destabilize training.

Matches the "Detection of Heterogeneity - System Evaluation: Label Distribution
Skew" subsection of data_heterogeneity_holder.tex (writing repo).
"""

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import df_to_latex_table

from plotting import FONTSIZE, savefig, CSV_DIR, TABLES_DIR


def build_label_skew_table(stats_df, out_name='label_skew_summary'):
    """CV and summary stats for fraud count / fraud rate across banks — the
    numbers the 'Label Distribution Skew' paragraph needs to cite."""
    rows = []
    for col in ['n_fraud', 'fraud_rate']:
        vals = stats_df[col]
        cv = vals.std() / vals.mean() if vals.mean() > 0 else float('nan')
        rows.append({
            'Metric': col,
            'Min': round(vals.min(), 2),
            'Median': round(vals.median(), 2),
            'Mean': round(vals.mean(), 2),
            'Max': round(vals.max(), 2),
            'CV': round(cv, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / f'{out_name}.csv', index=False)
    df_to_latex_table(df, TABLES_DIR / f'{out_name}.tex')
    return df


def run(stats_df, filtered_stats_df=None, min_edges=None):
    """filtered_stats_df: optional bank-size-filtered subset (see
    quantity_skew.build_filtered_views) — when given, adds an "all banks vs.
    filtered" fraud-rate-scatter comparison, since small/thin banks can
    dominate the unfiltered scatter."""
    table = build_label_skew_table(stats_df)
    print("\nLabel skew summary (fraud count / fraud rate across banks):")
    print(table.to_string(index=False))

    # Fraud count / fraud rate distribution
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].hist(stats_df['n_fraud'], bins=50, edgecolor='black')
    axes[0].set_xlabel('Number of fraud transactions', fontsize=FONTSIZE)
    axes[0].set_ylabel('Number of banks', fontsize=FONTSIZE)
    axes[1].hist(stats_df['fraud_rate'], bins=50, edgecolor='black')
    axes[1].set_xlabel('Fraud rate (%)', fontsize=FONTSIZE)
    axes[1].set_ylabel('Number of banks', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('fraud_hist.pdf')

    # Sorted bars: edge count, fraud count, fraud rate (descending by edge count)
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5))
    sorted_df = stats_df.sort_values('n_edges', ascending=False)
    axes[0].bar(range(len(sorted_df)), sorted_df['n_edges'], width=1.0)
    axes[0].set_ylabel('Number of edges', fontsize=FONTSIZE)
    axes[1].bar(range(len(sorted_df)), sorted_df['n_fraud'], width=1.0)
    axes[1].set_ylabel('Fraud cases', fontsize=FONTSIZE)
    axes[2].bar(range(len(sorted_df)), sorted_df['fraud_rate'], width=1.0)
    axes[2].set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
    axes[2].set_xlabel('Banks (sorted by edge count)', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('fraud_sorted_bars.pdf')

    # Fraud rate vs. bank size (bubble = fraud count)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(stats_df['n_edges'], stats_df['fraud_rate'], s=stats_df['n_fraud'], alpha=0.6)
    ax.set_xlabel('Number of edges', fontsize=FONTSIZE)
    ax.set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
    ax.set_title('Bubble size = fraud count', fontsize=FONTSIZE)
    plt.tight_layout()
    savefig('fraud_scatter.pdf')

    # All banks vs. banks with >= min_edges edges — does dropping small/thin
    # banks change the size-vs-fraud-rate story?
    if filtered_stats_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(stats_df['n_edges'], stats_df['fraud_rate'], s=stats_df['n_fraud'], alpha=0.6)
        axes[0].set_xlabel('Number of edges', fontsize=FONTSIZE)
        axes[0].set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
        axes[0].set_title('All banks', fontsize=FONTSIZE)
        axes[1].scatter(filtered_stats_df['n_edges'], filtered_stats_df['fraud_rate'],
                         s=filtered_stats_df['n_fraud'], alpha=0.6)
        axes[1].set_xlabel('Number of edges', fontsize=FONTSIZE)
        axes[1].set_ylabel('Fraud rate (%)', fontsize=FONTSIZE)
        axes[1].set_title(f'Banks with >= {min_edges:,} edges', fontsize=FONTSIZE)
        plt.suptitle('Edges vs fraud rate (bubble size = fraud count)', fontsize=FONTSIZE)
        plt.tight_layout()
        savefig('fraud_scatter_filtered_comparison.pdf')

    return table
