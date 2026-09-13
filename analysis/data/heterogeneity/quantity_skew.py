"""Quantity Skew — how unevenly are nodes/edges distributed across banks?

Matches the "Detection of Heterogeneity - System Evaluation: Quantity Skew"
subsection of data_heterogeneity_holder.tex (writing repo).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')
from analysis_functions import df_to_latex_table

from plotting import FONTSIZE, thousands_fmt, savefig, CSV_DIR, TABLES_DIR

MIN_EDGES = 1000  # bank-size cutoff used by the "all banks vs. filtered" comparisons


def build_filtered_views(stats_df, ii_df, min_edges=MIN_EDGES):
    """Bank-size-filtered views of stats_df/ii_df, used by the 'all banks vs.
    banks with >= MIN_EDGES edges' comparisons in label_skew, pattern_covariate_shift,
    and inter_intra_bank — small/thin-data banks can dominate some plots, so it's
    worth checking whether the story changes once they're dropped."""
    filtered_stats_df = stats_df[stats_df['n_edges'] >= min_edges].copy()
    filtered_ii_df = ii_df[ii_df['n_total'] >= min_edges].copy()
    print(f"\nMIN_EDGES = {min_edges}")
    print(f"  Banks kept: {len(filtered_stats_df)} / {len(stats_df)}")
    print(f"  Banks kept (inter/intra): {len(filtered_ii_df)} / {len(ii_df)}")
    return filtered_stats_df, filtered_ii_df


def print_edge_distribution(edge_counts):
    """Console-only diagnostic: percentile/threshold breakdown of edge counts
    per bank, useful when picking a MIN_EDGES cutoff for the filtered
    comparisons in the other subsections. Not written to any table."""
    sorted_counts = edge_counts.sort_values(ascending=False)
    print("=== Top/Bottom Banks by Edge Count ===")
    print(f"\nTop 10:\n{sorted_counts.head(10)}")
    print(f"\nBottom 10:\n{sorted_counts.tail(10)}")

    print("\n=== Percentile Thresholds ===")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(edge_counts, p)
        count_below = (edge_counts <= val).sum()
        print(f"  {p:2d}th percentile: {val:,.0f} edges ({count_below} banks below)")

    print("\n=== Banks by Edge Count Thresholds ===")
    for t in [100, 500, 1000, 5000, 10000, 50000, 100000]:
        below = (edge_counts < t).sum()
        above = (edge_counts >= t).sum()
        print(f"  <{t:,}: {below} banks | >={t:,}: {above} banks")


def build_quantity_skew_table(stats_df, out_name='quantity_skew_summary'):
    """Summary stats (N banks, min/median/mean/max/CV of n_edges and n_nodes) —
    the numbers the 'Quantity Skew' paragraph needs to cite in the main text."""
    rows = []
    for col in ['n_edges', 'n_nodes']:
        vals = stats_df[col]
        cv = vals.std() / vals.mean() if vals.mean() > 0 else float('nan')
        rows.append({
            'Metric': col,
            'N banks': len(vals),
            'Min': int(vals.min()),
            'Median': int(vals.median()),
            'Mean': round(vals.mean(), 1),
            'Max': int(vals.max()),
            'CV': round(cv, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / f'{out_name}.csv', index=False)
    df_to_latex_table(df, TABLES_DIR / f'{out_name}.tex')
    return df


def run(stats_df):
    print_edge_distribution(stats_df['n_edges'])
    table = build_quantity_skew_table(stats_df)
    print("\nQuantity skew summary:")
    print(table.to_string(index=False))

    # Node/edge count distribution
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
    savefig('node_edge_hist.pdf')

    # Sorted bars (descending by edge count)
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
    savefig('edge_node_sorted_bars.pdf')

    return table
