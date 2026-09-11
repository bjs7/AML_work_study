"""Shared plotting helpers and output-path constants for the heterogeneity
analysis package (quantity_skew, label_skew, pattern_covariate_shift,
inter_intra_bank). Figure filenames are unchanged from the old bank_analysis.py
script — only their directory moved, from analysis/figs/ (flat) to
analysis/figs/heterogeneity/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FONTSIZE = 11

FIGS_DIR = Path('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/figs/heterogeneity')
CSV_DIR = Path('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables/heterogeneity')
TABLES_DIR = Path('/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables/heterogeneity')

FIGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

thousands_fmt = FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}')


def savefig(name):
    """Save the current figure to FIGS_DIR / name, creating the dir if needed."""
    plt.savefig(FIGS_DIR / name, bbox_inches='tight')


def plot_proportion_heatmap(df, cols, xlabel, bank_id_col='bank_id', figsize=(7, 5), fontsize=FONTSIZE):
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


def plot_proportion_stacked_bar(df, cols, bank_id_col='bank_id', sort_by='n_edges', N=50,
                                 legend_title='Category', figsize=(7, 4), fontsize=FONTSIZE):
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


def plot_all_vs_filtered(plot_fn, df_all, df_filtered, threshold, **kwargs):
    """Run `plot_fn(df, **kwargs)` once for the unfiltered data and once for banks
    with >= threshold edges, side by side. `plot_fn` must accept a DataFrame as its
    first positional arg and return None (draws on the current/new axes itself) —
    used for the simple hist/scatter helpers in this package, not the two builders
    above (which have their own figure layout).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.sca(axes[0])
    plot_fn(df_all, **kwargs)
    axes[0].set_title('All banks')
    plt.sca(axes[1])
    plot_fn(df_filtered, **kwargs)
    axes[1].set_title(f'Banks with >= {threshold:,} edges')
    plt.tight_layout()
    return fig
