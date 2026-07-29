"""
HPC batch analysis script — derived from batch_utilization_analysis_nb.py.

Changes from the notebook version:
  - All batches processed (N_BATCHES = None)
  - All seed edges sampled for cone metrics (N_CONE_SAMPLE_MAX = None)
  - Full batch graph used for Cell-0 visualization (no node cap)
  - Non-interactive backend (Agg) — no plt.show()
  - All outputs written to a single HPC_OUTPUT_DIR for easy download
  - Timing printed every 50 batches
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    # scripts/hpc/analysis_hpc/ → scripts/hpc/ → scripts/ → repo root (3 parents)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import copy
import logging
import random
from collections import defaultdict
import numpy as np
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos
import models.gnn_models
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn


# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
HPC_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'hpc_output')
os.makedirs(HPC_OUTPUT_DIR, exist_ok=True)

# stats
N_BATCHES         = None   # None = all batches
N_CONE_SAMPLE_MAX = None   # None = all seed edges; set to int to cap

# viz
BATCH_INDICES           = [0, 1, 5]
SEED_MUST_BE_LAUNDERING = True
SEED_PICK               = 3
_K_HOPS                 = 2

# output paths — everything goes to HPC_OUTPUT_DIR for one-shot download
OUTPUT_BASE   = os.path.join(HPC_OUTPUT_DIR, 'batch_stats')
OUTPUT_SUFFIX = '_hpc'
OUTPUT_DIR    = HPC_OUTPUT_DIR
_TABLE_DIR    = HPC_OUTPUT_DIR

# node-fill colours
_C2_A     = '#e74c3c'
_C2_B     = '#3498db'
_C2_BOTH  = '#9b59b6'
_C2_NONE  = '#d5d8dc'
_C2_SEED  = '#e67e22'
_C1B_CONE = '#f39c12'
_C4_LAUND = '#c0392b'
_C4_LEGIT = '#aab7b8'

_ARGS = [
    '--fl_algo',       'FedGraphSimple',
    '--model',         'GINe',
    '--size',          'small',
    '--ir',            'HI',
    '--batching',
    '--batching_mode', 'lazy_link_neighbor',
    '--ibm_hp',
    '--emlps',
    '--eval_mode',     'comparable',
]

sys.argv = ['batch_analysis_hpc'] + _ARGS


# ==============================================================
# ====================== SETUP ================================
# ==============================================================

utils.logger_setup()
logging.getLogger().setLevel(logging.WARNING)

parsers, df, scaler_encoders = utils.setup_get_data()

laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(
    parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

manager = Manager.get_algo_class(parsers)
manager.verbose_setup = True
print("Setting up parties...")
manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)
print("Party setup complete.")

from federated_learning.gnn.vertical_simple import setup
setup.setup_vertical_simple(manager, batching=True, batching_mode='lazy_link_neighbor')
manager.setup_model(ibm_gnn, laundering_values_test)


# ==============================================================
# =================== BATCH SAMPLING ==========================
# ==============================================================

mode         = 'train'
mode_parties = manager.get_parties_for_mode(mode)

batch_records = []
party_records = []
viz_batches   = []

_t0 = time.time()
print(f"Sampling batches from '{mode}' loader (N_BATCHES={N_BATCHES}, "
      f"N_CONE_SAMPLE_MAX={N_CONE_SAMPLE_MAX})...")

for batch_idx, batch in enumerate(manager.loaders[mode]):
    if N_BATCHES is not None and batch_idx >= N_BATCHES:
        break

    n_batch_edges = batch.edge_attr.shape[0]
    n_batch_nodes = batch.x.shape[0]

    seed_global_ids = batch.edge_label.long().numpy()
    n_seed_edges    = len(seed_global_ids)

    bl             = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    bl_filtered    = bl[bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]
    n_seed_used          = len(bl_filtered)
    seed_laundering_rate = bl['Is Laundering'].mean()

    seed_banks        = set(bl_filtered['From Bank'].values) | set(bl_filtered['To Bank'].values)
    seed_mode_parties = {k: v for k, v in mode_parties.items() if k in seed_banks}

    seed_ids_tensor  = torch.tensor(seed_global_ids, dtype=torch.long)
    batch_global_ids = torch.cat([batch.edge_attr[:, 0].long(), seed_ids_tensor]).unique()

    viz_batches.append((batch, batch_global_ids.clone(), bl))

    n_active = 0
    for bank_id, party in seed_mode_parties.items():
        party_graph = party.procs_data[f'{mode}_data']['df']
        mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)
        if mask.sum() == 0:
            continue

        matched_edge_index = party_graph.edge_index[:, mask]
        n_party_edges      = int(mask.sum())
        n_party_nodes      = int(matched_edge_index.reshape(-1).unique().shape[0])

        matched_global_ids = party_graph.edge_attr[mask, 0].long().numpy()
        party_seed_ids     = set(matched_global_ids) & set(seed_global_ids)
        n_party_seed_edges = len(party_seed_ids)

        party_records.append({
            'batch_idx':           batch_idx,
            'bank_id':             bank_id,
            'n_party_edges':       n_party_edges,
            'n_party_nodes':       n_party_nodes,
            'n_batch_edges':       n_batch_edges,
            'n_batch_nodes':       n_batch_nodes,
            'party_edge_fraction': n_party_edges / n_batch_edges,
            'party_node_fraction': n_party_nodes / n_batch_nodes,
            'n_party_seed_edges':  n_party_seed_edges,
            'n_seed_edges':        n_seed_used,
            'party_seed_coverage': n_party_seed_edges / n_seed_used if n_seed_used > 0 else 0,
        })
        n_active += 1

    _N_CONE_SAMPLE = n_seed_edges if N_CONE_SAMPLE_MAX is None else min(N_CONE_SAMPLE_MAX, n_seed_edges)
    _K_CONE        = 2

    _radj_s = defaultdict(set)
    _esrc_s = batch.edge_index[0].tolist()
    _edst_s = batch.edge_index[1].tolist()
    _egid_s = batch.edge_attr[:, 0].long().numpy().tolist()
    for _s, _d in zip(_esrc_s, _edst_s):
        _radj_s[_d].add(_s)
    _e2g_s = {(_s, _d): _g for _s, _d, _g in zip(_esrc_s, _edst_s, _egid_s)}

    _adj_s_u = defaultdict(set)
    for _s, _d in zip(_esrc_s, _edst_s):
        _adj_s_u[_s].add(_d)
        _adj_s_u[_d].add(_s)

    _pgids_s = {}
    for _bk, _pty in seed_mode_parties.items():
        _pg  = _pty.procs_data[f'{mode}_data']['df']
        _mk  = torch.isin(_pg.edge_attr[:, 0].long(), batch_global_ids)
        _pgids_s[_bk] = set(_pg.edge_attr[_mk, 0].long().numpy().tolist()) if _mk.sum() > 0 else set()

    _rng_cone    = np.random.default_rng(seed=batch_idx)
    _cone_idx    = _rng_cone.choice(n_seed_edges, _N_CONE_SAMPLE, replace=False)
    _all_lbl_idx = set(manager.ctx[mode]['df_labels'].index)
    _df_lbls     = manager.ctx[mode]['df_labels']

    _batch_egids_np        = batch.edge_attr[:, 0].long().numpy()
    _known_b               = [g for g in _batch_egids_np if g in _all_lbl_idx]
    _batch_edge_laund_rate = float(_df_lbls.loc[_known_b, 'Is Laundering'].mean()) if _known_b else float('nan')

    _cone_from_cov, _cone_to_cov, _cone_union_cov, _cone_neither = [], [], [], []
    _cone_unique_from, _cone_unique_to, _cone_overlap_cov, _cone_nesting_idx = [], [], [], []
    _cone_laund_frac_all, _cone_laund_frac_laund = [], []
    _cone_n_nodes_l, _cone_n_edges_l = [], []
    _neigh_n_nodes_l, _neigh_n_edges_l = [], []
    _cone_asymmetry_l = []
    _cone_from_cov_il, _cone_to_cov_il, _cone_union_cov_il, _cone_neither_il = [], [], [], []
    _cone_unique_from_il, _cone_unique_to_il, _cone_overlap_cov_il, _cone_nesting_idx_il = [], [], [], []
    _cone_n_nodes_il, _cone_n_edges_il = [], []
    _neigh_n_nodes_il, _neigh_n_edges_il = [], []
    _cone_asymmetry_il = []

    for _si in _cone_idx:
        _csrc = int(batch.edge_label_index[0, _si])
        _cdst = int(batch.edge_label_index[1, _si])
        _cgid = int(batch.edge_label[_si])
        if _cgid not in _all_lbl_idx:
            continue
        _cr = _df_lbls.loc[_cgid]
        _ca, _cb = _cr['From Bank'], _cr['To Bank']

        _fr, _vi = {_csrc, _cdst}, {_csrc, _cdst}
        for _ in range(_K_CONE):
            _nxt = {_nb for _n in _fr for _nb in _radj_s[_n] if _nb not in _vi}
            _vi |= _nxt; _fr = _nxt

        _fr_u, _vi_u = {_csrc, _cdst}, {_csrc, _cdst}
        for _ in range(_K_CONE):
            _nxt_u = {_nb for _n in _fr_u for _nb in _adj_s_u[_n] if _nb not in _vi_u}
            _vi_u |= _nxt_u; _fr_u = _nxt_u
        _neigh_n_edges_i = sum(1 for _s, _d in zip(_esrc_s, _edst_s) if _s in _vi_u and _d in _vi_u)

        _cgids = {_e2g_s[(_s, _d)] for _s, _d in zip(_esrc_s, _edst_s)
                  if _s in _vi and _d in _vi and (_s, _d) in _e2g_s}
        if not _cgids:
            continue

        _nc        = len(_cgids)
        _ga        = _pgids_s.get(_ca, set())
        _gb        = _pgids_s.get(_cb, set())
        _from_i    = len(_cgids & _ga) / _nc
        _to_i      = len(_cgids & _gb) / _nc
        _union_i   = len(_cgids & (_ga | _gb)) / _nc
        _neither_i = len(_cgids - _ga - _gb) / _nc
        _overlap_i = _from_i + _to_i - _union_i

        _cone_from_cov.append(_from_i)
        _cone_to_cov.append(_to_i)
        _cone_union_cov.append(_union_i)
        _cone_neither.append(_neither_i)
        _cone_unique_from.append(_union_i - _to_i)
        _cone_unique_to.append(_union_i - _from_i)
        _cone_overlap_cov.append(_overlap_i)
        _min_cov = min(_from_i, _to_i)
        _cone_nesting_idx.append(_overlap_i / _min_cov if _min_cov > 0 else float('nan'))
        _cone_n_nodes_l.append(len(_vi))
        _cone_n_edges_l.append(_nc)
        _neigh_n_nodes_l.append(len(_vi_u))
        _neigh_n_edges_l.append(_neigh_n_edges_i)
        _cone_asymmetry_l.append(abs(_from_i - _to_i))

        _cgids_known  = [g for g in _cgids if g in _all_lbl_idx]
        _laund_frac_i = (sum(1 for g in _cgids_known if bool(_df_lbls.loc[g, 'Is Laundering'])) / len(_cgids_known)
                         if _cgids_known else float('nan'))
        _cone_laund_frac_all.append(_laund_frac_i)
        if bool(_cr['Is Laundering']):
            _cone_laund_frac_laund.append(_laund_frac_i)
            _cone_from_cov_il.append(_from_i)
            _cone_to_cov_il.append(_to_i)
            _cone_union_cov_il.append(_union_i)
            _cone_neither_il.append(_neither_i)
            _cone_overlap_cov_il.append(_overlap_i)
            _cone_unique_from_il.append(_union_i - _to_i)
            _cone_unique_to_il.append(_union_i - _from_i)
            _min_cov_il = min(_from_i, _to_i)
            _cone_nesting_idx_il.append(_overlap_i / _min_cov_il if _min_cov_il > 0 else float('nan'))
            _cone_n_nodes_il.append(len(_vi))
            _cone_n_edges_il.append(_nc)
            _neigh_n_nodes_il.append(len(_vi_u))
            _neigh_n_edges_il.append(_neigh_n_edges_i)
            _cone_asymmetry_il.append(abs(_from_i - _to_i))

    _mean_nesting = float(np.nanmean(_cone_nesting_idx))      if _cone_nesting_idx      else float('nan')
    _mean_l_laund = float(np.nanmean(_cone_laund_frac_laund)) if _cone_laund_frac_laund else float('nan')
    batch_records.append({
        'batch_idx':             batch_idx,
        'n_batch_edges':         n_batch_edges,
        'n_batch_nodes':         n_batch_nodes,
        'n_seed_edges':          n_seed_edges,
        'n_seed_used':           n_seed_used,
        'seed_laundering_rate':  seed_laundering_rate,
        'n_active_parties':      n_active,
        'batch_edge_laund_rate': _batch_edge_laund_rate,
        'cone_from_cov':         float(np.mean(_cone_from_cov))    if _cone_from_cov    else float('nan'),
        'cone_to_cov':           float(np.mean(_cone_to_cov))      if _cone_to_cov      else float('nan'),
        'cone_union_cov':        float(np.mean(_cone_union_cov))   if _cone_union_cov   else float('nan'),
        'cone_neither_frac':     float(np.mean(_cone_neither))     if _cone_neither     else float('nan'),
        'cone_unique_from':      float(np.mean(_cone_unique_from)) if _cone_unique_from else float('nan'),
        'cone_unique_to':        float(np.mean(_cone_unique_to))   if _cone_unique_to   else float('nan'),
        'cone_overlap_cov':      float(np.mean(_cone_overlap_cov)) if _cone_overlap_cov else float('nan'),
        'cone_nesting_idx':      _mean_nesting,
        'cone_laund_frac':       float(np.nanmean(_cone_laund_frac_all)) if _cone_laund_frac_all else float('nan'),
        'cone_laund_frac_laund': _mean_l_laund,
        'cone_laund_enrichment': (_mean_l_laund / _batch_edge_laund_rate
                                  if not (np.isnan(_mean_l_laund) or np.isnan(_batch_edge_laund_rate) or _batch_edge_laund_rate == 0)
                                  else float('nan')),
        'cone_n_nodes':          float(np.mean(_cone_n_nodes_l))    if _cone_n_nodes_l    else float('nan'),
        'cone_n_edges':          float(np.mean(_cone_n_edges_l))    if _cone_n_edges_l    else float('nan'),
        'neigh_n_nodes':         float(np.mean(_neigh_n_nodes_l))   if _neigh_n_nodes_l   else float('nan'),
        'neigh_n_edges':         float(np.mean(_neigh_n_edges_l))   if _neigh_n_edges_l   else float('nan'),
        'cone_asymmetry':        float(np.mean(_cone_asymmetry_l))  if _cone_asymmetry_l  else float('nan'),
        'cone_from_cov_std':     float(np.std(_cone_from_cov))      if len(_cone_from_cov)  > 1 else float('nan'),
        'cone_to_cov_std':       float(np.std(_cone_to_cov))        if len(_cone_to_cov)    > 1 else float('nan'),
        'cone_union_cov_std':    float(np.std(_cone_union_cov))     if len(_cone_union_cov) > 1 else float('nan'),
        'cone_from_cov_il':      float(np.mean(_cone_from_cov_il))       if _cone_from_cov_il      else float('nan'),
        'cone_to_cov_il':        float(np.mean(_cone_to_cov_il))         if _cone_to_cov_il        else float('nan'),
        'cone_union_cov_il':     float(np.mean(_cone_union_cov_il))      if _cone_union_cov_il     else float('nan'),
        'cone_neither_frac_il':  float(np.mean(_cone_neither_il))        if _cone_neither_il       else float('nan'),
        'cone_overlap_cov_il':   float(np.mean(_cone_overlap_cov_il))    if _cone_overlap_cov_il   else float('nan'),
        'cone_unique_from_il':   float(np.mean(_cone_unique_from_il))    if _cone_unique_from_il   else float('nan'),
        'cone_unique_to_il':     float(np.mean(_cone_unique_to_il))      if _cone_unique_to_il     else float('nan'),
        'cone_nesting_idx_il':   float(np.nanmean(_cone_nesting_idx_il)) if _cone_nesting_idx_il   else float('nan'),
        'cone_n_nodes_il':       float(np.mean(_cone_n_nodes_il))        if _cone_n_nodes_il       else float('nan'),
        'cone_n_edges_il':       float(np.mean(_cone_n_edges_il))        if _cone_n_edges_il       else float('nan'),
        'neigh_n_nodes_il':      float(np.mean(_neigh_n_nodes_il))       if _neigh_n_nodes_il      else float('nan'),
        'neigh_n_edges_il':      float(np.mean(_neigh_n_edges_il))       if _neigh_n_edges_il      else float('nan'),
        'cone_asymmetry_il':     float(np.mean(_cone_asymmetry_il))      if _cone_asymmetry_il     else float('nan'),
    })

    if (batch_idx + 1) % 50 == 0:
        _elapsed = time.time() - _t0
        print(f"  {batch_idx + 1} batches in {_elapsed:.0f}s  ({_elapsed / (batch_idx + 1):.1f}s/batch)")

batch_df = pd.DataFrame(batch_records)
party_df = pd.DataFrame(party_records)
_elapsed_total = time.time() - _t0
print(f"Sampling done: {len(batch_df)} batches in {_elapsed_total:.0f}s")

batch_df.to_csv(f'{OUTPUT_BASE}_batch.csv', index=False)
party_df.to_csv(f'{OUTPUT_BASE}_party.csv', index=False)
print(f"Saved: {OUTPUT_BASE}_batch.csv")
print(f"Saved: {OUTPUT_BASE}_party.csv")


# ==============================================================
# ===== TABLES 1a / 1b: MP CONE COVERAGE — COMBINED ===========
# ==============================================================

_cone_cols_a = [
    'cone_from_cov', 'cone_to_cov', 'cone_union_cov',
    'cone_neither_frac', 'cone_asymmetry',
    'cone_n_nodes', 'cone_n_edges',
    'neigh_n_nodes', 'neigh_n_edges',
]
_cone_cols_b = [
    'cone_overlap_cov', 'cone_unique_from', 'cone_unique_to', 'cone_nesting_idx',
    'cone_laund_frac', 'cone_laund_frac_laund', 'cone_laund_enrichment',
]
_cone_cols_a_il = [
    'cone_from_cov_il', 'cone_to_cov_il', 'cone_union_cov_il',
    'cone_neither_frac_il', 'cone_asymmetry_il',
    'cone_n_nodes_il', 'cone_n_edges_il',
    'neigh_n_nodes_il', 'neigh_n_edges_il',
]
_cone_cols_b_il = [
    'cone_overlap_cov_il', 'cone_unique_from_il', 'cone_unique_to_il', 'cone_nesting_idx_il',
    'cone_laund_frac_laund',
]

_cone_summary_a    = batch_df[_cone_cols_a].agg(['mean', 'std']).round(4)
_cone_summary_b    = batch_df[_cone_cols_b].agg(['mean', 'std']).round(4)
_cone_summary_a_il = batch_df[_cone_cols_a_il].agg(['mean', 'std']).round(4)
_cone_summary_b_il = batch_df[_cone_cols_b_il].agg(['mean', 'std']).round(4)

_il_a = _cone_summary_a_il.rename(columns=lambda c: c.replace('_il', ''))
_il_b = _cone_summary_b_il.rename(
    columns=lambda c: c.replace('_il', '') if c.endswith('_il') else c)

_cone_summary_a.to_csv(   os.path.join(_TABLE_DIR, 'cone_coverage_summary_a.csv'))
_cone_summary_b.to_csv(   os.path.join(_TABLE_DIR, 'cone_coverage_summary_b.csv'))
_cone_summary_a_il.to_csv(os.path.join(_TABLE_DIR, 'cone_coverage_illicit_a.csv'))
_cone_summary_b_il.to_csv(os.path.join(_TABLE_DIR, 'cone_coverage_illicit_b.csv'))

pd.concat({'All seeds': _cone_summary_a,    'Illicit seeds': _il_a}).to_csv(
    os.path.join(_TABLE_DIR, 'cone_coverage_combined_a.csv'))
pd.concat({'All seeds': _cone_summary_b,    'Illicit seeds': _il_b}).to_csv(
    os.path.join(_TABLE_DIR, 'cone_coverage_combined_b.csv'))


def _grouped_latex(groups, cols):
    labels = [c.replace('_', r'\_') for c in cols]
    n = len(cols)
    out = [r'\begin{tabular}{l' + 'r' * n + '}', r'\hline',
           '  & ' + ' & '.join(labels) + r' \\', r'\hline']
    for grp, df_g in groups.items():
        out.append(r'  \textit{' + grp + '} & ' + ' & '.join([''] * n) + r' \\')
        for stat in df_g.index:
            vals = []
            for c in cols:
                if c in df_g.columns:
                    v = df_g.loc[stat, c]
                    vals.append(f'{v:.4f}' if not pd.isna(v) else '---')
                else:
                    vals.append('---')
            out.append(r'  \quad ' + stat + ' & ' + ' & '.join(vals) + r' \\')
        out.append(r'  \hline')
    out.append(r'\end{tabular}')
    return '\n'.join(out)

with open(os.path.join(_TABLE_DIR, 'cone_coverage_combined_a.tex'), 'w') as _f:
    _f.write(_grouped_latex({'All seeds': _cone_summary_a, 'Illicit seeds': _il_a}, _cone_cols_a))
with open(os.path.join(_TABLE_DIR, 'cone_coverage_combined_b.tex'), 'w') as _f:
    _f.write(_grouped_latex({'All seeds': _cone_summary_b, 'Illicit seeds': _il_b}, _cone_cols_b))

print("Saved cone coverage tables.")


# ==============================================================
# ===== TABLE 2: PARTY BATCH COVERAGE SUMMARY =================
# ==============================================================

_party_cols = [
    'party_edge_fraction', 'party_node_fraction', 'party_seed_coverage',
    'n_party_edges', 'n_party_nodes', 'n_party_seed_edges',
]
_party_summary = party_df[_party_cols].agg(['mean', 'std']).round(4)
_party_summary.to_csv(os.path.join(_TABLE_DIR, 'party_batch_coverage_summary.csv'))
(_party_summary
 .rename(columns=lambda c: c.replace('_', r'\_'))
 .to_latex(os.path.join(_TABLE_DIR, 'party_batch_coverage_summary.tex'), escape=False))
print("Saved party coverage table.")


# ==============================================================
# ===== LOAD BATCHES FOR VISUALIZATION ========================
# ==============================================================
# Loads BATCH_INDICES batches. Cell-0 BFS uses all seed nodes as
# starting points and no node cap — produces a full-batch graph.

_all_labels = manager.ctx[mode]['df_labels']

print(f"\nLoading visualization batches {BATCH_INDICES}...")
torch.manual_seed(1)
_loaded: dict = {}
for _bi_load, _batch_obj in enumerate(manager.loaders[mode]):
    if _bi_load in BATCH_INDICES:
        _loaded[_bi_load] = _batch_obj
    if len(_loaded) == len(BATCH_INDICES):
        break

_batches = []

for _bi_enum, _batch_idx in enumerate(BATCH_INDICES):
    assert _batch_idx in _loaded, \
        f"Batch {_batch_idx} not found — loader has {max(_loaded)+1} batches"
    _vb = _loaded[_batch_idx]
    print(f"\n--- Batch {_batch_idx} ---")
    print(f"  {_vb.x.shape[0]} nodes · {_vb.edge_attr.shape[0]} edges · "
          f"{_vb.edge_label.shape[0]} seed transactions")

    _node_bank: dict = {}
    def _reg_own(edge_index, global_ids):
        gids = global_ids.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            gid = int(gids[i])
            if gid not in _all_labels.index:
                continue
            row = _all_labels.loc[gid]
            if src not in _node_bank: _node_bank[src] = row['From Bank']
            if dst not in _node_bank: _node_bank[dst] = row['To Bank']
    _reg_own(_vb.edge_index, _vb.edge_attr[:, 0].long())
    _reg_own(_vb.edge_label_index, _vb.edge_label)

    _radj: dict = defaultdict(set)
    for _s, _d in zip(_vb.edge_index[0].tolist(), _vb.edge_index[1].tolist()):
        _radj[_d].add(_s)

    _laund_pairs: set = set()
    for _s, _d, _g in zip(
        _vb.edge_index[0].tolist(),
        _vb.edge_index[1].tolist(),
        _vb.edge_attr[:, 0].long().numpy().tolist(),
    ):
        if int(_g) in _all_labels.index and bool(_all_labels.loc[int(_g), 'Is Laundering']):
            _laund_pairs.add((int(_s), int(_d)))

    _src_ov = _vb.edge_index[0].numpy().tolist()
    _dst_ov = _vb.edge_index[1].numpy().tolist()
    _adj_ov: dict = defaultdict(set)
    for _s, _d in zip(_src_ov, _dst_ov):
        _adj_ov[_s].add(_d)
        _adj_ov[_d].add(_s)

    torch.manual_seed(1)
    _n_seeds_total = _vb.edge_label_index.shape[1]
    _seed_gids_all = _vb.edge_label.long().numpy()
    _in_lbl        = pd.Series(_seed_gids_all).isin(_all_labels.index)
    _known_pos     = _in_lbl[_in_lbl].index.to_numpy()

    if SEED_MUST_BE_LAUNDERING and len(_known_pos) > 0:
        _laund_flags = _all_labels.loc[_seed_gids_all[_known_pos], 'Is Laundering'].values.astype(bool)
        _laund_pos   = _known_pos[_laund_flags]
        if len(_laund_pos) > 0:
            _pick        = (SEED_PICK + _bi_enum) % len(_laund_pos)
            _seed_idx_ov = torch.tensor([int(_laund_pos[_pick])], dtype=torch.long)
        else:
            print("  Warning: no laundering seeds found — picking randomly.")
            _seed_idx_ov = torch.randperm(_n_seeds_total)[:1]
    else:
        _seed_idx_ov = torch.randperm(_n_seeds_total)[:1]

    _seed_pos      = int(_seed_idx_ov[0])
    _seed_gid      = int(_vb.edge_label[_seed_pos])
    _seed_src      = int(_vb.edge_label_index[0, _seed_pos])
    _seed_dst      = int(_vb.edge_label_index[1, _seed_pos])
    _seed_row      = _all_labels.loc[_seed_gid]
    _pa            = _seed_row['From Bank']
    _pb            = _seed_row['To Bank']
    _is_laundering = bool(_seed_row['Is Laundering'])
    print(f"  Seed {_seed_gid}: {_pa} → {_pb}  (laundering={_is_laundering})")

    _pa_nodes: set = set()
    _pb_nodes: set = set()
    for _ei in range(_vb.edge_index.shape[1]):
        _s2, _d2 = int(_vb.edge_index[0, _ei]), int(_vb.edge_index[1, _ei])
        _bs = _node_bank.get(_s2)
        _bd = _node_bank.get(_d2)
        if _bs == _pa or _bd == _pa:
            _pa_nodes.add(_s2); _pa_nodes.add(_d2)
        if _bs == _pb or _bd == _pb:
            _pb_nodes.add(_s2); _pb_nodes.add(_d2)

    def _fill(n, _a=_pa_nodes, _b=_pb_nodes):
        in_a, in_b = n in _a, n in _b
        if in_a and in_b: return _C2_BOTH
        elif in_a:        return _C2_A
        elif in_b:        return _C2_B
        else:             return _C2_NONE

    _frontier_cone = {_seed_src, _seed_dst}
    _cone: set     = set(_frontier_cone)
    for _ in range(_K_HOPS):
        _next_cone = set()
        for _n in _frontier_cone:
            for _nb in _radj[_n]:
                if _nb not in _cone:
                    _next_cone.add(_nb)
        _cone         |= _next_cone
        _frontier_cone = _next_cone

    _frontier_ov = {_seed_src, _seed_dst}
    _visited_ov: set = set(_frontier_ov)
    for _ in range(_K_HOPS):
        _next_ov = set()
        for _n in _frontier_ov:
            for _nb in _adj_ov[_n]:
                if _nb not in _visited_ov:
                    _next_ov.add(_nb)
        _visited_ov  |= _next_ov
        _frontier_ov  = _next_ov

    # Cell 0: full-batch BFS from all seed nodes (no node cap)
    _c0_init = set()
    for _i in range(_n_seeds_total):
        _c0_init.add(int(_vb.edge_label_index[0, _i]))
        _c0_init.add(int(_vb.edge_label_index[1, _i]))
    _frontier_c0 = set(_c0_init)
    _visited_c0: set = set(_frontier_c0)
    while _frontier_c0:
        _next_c0 = set()
        for _n in _frontier_c0:
            for _nb in _adj_ov[_n]:
                if _nb not in _visited_c0:
                    _next_c0.add(_nb)
        _visited_c0 |= _next_c0
        _frontier_c0  = _next_c0
    _c0_edges = [
        (int(_s), int(_d)) for _s, _d in zip(_src_ov, _dst_ov)
        if int(_s) in _visited_c0 and int(_d) in _visited_c0 and int(_s) != int(_d)
    ]
    _is_full_c0 = len(_visited_c0) >= _vb.x.shape[0]
    _c0_label   = 'full batch' if _is_full_c0 else f'{len(_visited_c0)} nodes'
    _G_c0 = nx.DiGraph()
    _G_c0.add_nodes_from(_visited_c0)
    _G_c0.add_edges_from(_c0_edges)
    print(f"  Cell 0: {len(_visited_c0)}/{_vb.x.shape[0]} nodes  ({_c0_label})")
    print("  Computing Cell 0 layout...")
    _pos_c0 = nx.spring_layout(_G_c0, seed=42, k=2, iterations=80)

    _ov_edges = [
        (int(_s), int(_d)) for _s, _d in zip(_src_ov, _dst_ov)
        if int(_s) in _visited_ov and int(_d) in _visited_ov and int(_s) != int(_d)
    ]
    _G_ov = nx.DiGraph()
    _G_ov.add_nodes_from(_visited_ov)
    _G_ov.add_edges_from(_ov_edges)
    print(f"  Cell 1: {len(_visited_ov)} nodes · {len(_ov_edges)} edges")
    print("  Computing Cell 1 layout...")
    _pos_ov = nx.spring_layout(_G_ov, seed=42, k=4, iterations=150)

    _cone_edges = [
        (int(_s), int(_d))
        for _s, _d in zip(_vb.edge_index[0].tolist(), _vb.edge_index[1].tolist())
        if int(_s) in _cone and int(_d) in _cone and int(_s) != int(_d)
    ]
    _G_cone = nx.DiGraph()
    _G_cone.add_nodes_from(_cone)
    _G_cone.add_edges_from(_cone_edges)
    _pos_cone_known = {n: _pos_ov[n] for n in _G_cone.nodes() if n in _pos_ov}
    _pos_cone_miss  = {n for n in _G_cone.nodes() if n not in _pos_ov}
    if _pos_cone_miss:
        _pos_cone = nx.spring_layout(_G_cone, pos=_pos_cone_known,
                                     fixed=list(_pos_cone_known.keys()),
                                     seed=42, k=1.0, iterations=50)
    else:
        _pos_cone = _pos_cone_known
    print(f"  Cell 3: {_G_cone.number_of_nodes()} nodes · {_G_cone.number_of_edges()} edges")

    _batches.append({
        'batch_idx':     _batch_idx,
        'n_nodes_batch': _vb.x.shape[0],
        'n_edges_batch': _vb.edge_attr.shape[0],
        'n_seeds_total': _n_seeds_total,
        'seed_gid':      _seed_gid,
        'seed_src':      _seed_src,
        'seed_dst':      _seed_dst,
        'pa':            _pa,
        'pb':            _pb,
        'is_laundering': _is_laundering,
        'node_bank':     _node_bank,
        'pa_nodes':      _pa_nodes,
        'pb_nodes':      _pb_nodes,
        'laund_pairs':   _laund_pairs,
        'cone':          _cone,
        'visited_ov':    _visited_ov,
        'fill':          _fill,
        'visited_c0':    _visited_c0,
        'G_c0':          _G_c0,
        'pos_c0':        _pos_c0,
        'c0_edges':      _c0_edges,
        'is_full_c0':    _is_full_c0,
        'c0_label':      _c0_label,
        'neigh_in_c0':   _visited_ov & _visited_c0,
        'neigh_out_c0':  _visited_ov - _visited_c0,
        'G_ov':          _G_ov,
        'pos_ov':        _pos_ov,
        'ov_edges':      _ov_edges,
        'G_cone':        _G_cone,
        'pos_cone':      _pos_cone,
        'cone_edges':    _cone_edges,
    })

_N = len(_batches)
print(f"\nAll {_N} batches loaded and pre-computed.")


# ==============================================================
# ===== CELL 0a: BATCH OVERVIEW ===============================
# ==============================================================

_VIZ_CELL0A = os.path.join(OUTPUT_DIR, f'batch_viz_cell0a{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    nx.draw_networkx_nodes(bd['G_c0'], bd['pos_c0'], ax=ax,
                           node_color='#3498db', node_size=50,
                           edgecolors='none', linewidths=0, alpha=0.85)
    nx.draw_networkx_edges(bd['G_c0'], bd['pos_c0'], ax=ax,
                           edge_color='#aaaaaa', width=0.7, arrows=True,
                           arrowsize=10, arrowstyle='->', alpha=0.35,
                           connectionstyle='arc3,rad=0.05')
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {len(bd["visited_c0"])}/{bd["n_nodes_batch"]} nodes  '
        f'({bd["c0_label"]})\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle('Cell 0a: Full batch overview', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL0A, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL0A}")


# ==============================================================
# ===== CELL 0b: BATCH WITH K-HOP NEIGHBOURHOOD HIGHLIGHTED ===
# ==============================================================

_VIZ_CELL0B = os.path.join(OUTPUT_DIR, f'batch_viz_cell0b{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    G_c0     = bd['G_c0']
    pos_c0   = bd['pos_c0']
    vis_ov   = bd['visited_ov']
    seed_src = bd['seed_src']
    seed_dst = bd['seed_dst']
    _bg_nds, _fg_nds = [], []
    _fg_nc, _fg_ns, _fg_nec, _fg_nlw = [], [], [], []
    for _n in G_c0.nodes():
        if _n not in vis_ov:
            _bg_nds.append(_n)
        else:
            _fg_nds.append(_n)
            if _n == seed_src:
                _fg_nc.append(_C2_SEED); _fg_ns.append(280)
                _fg_nec.append(_C2_A);   _fg_nlw.append(2.5)
            elif _n == seed_dst:
                _fg_nc.append(_C2_SEED); _fg_ns.append(280)
                _fg_nec.append(_C2_B);   _fg_nlw.append(2.5)
            else:
                _fg_nc.append('#3498db'); _fg_ns.append(50)
                _fg_nec.append('none');   _fg_nlw.append(0.0)
    _fg_edges = [(s, d) for s, d in G_c0.edges() if s in vis_ov and d in vis_ov]
    _bg_edges = [(s, d) for s, d in G_c0.edges() if s not in vis_ov or d not in vis_ov]
    if _bg_nds:
        nx.draw_networkx_nodes(G_c0, pos_c0, ax=ax, nodelist=_bg_nds,
                               node_color='#d5d8dc', node_size=50,
                               edgecolors='none', linewidths=0, alpha=0.75)
    nx.draw_networkx_nodes(G_c0, pos_c0, ax=ax, nodelist=_fg_nds,
                           node_color=_fg_nc, node_size=_fg_ns,
                           edgecolors=_fg_nec, linewidths=_fg_nlw, alpha=0.95)
    if _bg_edges:
        nx.draw_networkx_edges(G_c0, pos_c0, ax=ax, edgelist=_bg_edges,
                               edge_color='#aaaaaa', width=0.7, arrows=True,
                               arrowsize=10, arrowstyle='->', alpha=0.35,
                               connectionstyle='arc3,rad=0.05')
    if _fg_edges:
        nx.draw_networkx_edges(G_c0, pos_c0, ax=ax, edgelist=_fg_edges,
                               edge_color='#888888', width=0.9, arrows=True,
                               arrowsize=12, arrowstyle='->', alpha=0.5,
                               connectionstyle='arc3,rad=0.05')
    _nin  = len(bd['neigh_in_c0'])
    _nout = len(bd['neigh_out_c0'])
    _miss = f'  ({_nout} neigh. nodes outside BFS)' if _nout else ''
    ax.legend(handles=[
        mpatches.Patch(color='#3498db',  label=f'{_K_HOPS}-hop neighbourhood ({_nin} nodes)'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src (party {bd["pa"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst (party {bd["pb"]})'),
        mpatches.Patch(color='#d5d8dc', alpha=0.75, label='Rest of batch (greyed)'),
    ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {_K_HOPS}-hop neighbourhood highlighted{_miss}\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(f'Cell 0b: Batch with {_K_HOPS}-hop neighbourhood highlighted', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL0B, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL0B}")


# ==============================================================
# ===== CELL 1: K-HOP NEIGHBOURHOOD OF SEED ==================
# ==============================================================

_VIZ_CELL1 = os.path.join(OUTPUT_DIR, f'batch_viz_cell1_neighborhood{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    nx.draw_networkx_nodes(bd['G_ov'], bd['pos_ov'], ax=ax,
                           node_color='#3498db', node_size=100,
                           edgecolors='none', linewidths=0, alpha=0.85)
    nx.draw_networkx_edges(bd['G_ov'], bd['pos_ov'], ax=ax,
                           edge_color='#888888', width=1.2, arrows=True,
                           arrowsize=14, arrowstyle='->', alpha=0.5,
                           connectionstyle='arc3,rad=0.05')
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {len(bd["visited_ov"])} nodes · {len(bd["ov_edges"])} edges\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(f'Cell 1: {_K_HOPS}-hop undirected neighbourhood of seed', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL1, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL1}")


# ==============================================================
# ===== CELL 1b: CONE HIGHLIGHTED IN NEIGHBOURHOOD ============
# ==============================================================

_VIZ_CELL1B = os.path.join(OUTPUT_DIR, f'batch_viz_cell1b_cone_in_neighborhood{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    G_ov     = bd['G_ov']
    pos_ov   = bd['pos_ov']
    cone     = bd['cone']
    seed_src = bd['seed_src']
    seed_dst = bd['seed_dst']
    _nc, _ns, _nec, _nlw = [], [], [], []
    for _n in G_ov.nodes():
        if _n == seed_src:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_A);   _nlw.append(2.5)
        elif _n == seed_dst:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_B);   _nlw.append(2.5)
        elif _n in cone:
            _nc.append(_C1B_CONE); _ns.append(130)
            _nec.append('none');   _nlw.append(0.0)
        else:
            _nc.append('#3498db'); _ns.append(80)
            _nec.append('none');   _nlw.append(0.0)
    _n_in_cone = sum(1 for n in G_ov.nodes() if n in cone and n not in {seed_src, seed_dst})
    nx.draw_networkx_nodes(G_ov, pos_ov, ax=ax,
                           node_color=_nc, node_size=_ns,
                           edgecolors=_nec, linewidths=_nlw, alpha=0.85)
    nx.draw_networkx_edges(G_ov, pos_ov, ax=ax,
                           edge_color='#888888', width=1.2, arrows=True,
                           arrowsize=14, arrowstyle='->', alpha=0.5,
                           connectionstyle='arc3,rad=0.05')
    ax.legend(handles=[
        mpatches.Patch(color=_C1B_CONE, label=f'In {_K_HOPS}-hop MP cone ({_n_in_cone})'),
        mpatches.Patch(color='#3498db', label='Neighbourhood but outside cone'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb"]})'),
    ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {len(bd["visited_ov"])} nodes  ·  cone: {_n_in_cone}\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(f'Cell 1b: {_K_HOPS}-hop neighbourhood — gold = feeds prediction via MP', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL1B, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL1B}")


# ==============================================================
# ===== CELL 2: PARTY VISIBILITY ON K-HOP NEIGHBOURHOOD =======
# ==============================================================

_VIZ_CELL2 = os.path.join(OUTPUT_DIR, f'batch_viz_cell2_visibility{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    G_ov     = bd['G_ov']
    pos_ov   = bd['pos_ov']
    cone     = bd['cone']
    fill     = bd['fill']
    seed_src = bd['seed_src']
    seed_dst = bd['seed_dst']
    _grey_dim = [n for n in G_ov.nodes()
                 if fill(n) == _C2_NONE and n not in cone and n not in {seed_src, seed_dst}]
    _grey_set = set(_grey_dim)
    _p2_nds, _p2_nc, _p2_ns, _p2_nec, _p2_nlw = [], [], [], [], []
    for _n in G_ov.nodes():
        if _n in _grey_set: continue
        _p2_nds.append(_n)
        if _n == seed_src:
            _p2_nc.append(_C2_SEED); _p2_ns.append(280)
            _p2_nec.append(_C2_A);   _p2_nlw.append(2.5)
        elif _n == seed_dst:
            _p2_nc.append(_C2_SEED); _p2_ns.append(280)
            _p2_nec.append(_C2_B);   _p2_nlw.append(2.5)
        elif _n in cone:
            _p2_nc.append(fill(_n)); _p2_ns.append(80)
            _p2_nec.append('#f39c12'); _p2_nlw.append(1.5)
        else:
            _p2_nc.append(fill(_n)); _p2_ns.append(70)
            _p2_nec.append('none');   _p2_nlw.append(0.0)
    if _grey_dim:
        nx.draw_networkx_nodes(G_ov, pos_ov, ax=ax, nodelist=_grey_dim,
                               node_color=_C2_NONE, node_size=50,
                               edgecolors='none', linewidths=0, alpha=0.75)
    nx.draw_networkx_nodes(G_ov, pos_ov, ax=ax, nodelist=_p2_nds,
                           node_color=_p2_nc, node_size=_p2_ns,
                           edgecolors=_p2_nec, linewidths=_p2_nlw, alpha=0.85)
    nx.draw_networkx_edges(G_ov, pos_ov, ax=ax,
                           edge_color='#888888', width=1.2, arrows=True,
                           arrowsize=14, arrowstyle='->', alpha=0.5,
                           connectionstyle='arc3,rad=0.05')
    ax.legend(handles=[
        mpatches.Patch(color=_C2_A,    label=f'Party {bd["pa"]}'),
        mpatches.Patch(color=_C2_B,    label=f'Party {bd["pb"]}'),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor='white', edgecolor='#f39c12', linewidth=1.5,
                       label=f'Gold border = in {_K_HOPS}-hop MP cone'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb"]})'),
    ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'Batch {bd["batch_idx"]} — party visibility\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(f'Cell 2: Party visibility · {_K_HOPS}-hop neighbourhood  '
             f'(gold border = in MP cone · grey dim = neither party)', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL2, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL2}")


# ==============================================================
# ===== CELL 3: MP CONE — PARTY COLOURED =====================
# ==============================================================

_VIZ_CELL3 = os.path.join(OUTPUT_DIR, f'batch_viz_cell3_mp_cone{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    G_cone   = bd['G_cone']
    pos_cone = bd['pos_cone']
    fill     = bd['fill']
    seed_src = bd['seed_src']
    seed_dst = bd['seed_dst']
    _nc, _ns, _nec, _nlw = [], [], [], []
    for _n in G_cone.nodes():
        if _n == seed_src:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_A);   _nlw.append(2.5)
        elif _n == seed_dst:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_B);   _nlw.append(2.5)
        else:
            _nc.append(fill(_n)); _ns.append(130)
            _nec.append('#2c3e50'); _nlw.append(0.5)
    nx.draw_networkx_nodes(G_cone, pos_cone, ax=ax,
                           node_color=_nc, node_size=_ns,
                           edgecolors=_nec, linewidths=_nlw)
    nx.draw_networkx_edges(G_cone, pos_cone, ax=ax,
                           edge_color='#5d6d7e', width=2.0,
                           arrows=True, alpha=0.8,
                           connectionstyle='arc3,rad=0.08', arrowsize=20)
    ax.legend(handles=[
        mpatches.Patch(color=_C2_A,    label=f'Party {bd["pa"]}'),
        mpatches.Patch(color=_C2_B,    label=f'Party {bd["pb"]}'),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb"]})'),
    ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {G_cone.number_of_nodes()} nodes · '
        f'{G_cone.number_of_edges()} edges\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(f'Cell 3: {_K_HOPS}-hop MP cone — party colours', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL3, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL3}")


# ==============================================================
# ===== CELL 3b: PER-PARTY VIEW OF MP CONE (2 × N grid) ======
# ==============================================================

_VIZ_CELL3B = os.path.join(OUTPUT_DIR, f'batch_viz_cell3b_party_views{OUTPUT_SUFFIX}.pdf')

def _cone_layers(G_cone, seed_src, seed_dst, party_nodes, party_color):
    grey = [n for n in G_cone.nodes()
            if n not in party_nodes and n not in {seed_src, seed_dst}]
    vis_nds, vis_nc, vis_ns, vis_nec, vis_nlw = [], [], [], [], []
    for _n in G_cone.nodes():
        if _n not in party_nodes and _n not in {seed_src, seed_dst}:
            continue
        vis_nds.append(_n)
        if _n == seed_src:
            vis_nc.append(_C2_SEED); vis_ns.append(280)
            vis_nec.append(_C2_A);   vis_nlw.append(2.5)
        elif _n == seed_dst:
            vis_nc.append(_C2_SEED); vis_ns.append(280)
            vis_nec.append(_C2_B);   vis_nlw.append(2.5)
        else:
            vis_nc.append(party_color); vis_ns.append(130)
            vis_nec.append('#2c3e50'); vis_nlw.append(0.5)
    return grey, vis_nds, vis_nc, vis_ns, vis_nec, vis_nlw

fig, axes = plt.subplots(2, _N, figsize=(8 * _N, 22))
if _N == 1: axes = axes.reshape(2, 1)
for col, bd in enumerate(_batches):
    ax_a, ax_b = axes[0, col], axes[1, col]
    G_cone    = bd['G_cone']
    pos_cone  = bd['pos_cone']
    node_bank = bd['node_bank']
    seed_src  = bd['seed_src']
    seed_dst  = bd['seed_dst']
    for ax, party, party_color, p_nodes, p_bank in [
        (ax_a, bd['pa'], _C2_A, bd['pa_nodes'], bd['pa']),
        (ax_b, bd['pb'], _C2_B, bd['pb_nodes'], bd['pb']),
    ]:
        grey, vis_nds, vis_nc, vis_ns, vis_nec, vis_nlw = _cone_layers(
            G_cone, seed_src, seed_dst, p_nodes, party_color)
        if grey:
            nx.draw_networkx_nodes(G_cone, pos_cone, ax=ax, nodelist=grey,
                                   node_color=_C2_NONE, node_size=80,
                                   edgecolors='none', linewidths=0, alpha=0.4)
        nx.draw_networkx_nodes(G_cone, pos_cone, ax=ax, nodelist=vis_nds,
                               node_color=vis_nc, node_size=vis_ns,
                               edgecolors=vis_nec, linewidths=vis_nlw, alpha=0.9)
        _obs   = [(s, d) for s, d in G_cone.edges()
                  if node_bank.get(s) == p_bank or node_bank.get(d) == p_bank]
        _unobs = [(s, d) for s, d in G_cone.edges()
                  if node_bank.get(s) != p_bank and node_bank.get(d) != p_bank]
        if _unobs:
            nx.draw_networkx_edges(G_cone, pos_cone, ax=ax, edgelist=_unobs,
                                   edge_color='#5d6d7e', width=0.5,
                                   arrows=True, alpha=0.2,
                                   connectionstyle='arc3,rad=0.08', arrowsize=14)
        if _obs:
            nx.draw_networkx_edges(G_cone, pos_cone, ax=ax, edgelist=_obs,
                                   edge_color='#5d6d7e', width=2.0,
                                   arrows=True, alpha=0.8,
                                   connectionstyle='arc3,rad=0.08', arrowsize=20)
        ax.legend(handles=[
            mpatches.Patch(color=party_color, label=f'Party {party}'),
            mpatches.Patch(color=_C2_NONE,    label='Not observed by this party'),
            mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                           label=f'Seed src ({bd["pa"]})'),
            mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                           label=f'Seed dst ({bd["pb"]})'),
        ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
        ax.set_title(
            f'Batch {bd["batch_idx"]} · Party {party} — '
            f'{len(vis_nds)}/{G_cone.number_of_nodes()} nodes observed  '
            f'({len(grey)} not observed)',
            fontsize=17,
        )
        ax.axis('off')
fig.suptitle(
    f'Cell 3b: Per-party cone views — grey = in cone but party cannot observe',
    fontsize=17,
)
plt.tight_layout()
plt.savefig(_VIZ_CELL3B, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL3B}")


# ==============================================================
# ===== CELL 4: LAUNDERING ENRICHMENT IN MP CONE ==============
# ==============================================================

_VIZ_CELL4 = os.path.join(OUTPUT_DIR, f'batch_viz_cell4_laund_cone{OUTPUT_SUFFIX}.pdf')
fig, axes = plt.subplots(1, _N, figsize=(8 * _N, 14))
if _N == 1: axes = [axes]
for ax, bd in zip(axes, _batches):
    G_cone   = bd['G_cone']
    pos_cone = bd['pos_cone']
    fill     = bd['fill']
    seed_src = bd['seed_src']
    seed_dst = bd['seed_dst']
    _nc, _ns, _nec, _nlw = [], [], [], []
    for _n in G_cone.nodes():
        if _n == seed_src:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_A);   _nlw.append(2.5)
        elif _n == seed_dst:
            _nc.append(_C2_SEED); _ns.append(280)
            _nec.append(_C2_B);   _nlw.append(2.5)
        else:
            _nc.append(fill(_n)); _ns.append(130)
            _nec.append('#2c3e50'); _nlw.append(0.5)
    _ec, _ew, _n_laund = [], [], 0
    for _s, _d in G_cone.edges():
        _is_l = (_s, _d) in bd['laund_pairs']
        if _is_l:
            _ec.append(_C4_LAUND); _ew.append(4.0); _n_laund += 1
        else:
            _ec.append(_C4_LEGIT); _ew.append(1.5)
    print(f"  Batch {bd['batch_idx']} — cone {G_cone.number_of_nodes()} nodes · "
          f"{G_cone.number_of_edges()} edges · laundering pairs: {_n_laund}")
    nx.draw_networkx_nodes(G_cone, pos_cone, ax=ax,
                           node_color=_nc, node_size=_ns,
                           edgecolors=_nec, linewidths=_nlw)
    nx.draw_networkx_edges(G_cone, pos_cone, ax=ax,
                           edge_color=_ec, width=_ew,
                           arrows=True, alpha=0.85,
                           connectionstyle='arc3,rad=0.08', arrowsize=20)
    ax.legend(handles=[
        mpatches.Patch(color=_C2_A,    label=f'Party {bd["pa"]}'),
        mpatches.Patch(color=_C2_B,    label=f'Party {bd["pb"]}'),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb"]})'),
        mpatches.Patch(color=_C4_LAUND, label='Laundering edge'),
        mpatches.Patch(color=_C4_LEGIT, label='Legitimate edge'),
    ], loc='upper right', fontsize=12, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'Batch {bd["batch_idx"]} — {G_cone.number_of_nodes()} nodes\n'
        f'seed {bd["seed_gid"]}: {bd["pa"]} → {bd["pb"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')
fig.suptitle(
    f'Cell 4: {_K_HOPS}-hop MP cone — red = edge pair carries ≥1 laundering tx',
    fontsize=17,
)
plt.tight_layout()
plt.savefig(_VIZ_CELL4, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {_VIZ_CELL4}")


# ==============================================================
# ===== DONE ===================================================
# ==============================================================

print(f"\nAll outputs written to: {HPC_OUTPUT_DIR}")
print("Contents:")
for _f in sorted(os.listdir(HPC_OUTPUT_DIR)):
    _fp = os.path.join(HPC_OUTPUT_DIR, _f)
    print(f"  {_f:50s}  {os.path.getsize(_fp) / 1024:8.1f} KB")
