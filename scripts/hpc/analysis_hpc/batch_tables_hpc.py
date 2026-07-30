"""
HPC batch analysis — statistics and tables only.

Iterates all batches, samples all seed edges for cone metrics, and
writes CSV/LaTeX tables to hpc_output/.  No visualizations.

For graph visualizations run batch_viz_hpc.py.

Changes from the notebook version:
  - All batches processed (N_BATCHES = None)
  - All seed edges sampled for cone metrics (N_CONE_SAMPLE_MAX = None)
  - Non-interactive backend not required (no matplotlib used)
  - All outputs written to HPC_OUTPUT_DIR for one-shot download
  - Timing printed every 50 batches
"""

import sys
import os
import time

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    # scripts/hpc/analysis_hpc/ → scripts/hpc/ → scripts/ → repo root (4 parents)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import copy
import logging
from collections import defaultdict
import numpy as np
import torch
import pandas as pd

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos
import models.gnn_models
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn
from configs.paths import get_data_path


# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_BATCHES         = None   # None = all batches
N_CONE_SAMPLE_MAX = 200    # cone BFS samples per batch; None = all seed edges (very slow)

_EVAL_MODE = 'comparable'  # 'comparable' | 'system' — drives output subdir + _ARGS

HPC_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'hpc_output', _EVAL_MODE)
os.makedirs(HPC_OUTPUT_DIR, exist_ok=True)

OUTPUT_BASE = os.path.join(HPC_OUTPUT_DIR, 'batch_stats')
_TABLE_DIR  = HPC_OUTPUT_DIR

_ARGS = [
    '--fl_algo',       'FedGraphSimple',
    '--model',         'GINe',
    '--size',          'small',
    '--ir',            'HI',
    '--batching',
    '--batching_mode', 'lazy_link_neighbor',
    '--ibm_hp',
    '--emlps',
    '--eval_mode',     _EVAL_MODE,
]

sys.argv = ['batch_tables_hpc'] + _ARGS


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

# --- Attempt metadata (keyed by DataFrame row index, same as batch.edge_label) ---
_csv_path = (f"{get_data_path()}/AML_work_study/"
             f"formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
_raw_meta   = pd.read_csv(_csv_path, usecols=['AttemptID', 'Pattern'])
_train_eids = set(manager.ctx['train']['df_labels'].index)
_valid_att  = _raw_meta[_raw_meta.index.isin(_train_eids) & (_raw_meta['AttemptID'] >= 0)]
_attempt_to_train_eids = (
    _valid_att.groupby('AttemptID').apply(lambda g: set(g.index)).to_dict()
)
print(f"Attempt metadata loaded: {len(_attempt_to_train_eids)} unique attempts in train split.")


# ==============================================================
# =================== BATCH SAMPLING ==========================
# ==============================================================

def _nanmean(lst):
    """nanmean that returns nan without warning when all values are nan."""
    valid = [v for v in lst if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float('nan')

mode         = 'train'
mode_parties = manager.get_parties_for_mode(mode)

batch_records   = []
party_records   = []
attempt_records = []

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
    _batch_egids_set = set(batch_global_ids.tolist())

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
    _e2g_s = defaultdict(set)
    for _s, _d, _g in zip(_esrc_s, _edst_s, _egid_s):
        _e2g_s[(_s, _d)].add(_g)

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

    _unique_seed_nodes = (set(batch.edge_label_index[0, _cone_idx].tolist()) |
                          set(batch.edge_label_index[1, _cone_idx].tolist()))
    _bfs_cone_cache = {}
    for _node in _unique_seed_nodes:
        _fr, _vi = {_node}, {_node}
        for _ in range(_K_CONE):
            _nxt = {_nb for _n in _fr for _nb in _radj_s[_n] if _nb not in _vi}
            _vi |= _nxt; _fr = _nxt
        _bfs_cone_cache[_node] = _vi
    _bfs_neigh_cache = {}
    for _node in _unique_seed_nodes:
        _fr_u, _vi_u = {_node}, {_node}
        for _ in range(_K_CONE):
            _nxt_u = {_nb for _n in _fr_u for _nb in _adj_s_u[_n] if _nb not in _vi_u}
            _vi_u |= _nxt_u; _fr_u = _nxt_u
        _bfs_neigh_cache[_node] = _vi_u

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

        _vi   = _bfs_cone_cache[_csrc] | _bfs_cone_cache[_cdst]
        _vi_u = _bfs_neigh_cache[_csrc] | _bfs_neigh_cache[_cdst]
        _neigh_n_edges_i = sum(1 for _s, _d in zip(_esrc_s, _edst_s) if _s in _vi_u and _d in _vi_u)

        _cgids = {_g for _s, _d in zip(_esrc_s, _edst_s)
                  if _s in _vi and _d in _vi and (_s, _d) in _e2g_s
                  for _g in _e2g_s[(_s, _d)]}
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

    # --- attempt pattern coverage (all illicit seeds, deduplicated by AttemptID) ---
    _seen_attempts = set()
    for _gid in seed_global_ids:
        _gid_i = int(_gid)
        if _gid_i not in _all_lbl_idx or _gid_i >= len(_raw_meta):
            continue
        _aid = int(_raw_meta.at[_gid_i, 'AttemptID'])
        if _aid < 0 or _aid in _seen_attempts:
            continue
        _seen_attempts.add(_aid)
        _pat = int(_raw_meta.at[_gid_i, 'Pattern'])

        _attempt_eids   = _attempt_to_train_eids.get(_aid, set())
        _n_total        = len(_attempt_eids)
        _att_in_batch   = _attempt_eids & _batch_egids_set
        _n_in_batch     = len(_att_in_batch)
        _att_batch_frac = _n_in_batch / _n_total if _n_total > 0 else float('nan')

        _cr_a = _df_lbls.loc[_gid_i]
        _ca_a, _cb_a = _cr_a['From Bank'], _cr_a['To Bank']
        _ga_a = _pgids_s.get(_ca_a, set())
        _gb_a = _pgids_s.get(_cb_a, set())

        if _n_in_batch > 0:
            _att_from  = len(_att_in_batch & _ga_a) / _n_in_batch
            _att_to    = len(_att_in_batch & _gb_a) / _n_in_batch
            _att_union = len(_att_in_batch & (_ga_a | _gb_a)) / _n_in_batch
            _att_neith = len(_att_in_batch - _ga_a - _gb_a) / _n_in_batch
        else:
            _att_from = _att_to = _att_union = _att_neith = float('nan')

        attempt_records.append({
            'batch_idx':            batch_idx,
            'attempt_id':           _aid,
            'pattern':              _pat,
            'n_attempt_train':      _n_total,
            'n_attempt_in_batch':   _n_in_batch,
            'attempt_batch_frac':   _att_batch_frac,
            'attempt_from_cov':     _att_from,
            'attempt_to_cov':       _att_to,
            'attempt_union_cov':    _att_union,
            'attempt_neither_frac': _att_neith,
        })

    _mean_nesting = _nanmean(_cone_nesting_idx)
    _mean_l_laund = _nanmean(_cone_laund_frac_laund)
    batch_records.append({
        'batch_idx':             batch_idx,
        'n_batch_edges':         n_batch_edges,
        'n_batch_nodes':         n_batch_nodes,
        'n_seed_edges':          n_seed_edges,
        'n_seed_used':           n_seed_used,
        'seed_laundering_rate':  seed_laundering_rate,
        'n_active_parties':      n_active,
        'n_unique_attempts':     len(_seen_attempts),
        'batch_edge_laund_rate': _batch_edge_laund_rate,
        'cone_from_cov':         float(np.mean(_cone_from_cov))    if _cone_from_cov    else float('nan'),
        'cone_to_cov':           float(np.mean(_cone_to_cov))      if _cone_to_cov      else float('nan'),
        'cone_union_cov':        float(np.mean(_cone_union_cov))   if _cone_union_cov   else float('nan'),
        'cone_neither_frac':     float(np.mean(_cone_neither))     if _cone_neither     else float('nan'),
        'cone_unique_from':      float(np.mean(_cone_unique_from)) if _cone_unique_from else float('nan'),
        'cone_unique_to':        float(np.mean(_cone_unique_to))   if _cone_unique_to   else float('nan'),
        'cone_overlap_cov':      float(np.mean(_cone_overlap_cov)) if _cone_overlap_cov else float('nan'),
        'cone_nesting_idx':      _mean_nesting,
        'cone_laund_frac':       _nanmean(_cone_laund_frac_all),
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
        'cone_nesting_idx_il':   _nanmean(_cone_nesting_idx_il),
        'cone_n_nodes_il':       float(np.mean(_cone_n_nodes_il))        if _cone_n_nodes_il       else float('nan'),
        'cone_n_edges_il':       float(np.mean(_cone_n_edges_il))        if _cone_n_edges_il       else float('nan'),
        'neigh_n_nodes_il':      float(np.mean(_neigh_n_nodes_il))       if _neigh_n_nodes_il      else float('nan'),
        'neigh_n_edges_il':      float(np.mean(_neigh_n_edges_il))       if _neigh_n_edges_il      else float('nan'),
        'cone_asymmetry_il':     float(np.mean(_cone_asymmetry_il))      if _cone_asymmetry_il     else float('nan'),
    })

    if (batch_idx + 1) % 50 == 0:
        _elapsed = time.time() - _t0
        print(f"  {batch_idx + 1} batches in {_elapsed:.0f}s  ({_elapsed / (batch_idx + 1):.1f}s/batch)")

batch_df   = pd.DataFrame(batch_records)
party_df   = pd.DataFrame(party_records)
attempt_df = pd.DataFrame(attempt_records)
_elapsed_total = time.time() - _t0
print(f"Sampling done: {len(batch_df)} batches in {_elapsed_total:.0f}s")

batch_df.to_csv(f'{OUTPUT_BASE}_batch.csv', index=False)
party_df.to_csv(f'{OUTPUT_BASE}_party.csv', index=False)
attempt_df.to_csv(f'{OUTPUT_BASE}_attempt.csv', index=False)
print(f"Saved: {OUTPUT_BASE}_batch.csv")
print(f"Saved: {OUTPUT_BASE}_party.csv")
print(f"Saved: {OUTPUT_BASE}_attempt.csv")


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
# ===== TABLE 3: ATTEMPT PATTERN COVERAGE =====================
# ==============================================================

_att_cols = [
    'attempt_batch_frac',
    'attempt_from_cov', 'attempt_to_cov', 'attempt_union_cov', 'attempt_neither_frac',
]

if not attempt_df.empty:
    _att_overall = attempt_df[_att_cols].agg(['mean', 'std']).round(4)
    _att_overall.to_csv(os.path.join(_TABLE_DIR, 'attempt_coverage_overall.csv'))

    _att_by_pattern = (
        attempt_df.groupby('pattern')[_att_cols]
        .agg(['mean', 'std'])
        .round(4)
    )
    _att_by_pattern.to_csv(os.path.join(_TABLE_DIR, 'attempt_coverage_by_pattern.csv'))

    (_att_overall
     .rename(columns=lambda c: c.replace('_', r'\_'))
     .to_latex(os.path.join(_TABLE_DIR, 'attempt_coverage_overall.tex'), escape=False))

    print("Saved attempt coverage tables.")
else:
    print("No attempt records collected — attempt tables skipped.")


# ==============================================================
# ===== DONE ===================================================
# ==============================================================

print(f"\nAll outputs written to: {HPC_OUTPUT_DIR}")
print("Contents:")
for _f in sorted(os.listdir(HPC_OUTPUT_DIR)):
    _fp = os.path.join(HPC_OUTPUT_DIR, _f)
    print(f"  {_f:50s}  {os.path.getsize(_fp) / 1024:8.1f} KB")
