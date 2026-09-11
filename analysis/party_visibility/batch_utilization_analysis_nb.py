# %%
"""
Batch utilization analysis — statistics and tables.

Produces Tables 1a, 1b, 2 and their CSVs/LaTeX files. Covers:
  - MP cone coverage: how much of the reverse-BFS cone each party sees
  - Party batch coverage: fraction of each batch each party processes

For graph visualizations run batch_viz_explore_nb.py.
"""

import sys
import os

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
from federated_learning.gnn.fedgraph.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn

# %%

# extend such that the visualization is conditioned on the pattern, 
# or whether it is observable by one or two parties (like whether both parties are in the comparable split) 
# or do one for comparable and one for system?

# in theory simple vertical gnn could be sped up? Like if parties only calculate embeddings for the edges 
# that are seed batches? So like parties do this now? But what if they ignored the 
# transactions they have, that are not one of the seeds? But they have to tell that those transactions are not 
# message passing to the transactions that are seed transactions

# more than one seed in a transaction, or like them being connected within 2 hops? Like does the seed transaction
# have illicit transactions within its 2-hop neighborhood, one could look at stats for that.

# using just the train data here right?


# One could also look at extending the batch stats, like look at matching the edges with the IDs of 
# the laundering attempts. Like such that one could see, for a given laundering attempt how does it get fragmented 
# or like how does it look in the full batch. How does it get fragemented of how much of it is lost in the 
# local subgraphs. One thing to remember, is that for a batch it might not be the case that all the transactions 
# of a laundering attempt are seed nodes, and so in such a case one would look at how much of the pattern is 
# observed on average of the batches? 


# %%

# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

N_BATCHES   = 5     # batches to sample for statistics (Tables 1 & 2)
OUTPUT_BASE = 'batch_stats'

_EVAL_MODE = 'comparable'  # 'comparable' | 'system'

_ARGS = [
    '--fl_algo',       'SplitFed',
    '--model',         'GINe',
    '--size',          'small',
    '--ir',            'HI',
    '--batching',
    '--batching_mode', 'lazy_link_neighbor',
    '--ibm_hp',
    '--emlps',
    '--eval_mode',     _EVAL_MODE,
    #'--testing', '--testing_frac', '0.05'
]

sys.argv = ['batch_utilization_analysis_nb'] + _ARGS


# %%

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
print("Setting up parties (this may take a while)...")
manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)
print("Party setup complete.")

from federated_learning.gnn.splitfed import setup
setup.setup_splitfed(manager, batching=True, batching_mode='lazy_link_neighbor')
manager.setup_model(ibm_gnn, laundering_values_test)

# --- Attempt metadata — index matches seed_global_ids (universal_features_restructure passes AttemptID through) ---
_train_df = manager.data['train_data']
_raw_meta  = _train_df[['AttemptID', 'Pattern']]
_valid_att = _raw_meta[_raw_meta['AttemptID'] >= 0]
_attempt_to_train_eids = (
    _valid_att.groupby('AttemptID').apply(lambda g: set(g.index)).to_dict()
)
print(f"Attempt metadata loaded: {len(_attempt_to_train_eids)} unique attempts in train split.")


# %%

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

print(f"Sampling {N_BATCHES} batches from '{mode}' loader...")

for batch_idx, batch in enumerate(manager.loaders[mode]):
    if batch_idx >= N_BATCHES:
        break

    # --- full batch graph from LinkNeighborLoader ---
    n_batch_edges = batch.edge_attr.shape[0]
    n_batch_nodes = batch.x.shape[0]

    # --- seed transactions ---
    seed_global_ids = batch.edge_label.long().numpy()
    n_seed_edges    = len(seed_global_ids)

    bl          = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    bl_filtered = bl[bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]
    n_seed_used = len(bl_filtered)
    seed_laundering_rate = bl['Is Laundering'].mean()

    # --- seed banks active in this batch ---
    seed_banks        = set(bl_filtered['From Bank'].values) | set(bl_filtered['To Bank'].values)
    seed_mode_parties = {k: v for k, v in mode_parties.items() if k in seed_banks}

    seed_ids_tensor  = torch.tensor(seed_global_ids, dtype=torch.long)
    batch_global_ids = torch.cat([batch.edge_attr[:, 0].long(), seed_ids_tensor]).unique()
    _batch_egids_set = set(batch_global_ids.tolist())

    # --- per party subgraph stats ---
    n_active = 0
    for bank_id, party in seed_mode_parties.items():
        party_graph = party.procs_data[f'{mode}_data']['df']
        mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)
        if mask.sum() == 0:
            continue

        matched_edge_index   = party_graph.edge_index[:, mask]
        n_party_edges        = int(mask.sum())
        n_party_nodes        = int(matched_edge_index.reshape(-1).unique().shape[0])

        matched_global_ids   = party_graph.edge_attr[mask, 0].long().numpy()
        party_seed_ids       = set(matched_global_ids) & set(seed_global_ids)
        n_party_seed_edges   = len(party_seed_ids)

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

    # --- 2-hop MP cone coverage (sampled over seed edges) ---
    _N_CONE_SAMPLE = min(25, n_seed_edges)
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

    # --- attempt pattern coverage (union of all banks across every seed of the attempt) ---
    # Two-pass: collect all banks per attempt first, then compute coverage.
    # This avoids the arbitrary first-seed dependence.
    #
    # KNOWN LIMITATIONS — consider if extending this metric:
    #
    # (1) _att_union measures "attempt edges in batch that at least one attempt-seed bank is
    #     party to", but being a party to an edge (from/to node) does not mean that edge's
    #     information flows INTO the seed's GNN embedding. In directed message passing,
    #     outgoing edges of a seed node (seed -> neighbour) do not contribute to the seed's
    #     representation — only incoming edges (predecessors) do. So _att_union counts both
    #     informationally relevant edges AND outgoing edges that carry info away from the seed.
    #
    # (2) The same directed-flow issue applies to any party-membership-based coverage metric
    #     (micro/macro per-seed averages would have the same blind spot).
    #
    # (3) A non-attempt seed S_other can drag an attempt edge T into the batch (T feeds into
    #     S_other's input neighbourhood). T is then in _att_in_batch, and if T's own banks
    #     overlap with _info['banks'] it counts toward _att_union — but S_other's bank (which
    #     actually sampled T) is never added to _info['banks'] because S_other is benign.
    #     So the bank that benefits from T is invisible to this metric.
    #
    # BETTER METRIC (not yet implemented):
    #     For each attempt seed node, compute its K-hop INPUT cone via reverse BFS (following
    #     incoming edges backward, i.e. the same traversal LinkNeighborLoader performs).
    #     Union those cones across all attempt seeds, then intersect with _attempt_eids.
    #     This gives "attempt edges whose information genuinely reaches at least one attempt
    #     seed's embedding within K message-passing rounds."
    #     Can then be aggregated as union (current style), micro (per seed, pool all),
    #     or macro (per seed, average within attempt then across attempts).
    #     Cost: one reverse BFS per unique attempt-seed node per batch — check runtime first.
    _att_seeds = {}  # aid -> {'pat': int, 'banks': set, 'n_seeds': int}
    for _gid in seed_global_ids:
        _gid_i = int(_gid)
        if _gid_i not in _all_lbl_idx or _gid_i >= len(_raw_meta):
            continue
        _aid = int(_raw_meta.at[_gid_i, 'AttemptID'])
        if _aid < 0:
            continue
        _cr_a = _df_lbls.loc[_gid_i]
        if _aid not in _att_seeds:
            _att_seeds[_aid] = {'pat': int(_raw_meta.at[_gid_i, 'Pattern']), 'banks': set(), 'n_seeds': 0}
        _att_seeds[_aid]['banks'].add(_cr_a['From Bank'])
        _att_seeds[_aid]['banks'].add(_cr_a['To Bank'])
        _att_seeds[_aid]['n_seeds'] += 1

    for _aid, _info in _att_seeds.items():
        _attempt_eids   = _attempt_to_train_eids.get(_aid, set())
        _n_total        = len(_attempt_eids)
        _att_in_batch   = _attempt_eids & _batch_egids_set
        _n_in_batch     = len(_att_in_batch)
        _att_batch_frac = _n_in_batch / _n_total if _n_total > 0 else float('nan')

        _all_bank_gids = set()
        for _bk in _info['banks']:
            _all_bank_gids |= _pgids_s.get(_bk, set())

        if _n_in_batch > 0:
            _att_union = len(_att_in_batch & _all_bank_gids) / _n_in_batch
            _att_neith = len(_att_in_batch - _all_bank_gids) / _n_in_batch
        else:
            _att_union = _att_neith = float('nan')

        attempt_records.append({
            'batch_idx':                batch_idx,
            'attempt_id':               _aid,
            'pattern':                  _info['pat'],
            'n_attempt_train':          _n_total,
            'n_attempt_in_batch':       _n_in_batch,
            'n_attempt_seeds_in_batch': _info['n_seeds'],
            'attempt_batch_frac':       _att_batch_frac,
            'attempt_union_cov':        _att_union,
            'attempt_neither_frac':     _att_neith,
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
        'n_unique_attempts':     len(_att_seeds),
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

    if (batch_idx + 1) % 10 == 0:
        print(f"  {batch_idx + 1}/{N_BATCHES} batches processed")

batch_df   = pd.DataFrame(batch_records)
party_df   = pd.DataFrame(party_records)
attempt_df = pd.DataFrame(attempt_records)


# %%

# ==============================================================
# ===== TABLES 1a / 1b: MP CONE COVERAGE — COMBINED ===========
# ==============================================================

_TABLE_DIR = os.path.expanduser(
    '~/projects/AML_work_study/writing/Experimental-Protocol/tables/party_visibility'
)
os.makedirs(_TABLE_DIR, exist_ok=True)

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

print("=== Table 1a: all seeds ===");     print(_cone_summary_a.to_string());    print()
print("=== Table 1a: illicit seeds ==="); print(_cone_summary_a_il.to_string()); print()
print("=== Table 1b: all seeds ===");     print(_cone_summary_b.to_string());    print()
print("=== Table 1b: illicit seeds ==="); print(_cone_summary_b_il.to_string()); print()

_cone_summary_a.to_csv(   os.path.join(_TABLE_DIR, 'cone_coverage_summary_a.csv'))
_cone_summary_b.to_csv(   os.path.join(_TABLE_DIR, 'cone_coverage_summary_b.csv'))
_cone_summary_a_il.to_csv(os.path.join(_TABLE_DIR, 'cone_coverage_illicit_a.csv'))
_cone_summary_b_il.to_csv(os.path.join(_TABLE_DIR, 'cone_coverage_illicit_b.csv'))

pd.concat({'All seeds': _cone_summary_a,    'Illicit seeds': _il_a}).to_csv(
    os.path.join(_TABLE_DIR, 'cone_coverage_combined_a.csv'))
pd.concat({'All seeds': _cone_summary_b,    'Illicit seeds': _il_b}).to_csv(
    os.path.join(_TABLE_DIR, 'cone_coverage_combined_b.csv'))


def _grouped_latex(groups, cols):
    """LaTeX tabular with row-group headers."""
    labels = [c.replace('_', r'\_') for c in cols]
    n = len(cols)
    out = [r'\begin{tabular}{l' + 'r' * n + '}', r'\hline',
           '  & ' + ' & '.join(labels) + r' \\', r'\hline']
    for grp, df in groups.items():
        out.append(r'  \textit{' + grp + '} & ' + ' & '.join([''] * n) + r' \\')
        for stat in df.index:
            vals = []
            for c in cols:
                if c in df.columns:
                    v = df.loc[stat, c]
                    vals.append(f'{v:.4f}' if not pd.isna(v) else '---')
                else:
                    vals.append('---')
            out.append(r'  \quad ' + stat + ' & ' + ' & '.join(vals) + r' \\')
        out.append(r'  \hline')
    out.append(r'\end{tabular}')
    return '\n'.join(out)

_groups_a = {'All seeds': _cone_summary_a, 'Illicit seeds': _il_a}
_groups_b = {'All seeds': _cone_summary_b, 'Illicit seeds': _il_b}

with open(os.path.join(_TABLE_DIR, 'cone_coverage_combined_a.tex'), 'w') as _f:
    _f.write(_grouped_latex(_groups_a, _cone_cols_a))
with open(os.path.join(_TABLE_DIR, 'cone_coverage_combined_b.tex'), 'w') as _f:
    _f.write(_grouped_latex(_groups_b, _cone_cols_b))

print(f"Saved to {_TABLE_DIR}")


# %%

# ==============================================================
# ===== TABLE 2: PARTY BATCH COVERAGE SUMMARY =================
# ==============================================================

_party_cols = [
    'party_edge_fraction', 'party_node_fraction', 'party_seed_coverage',
    'n_party_edges', 'n_party_nodes', 'n_party_seed_edges',
]
_party_summary = (
    party_df[_party_cols]
    .agg(['mean', 'std'])
    .round(4)
)
print("=== Table 2: Party Batch Coverage ===")
print(_party_summary.to_string())
print()
_party_summary.to_csv(os.path.join(_TABLE_DIR, 'party_batch_coverage_summary.csv'))
(_party_summary
 .rename(columns=lambda c: c.replace('_', r'\_'))
 .to_latex(os.path.join(_TABLE_DIR, 'party_batch_coverage_summary.tex'),
           escape=False))

print(f"Saved to {_TABLE_DIR}")

# %%

# ==============================================================
# ===== TABLE 3: ATTEMPT PATTERN COVERAGE =====================
# ==============================================================

_att_cols = [
    'attempt_batch_frac', 'n_attempt_seeds_in_batch',
    'attempt_union_cov', 'attempt_neither_frac',
]

if not attempt_df.empty:
    _att_overall = attempt_df[_att_cols].agg(['mean', 'std']).round(4)
    _att_by_pattern = attempt_df.groupby('pattern')[_att_cols].agg(['mean', 'std']).round(4)

    print("=== Table 3: Attempt coverage — overall ===")
    print(_att_overall.to_string()); print()
    print("=== Table 3: Attempt coverage — by pattern ===")
    print(_att_by_pattern.to_string()); print()

    _att_overall.to_csv(os.path.join(_TABLE_DIR, 'attempt_coverage_overall.csv'))
    _att_by_pattern.to_csv(os.path.join(_TABLE_DIR, 'attempt_coverage_by_pattern.csv'))
    attempt_df.to_csv(os.path.join(_TABLE_DIR, 'attempt_coverage_raw.csv'), index=False)
    print(f"Saved to {_TABLE_DIR}")
else:
    print("No attempt records collected.")

# %%
