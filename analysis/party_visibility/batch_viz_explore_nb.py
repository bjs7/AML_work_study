# %%
"""
Batch visualization explorer — graph figures for the report.

Loads specific batches by index, computes BFS neighbourhoods and
MP cones, then produces Cells 0a–4. Includes a batch scanner to
find batches with dense, informative neighbourhoods before committing
to BATCH_INDICES.

For statistics/tables run batch_utilization_analysis_nb.py.
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
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos
import models.gnn_models
from federated_learning.gnn.fedgraph.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn


# %%

# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

N_SCAN_BATCHES = 20    # batches the batch scanner inspects; None = all

# ---- visualization configuration ----
BATCH_INDICES           = [1, 3]   # specific batches to visualize side-by-side
SEED_MUST_BE_LAUNDERING = True         # True → pin a laundering seed per batch
SEED_PICK               = 3    # which laundering seed within each batch (0 = first, ...)
SEED_OVERRIDE           = {1: 430559}  # {batch_idx: global_txn_id} — pin a specific transaction
_K_HOPS                 = 2            # GNN depth for visualizations

OUTPUT_SUFFIX  = '_explore'
OUTPUT_DIR     = os.path.expanduser(
    '~/projects/AML_work_study/writing/Experimental-Protocol/figs/batching'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# node-fill colours (shared across all viz cells)
_C2_A     = '#e74c3c'   # red    — party A
_C2_B     = '#3498db'   # blue   — party B
_C2_BOTH  = '#9b59b6'   # purple — both parties
_C2_NONE  = '#d5d8dc'   # grey   — neither party
_C2_SEED  = '#e67e22'   # orange — seed nodes
_C1B_CONE = '#f39c12'   # gold   — cone highlight (Cell 1b)
_C4_LAUND = '#c0392b'   # dark red — laundering edges (Cell 4)
_C4_LEGIT = '#aab7b8'   # grey   — legitimate edges (Cell 4)

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
    #'--testing', '--testing_frac', '0.15'
]

sys.argv = ['batch_viz_explore_nb'] + _ARGS


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


# %%

# ==============================================================
# ===== BATCH SCANNER =========================================
# ==============================================================
# For each batch (up to N_SCAN_BATCHES), runs K-hop BFS from the
# laundering seed selected by SEED_PICK % n_laund — same formula
# as LOAD BATCHES — and reports neighbourhood and cone sizes.
#
# Use this to pick BATCH_INDICES before running the viz cells:
#   1. Run this cell, look at neigh_n to find dense batches.
#   2. Set BATCH_INDICES to the chosen indices.
#   3. Adjust SEED_PICK or SEED_OVERRIDE if needed.
#   4. Re-run from LOAD BATCHES downward.
#
# Seeded with torch.manual_seed(1), identical to LOAD BATCHES,
# so batch N here is exactly batch N there.

_SCAN_TOP_SHOW = 20

_scan_mode   = 'train'
_scan_labels = manager.ctx[_scan_mode]['df_labels']

_scan_records = []
torch.manual_seed(1)
print(f"Scanning {'all' if N_SCAN_BATCHES is None else N_SCAN_BATCHES} "
      f"batches (K={_K_HOPS} hops, seed=1)…")

for _si, _sb in enumerate(manager.loaders[_scan_mode]):
    if N_SCAN_BATCHES is not None and _si >= N_SCAN_BATCHES:
        break

    _sg_ids    = _sb.edge_label.long().numpy()
    _known_pos = [i for i, g in enumerate(_sg_ids) if g in _scan_labels.index]
    _laund_pos = [i for i in _known_pos
                  if bool(_scan_labels.loc[_sg_ids[i], 'Is Laundering'])]

    if not _laund_pos:
        _scan_records.append({'batch_idx': _si, 'n_laund': 0,
                              'actual_pick': -1, 'from_bank': '', 'to_bank': '',
                              'neigh_n': 0, 'neigh_e': 0, 'cone_n': 0})
        continue

    _actual_pick = SEED_PICK % len(_laund_pos)
    _lp          = _laund_pos[_actual_pick]
    _g           = int(_sg_ids[_lp])

    _sc_src = _sb.edge_index[0].tolist()
    _sc_dst = _sb.edge_index[1].tolist()
    _sc_adj_u = defaultdict(set)
    _sc_radj  = defaultdict(set)
    for _s, _d in zip(_sc_src, _sc_dst):
        _sc_adj_u[_s].add(_d); _sc_adj_u[_d].add(_s)
        _sc_radj[_d].add(_s)

    _sc_gid2edge = {
        int(_sb.edge_label[_i]): (int(_sb.edge_label_index[0, _i]),
                                   int(_sb.edge_label_index[1, _i]))
        for _i in range(_sb.edge_label_index.shape[1])
    }

    if _g not in _sc_gid2edge:
        _seed_row = _scan_labels.loc[_g]
        _scan_records.append({'batch_idx': _si, 'n_laund': len(_laund_pos),
                              'actual_pick': _actual_pick,
                              'from_bank': _seed_row['From Bank'],
                              'to_bank':   _seed_row['To Bank'],
                              'neigh_n': 0, 'neigh_e': 0, 'cone_n': 0})
        continue

    _csrc, _cdst = _sc_gid2edge[_g]

    _fr, _vi = {_csrc, _cdst}, {_csrc, _cdst}
    for _ in range(_K_HOPS):
        _nxt = {_nb for _n in _fr for _nb in _sc_adj_u[_n] if _nb not in _vi}
        _vi |= _nxt; _fr = _nxt
    _ne = sum(1 for _s, _d in zip(_sc_src, _sc_dst) if _s in _vi and _d in _vi)

    _fr_c, _vi_c = {_csrc, _cdst}, {_csrc, _cdst}
    for _ in range(_K_HOPS):
        _nxt_c = {_nb for _n in _fr_c for _nb in _sc_radj[_n] if _nb not in _vi_c}
        _vi_c |= _nxt_c; _fr_c = _nxt_c

    _seed_row = _scan_labels.loc[_g]
    _scan_records.append({
        'batch_idx':   _si,
        'n_laund':     len(_laund_pos),
        'actual_pick': _actual_pick,
        'from_bank':   _seed_row['From Bank'],
        'to_bank':     _seed_row['To Bank'],
        'neigh_n':     len(_vi),
        'neigh_e':     _ne,
        'cone_n':      len(_vi_c),
    })

_scan_df    = pd.DataFrame(_scan_records)
_scan_laund = _scan_df[_scan_df['n_laund'] > 0].reset_index(drop=True)

print(f"Done. {len(_scan_df)} batches scanned, "
      f"{len(_scan_laund)} have ≥1 laundering seed.\n")
print(f"Seed formula: SEED_PICK={SEED_PICK} % n_laund — same index in every batch "
      f"(SEED_OVERRIDE applies in LOAD BATCHES only).\n")
print(f"First {min(_SCAN_TOP_SHOW, len(_scan_laund))} batches with laundering seeds:")
print(f"  {'idx':>4}  {'n_laund':>7}  {'pick':>4}  "
      f"{'from_bank':<12}  {'to_bank':<12}  "
      f"{'neigh_n':>7}  {'neigh_e':>7}  {'cone_n':>6}")
print('  ' + '-' * 76)
for _, _r in _scan_laund.head(_SCAN_TOP_SHOW).iterrows():
    print(f"  {int(_r['batch_idx']):>4}  {int(_r['n_laund']):>7}  "
          f"{int(_r['actual_pick']):>4}  "
          f"{str(_r['from_bank']):<12}  {str(_r['to_bank']):<12}  "
          f"{int(_r['neigh_n']):>7}  {int(_r['neigh_e']):>7}  "
          f"{int(_r['cone_n']):>6}")

print()
print(f"→ Set BATCH_INDICES to indices from the top of this table.")
print(f"→ SEED_PICK is already {SEED_PICK} — change it in CONFIGURATION to explore other seeds.")


# %%

# ==============================================================
# ===== LOAD BATCHES FOR VISUALIZATION ========================
# ==============================================================

_C0_MAX_NODES = 5000
_C0_N_SEEDS   = 2

mode        = 'train'
_all_labels = manager.ctx[mode]['df_labels']

print(f"Loading batches {BATCH_INDICES} from '{mode}' loader...")
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
            if _batch_idx in SEED_OVERRIDE:
                _override_gid = SEED_OVERRIDE[_batch_idx]
                _override_matches = [i for i in _laund_pos if _seed_gids_all[i] == _override_gid]
                if _override_matches:
                    _seed_idx_ov = torch.tensor([_override_matches[0]], dtype=torch.long)
                    print(f"  Seed override: using txn {_override_gid}")
                else:
                    print(f"  Warning: override txn {_override_gid} not found — falling back to SEED_PICK.")
                    _seed_idx_ov = torch.tensor([int(_laund_pos[SEED_PICK % len(_laund_pos)])], dtype=torch.long)
            else:
                _pick        = SEED_PICK % len(_laund_pos)
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

    _c0_init = {_seed_src, _seed_dst}
    _c0_step = max(1, _n_seeds_total // _C0_N_SEEDS)
    for _i in range(0, _n_seeds_total, _c0_step):
        _c0_init.add(int(_vb.edge_label_index[0, _i]))
        _c0_init.add(int(_vb.edge_label_index[1, _i]))
    _frontier_c0 = set(_c0_init)
    _visited_c0: set = set(_frontier_c0)
    while _frontier_c0 and len(_visited_c0) < _C0_MAX_NODES:
        _next_c0 = set()
        for _n in _frontier_c0:
            for _nb in _adj_ov[_n]:
                if _nb not in _visited_c0:
                    _next_c0.add(_nb)
        _remaining = _C0_MAX_NODES - len(_visited_c0)
        if len(_next_c0) <= _remaining:
            _visited_c0 |= _next_c0
            _frontier_c0  = _next_c0
        else:
            _visited_c0 |= set(list(_next_c0)[:_remaining])
            break
    _c0_edges = [
        (int(_s), int(_d)) for _s, _d in zip(_src_ov, _dst_ov)
        if int(_s) in _visited_c0 and int(_d) in _visited_c0 and int(_s) != int(_d)
    ]
    _is_full_c0 = len(_visited_c0) >= _vb.x.shape[0]
    _c0_label   = 'full batch' if _is_full_c0 else f'cap={_C0_MAX_NODES}'
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
        'batch_label':   f'Batch {_bi_enum + 1}',
        'pa_label':      'Party A',
        'pb_label':      'Party B',
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

# Assign consistent party labels across batches (order of first appearance).
# Same bank always gets the same letter, so readers can track parties across panels.
_all_banks = []
for bd in _batches:
    for _b in [bd['pa'], bd['pb']]:
        if _b not in _all_banks:
            _all_banks.append(_b)
_bank_labels = {b: f'Party {chr(65 + i)}' for i, b in enumerate(_all_banks)}
for bd in _batches:
    bd['pa_label'] = _bank_labels[bd['pa']]
    bd['pb_label'] = _bank_labels[bd['pb']]

_N = len(_batches)
print(f"\nAll {_N} batches loaded and pre-computed.")
print("Party label mapping:", {v: k for k, v in _bank_labels.items()})


# %%

# ==============================================================
# ===== CELL 0a: BATCH OVERVIEW (PLACEHOLDER) =================
# ==============================================================

_VIZ_CELL0A = os.path.join(OUTPUT_DIR, f'batch_viz_cell0a{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        f'{bd["batch_label"]} — {len(bd["visited_c0"])}/{bd["n_nodes_batch"]} nodes  '
        f'({bd["c0_label"]})\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 0a: Batch overview (capped at {_C0_MAX_NODES} nodes)', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL0A, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL0A}")


# %%

# ==============================================================
# ===== CELL 0b: BATCH WITH K-HOP NEIGHBOURHOOD HIGHLIGHTED ===
# ==============================================================

_VIZ_CELL0B = os.path.join(OUTPUT_DIR, f'batch_viz_cell0b{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
    _miss = f'  ({_nout} neigh. nodes below cap)' if _nout else ''
    ax.legend(handles=[
        mpatches.Patch(color='#3498db',  label=f'{_K_HOPS}-hop neighbourhood ({_nin} nodes)'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa_label"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb_label"]})'),
        mpatches.Patch(color='#d5d8dc', alpha=0.75, label='Rest of batch (greyed)'),
    ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'{bd["batch_label"]} — {_K_HOPS}-hop neighbourhood highlighted{_miss}\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 0b: Batch with {_K_HOPS}-hop neighbourhood highlighted', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL0B, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL0B}")


# %%

# ==============================================================
# ===== CELL 1: K-HOP NEIGHBOURHOOD OF SEED ==================
# ==============================================================

_VIZ_CELL1 = os.path.join(OUTPUT_DIR, f'batch_viz_cell1_neighborhood{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        f'{bd["batch_label"]} — {len(bd["visited_ov"])} nodes · {len(bd["ov_edges"])} edges\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 1: {_K_HOPS}-hop undirected neighbourhood of seed', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL1, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL1}")


# %%

# ==============================================================
# ===== CELL 1b: CONE HIGHLIGHTED IN NEIGHBOURHOOD ============
# ==============================================================

_VIZ_CELL1B = os.path.join(OUTPUT_DIR, f'batch_viz_cell1b_cone_in_neighborhood{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        mpatches.Patch(color=_C1B_CONE,  label=f'In {_K_HOPS}-hop MP cone ({_n_in_cone})'),
        mpatches.Patch(color='#3498db',  label='Neighbourhood but outside cone'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa_label"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb_label"]})'),
    ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'{bd["batch_label"]} — {len(bd["visited_ov"])} nodes  '
        f'·  cone: {_n_in_cone}\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 1b: {_K_HOPS}-hop neighbourhood — gold = feeds prediction via MP', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL1B, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL1B}")


# %%

# ==============================================================
# ===== CELL 2: PARTY VISIBILITY ON K-HOP NEIGHBOURHOOD =======
# ==============================================================

_VIZ_CELL2 = os.path.join(OUTPUT_DIR, f'batch_viz_cell2_visibility{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        mpatches.Patch(color=_C2_A,    label=bd["pa_label"]),
        mpatches.Patch(color=_C2_B,    label=bd["pb_label"]),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor='white', edgecolor='#f39c12', linewidth=1.5,
                       label=f'Gold border = in {_K_HOPS}-hop MP cone'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa_label"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb_label"]})'),
    ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'{bd["batch_label"]} — party visibility\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 2: Party visibility · {_K_HOPS}-hop neighbourhood  '
             f'(gold border = in MP cone · grey dim = neither party)', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL2, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL2}")


# %%

# ==============================================================
# ===== CELL 3: MP CONE — PARTY COLOURED =====================
# ==============================================================

_VIZ_CELL3 = os.path.join(OUTPUT_DIR, f'batch_viz_cell3_mp_cone{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        mpatches.Patch(color=_C2_A,    label=bd["pa_label"]),
        mpatches.Patch(color=_C2_B,    label=bd["pb_label"]),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa_label"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb_label"]})'),
    ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'{bd["batch_label"]} — {G_cone.number_of_nodes()} nodes · '
        f'{G_cone.number_of_edges()} edges\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(f'Cell 3: {_K_HOPS}-hop MP cone — party colours', fontsize=17)
plt.tight_layout()
plt.savefig(_VIZ_CELL3, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL3}")


# %%

# ==============================================================
# ===== CELL 3b: PER-PARTY VIEW OF MP CONE (2 × N grid) ======
# ==============================================================

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

for _fig_path, _party_name, _nodes_attr, _bank_attr, _party_color in [
    (os.path.join(OUTPUT_DIR, f'batch_viz_cell3b_party_a{OUTPUT_SUFFIX}.pdf'),
     'Party A', 'pa_nodes', 'pa', _C2_A),
    (os.path.join(OUTPUT_DIR, f'batch_viz_cell3b_party_b{OUTPUT_SUFFIX}.pdf'),
     'Party B', 'pb_nodes', 'pb', _C2_B),
]:
    fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
    if _N == 1: axes = [axes]
    for ax, bd in zip(axes, _batches):
        G_cone    = bd['G_cone']
        pos_cone  = bd['pos_cone']
        node_bank = bd['node_bank']
        seed_src  = bd['seed_src']
        seed_dst  = bd['seed_dst']
        p_nodes   = bd[_nodes_attr]
        p_bank    = bd[_bank_attr]

        grey, vis_nds, vis_nc, vis_ns, vis_nec, vis_nlw = _cone_layers(
            G_cone, seed_src, seed_dst, p_nodes, _party_color)
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
            mpatches.Patch(color=_party_color, label=_party_name),
            mpatches.Patch(color=_C2_NONE,     label='Not observed by this party'),
            mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                           label=f'Seed src ({bd["pa_label"]})'),
            mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                           label=f'Seed dst ({bd["pb_label"]})'),
        ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
        ax.set_title(
            f'{bd["batch_label"]} · {_party_name} — '
            f'{len(vis_nds)}/{G_cone.number_of_nodes()} nodes observed  '
            f'({len(grey)} not observed)',
            fontsize=17,
        )
        ax.axis('off')
    fig.suptitle(
        f'Cell 3b: {_party_name} cone view — grey = in cone but party cannot observe',
        fontsize=17,
    )
    plt.tight_layout()
    plt.savefig(_fig_path, bbox_inches='tight', dpi=200)
    plt.show()
    print(f"Saved: {_fig_path}")


# %%

# ==============================================================
# ===== CELL 4: LAUNDERING ENRICHMENT IN MP CONE ==============
# ==============================================================

_VIZ_CELL4 = os.path.join(OUTPUT_DIR, f'batch_viz_cell4_laund_cone{OUTPUT_SUFFIX}.pdf')

fig, axes = plt.subplots(1, _N, figsize=(13 * _N, 14))
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
        mpatches.Patch(color=_C2_A,    label=bd["pa_label"]),
        mpatches.Patch(color=_C2_B,    label=bd["pb_label"]),
        mpatches.Patch(color=_C2_BOTH, label='Both parties'),
        mpatches.Patch(color=_C2_NONE, label='Neither party'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_A, linewidth=2,
                       label=f'Seed src ({bd["pa_label"]})'),
        mpatches.Patch(facecolor=_C2_SEED, edgecolor=_C2_B, linewidth=2,
                       label=f'Seed dst ({bd["pb_label"]})'),
        mpatches.Patch(color=_C4_LAUND, label='Laundering edge'),
        mpatches.Patch(color=_C4_LEGIT, label='Legitimate edge'),
    ], loc='upper right', fontsize=14, handlelength=2, handleheight=1.2)
    ax.set_title(
        f'{bd["batch_label"]} — {G_cone.number_of_nodes()} nodes\n'
        f'seed {bd["seed_gid"]}: {bd["pa_label"]} → {bd["pb_label"]}  '
        f'(laundering={bd["is_laundering"]})',
        fontsize=17,
    )
    ax.axis('off')

fig.suptitle(
    f'Cell 4: {_K_HOPS}-hop MP cone — red = edge pair carries ≥1 laundering tx '
    f'(not necessarily same scheme as seed)',
    fontsize=17,
)
plt.tight_layout()
plt.savefig(_VIZ_CELL4, bbox_inches='tight', dpi=200)
plt.show()
print(f"Saved: {_VIZ_CELL4}")

# %%
