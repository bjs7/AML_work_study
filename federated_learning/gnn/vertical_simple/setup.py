"""Setup for simplified vertical FL (no intersection tracking or ownership mapping).

No per-layer embedding exchange means we only need:
  - manager.ctx[mode][batch_num]['batch_labels']  — transactions to predict on
  - manager.ctx[mode][batch_num]['batch_parties']  — which banks participate
  - party.ctx[mode][batch_num]['graph_data']       — local subgraph per party

No intersects, ownership_mappings, or nodes_to_send are populated.
"""

import numpy as np
import torch
from collections import defaultdict

from data.data_utils import GraphData
from training.parallel import parallel_party_execute


def init_context_simple(manager):
    """Initialize context structures for manager and all parties."""
    modes = ['train', 'vali', 'test'] if 'test_data' in manager.data else ['train', 'vali']
    manager.ctx = {mode: defaultdict(lambda: defaultdict(dict)) for mode in modes}

    all_parties = manager.get_parties_for_mode(modes[-1])
    for party in all_parties.values():
        party.ctx = {mode: defaultdict(lambda: defaultdict(dict)) for mode in modes}


def setup_non_batching_simple(manager):
    """Set up batch labels and parties for non-batching mode.

    Mirrors setup_non_batching_data from vertical/setup.py.
    vali_data and test_data are cumulative, so we slice off the
    mode-specific rows (matching the original setup logic).
    """
    train_size = manager.data['train_data'].shape[0]

    manager.ctx['train'][None]['batch_labels'] = manager.data['train_data'][['From Bank', 'To Bank', 'Is Laundering']]
    manager.ctx['train'][None]['batch_parties'] = list(manager.parties.keys())

    manager.ctx['vali'][None]['batch_labels'] = manager.data['vali_data'][['From Bank', 'To Bank', 'Is Laundering']][train_size:]
    manager.ctx['vali'][None]['batch_parties'] = list(manager.get_parties_for_mode('vali').keys())

    if 'test_data' in manager.data:
        vali_size = manager.data['vali_data'].shape[0]
        manager.ctx['test'][None]['batch_labels'] = manager.data['test_data'][['From Bank', 'To Bank', 'Is Laundering']][vali_size:]
        manager.ctx['test'][None]['batch_parties'] = list(manager.get_parties_for_mode('test').keys())


def setup_simple_batching(manager, mode, batch_size=8192):
    """Generate batch subgraphs for each mode — no exchange setup.

    Mirrors gen_batch_data_simple from vertical/batching.py but skips
    get_batch_intersects, get_ownership_mappings, and get_nodes_to_send,
    since the simple forward pass needs none of those.
    """
    mode_parties = manager.get_parties_for_mode(mode)
    max_workers = getattr(manager.args['fl_parser'], 'max_workers', None)
    indices = manager.data[f'{mode}_data'].index.to_numpy().copy()

    if mode == 'train':
        np.random.shuffle(indices)

    batch_starts = range(0, len(indices), batch_size)
    manager.ctx[mode]['num_batches'] = len(batch_starts)

    df_labels = manager.data[f'{mode}_data'][['From Bank', 'To Bank', 'Is Laundering']]
    mode_party_set = set(mode_parties.keys())

    for batch_num, start in enumerate(batch_starts):
        batch_indices = indices[start:start + batch_size]
        batch_bl = df_labels.loc[batch_indices]
        manager.ctx[mode][batch_num]['batch_labels'] = batch_bl[
            batch_bl['From Bank'].isin(mode_party_set) | batch_bl['To Bank'].isin(mode_party_set)]

        batch_global_ids = torch.tensor(batch_indices, dtype=torch.long)

        def _build_subgraph(bank_id, party):
            party_graph = party.procs_data[f'{mode}_data']['df']
            mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)
            if mask.sum() == 0:
                return None
            matched_edge_index = party_graph.edge_index[:, mask]
            matched_edge_attr = party_graph.edge_attr[mask]
            sub_nodes = matched_edge_index.reshape(-1).unique()
            node_remap = torch.zeros(party_graph.x.shape[0], dtype=torch.long)
            node_remap[sub_nodes] = torch.arange(len(sub_nodes), dtype=torch.long)
            party.ctx[mode][batch_num]['graph_data'] = GraphData(
                x=party_graph.x[sub_nodes],
                edge_index=node_remap[matched_edge_index],
                edge_attr=matched_edge_attr,
            )
            return bank_id

        results = parallel_party_execute(mode_parties, _build_subgraph, max_workers=max_workers)
        banks_to_use = [bid for bid, result in results.items() if result is not None]
        manager.ctx[mode][batch_num]['batch_parties'] = banks_to_use


def setup_vertical_simple(manager, batching=True, batching_mode='simple'):
    """Set up simplified vertical FL — no intersection tracking or ownership mapping.

    Supported batching_mode values:
        'simple' (default): pre-computed batch subgraphs, no exchange setup.
        'lazy_link_neighbor': LinkNeighborLoader per epoch (memory-efficient),
            subgraphs computed on-the-fly without exchange data.
        non-batching (batching=False): full party graph per forward pass.

    Args:
        manager: The FL manager instance.
        batching: Whether to use batching.
        batching_mode: Which batch strategy to use when batching=True.
    """
    all_modes = ['train', 'vali', 'test'] if 'test_data' in manager.data else ['train', 'vali']

    init_context_simple(manager)

    if not batching:
        setup_non_batching_simple(manager)
    elif batching_mode == 'lazy_link_neighbor':
        from federated_learning.gnn.vertical.batching import setup_lazy_batch_loader
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            setup_lazy_batch_loader(manager, mode, batch_size=batch_size)
    else:
        # 'simple' or any other value: index-based batching without exchange
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            setup_simple_batching(manager, mode, batch_size=batch_size)
