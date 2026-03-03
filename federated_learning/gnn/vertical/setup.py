"""Setup functions for vertical federated learning.

These functions initialize the intersection tracking, context structures,
and node overlap mappings needed for vertical FL.
"""

import numpy as np
from collections import defaultdict

from .ownership import get_ownership_mappings, get_nodes_to_send


def init_context(manager):
    """Initialize context structures for manager and all parties."""
    modes = ['train', 'vali', 'test'] if 'test_data' in manager.data else ['train', 'vali']
    manager.ctx = {mode: defaultdict(lambda: defaultdict(dict)) for mode in modes}

    all_parties = manager.get_parties_for_mode(modes[-1])
    for party in all_parties.values():
        party.ctx = {mode: defaultdict(lambda: defaultdict(dict)) for mode in modes}


def setup_intersects(manager):
    """Set up intersection tracking between parties.

    For each cross-bank transaction, records which banks share that transaction
    and tracks the indices for train, vali, and test modes.
    """
    all_modes = list(manager.ctx.keys())  # ['train', 'vali', 'test'] or ['train', 'vali']

    # Use the largest dataset to find all cross-bank transactions
    full_key = 'test_data' if 'test_data' in manager.data else 'vali_data'
    mask = manager.data[full_key]['From Bank'] != manager.data[full_key]['To Bank']
    subset = manager.data[full_key][mask]

    if manager.args['data_parser'].eval_mode == 'comparable':
        mask = np.isin([subset['From Bank'], subset['To Bank']], list(manager.parties.keys()))
        mask = mask[0,:] & mask[1,:]
        subset = subset[mask]
    
    train_idx_set = set(manager.data['train_data'].index)
    vali_idx_set = set(manager.data['vali_data'].index)

    all_parties = manager.get_parties_for_mode(all_modes[-1])
    # list(manager.iter_parties())
    # all_parties = {bank_id: party for bank_id, party in list(manager.iter_parties())}

    for idx, from_bank, to_bank in zip(subset.index, subset['From Bank'].values, subset['To Bank'].values):
        from_bank, to_bank = int(from_bank), int(to_bank)

        # Determine which modes this transaction belongs to
        if idx in train_idx_set:
            modes = all_modes  # train, vali, test
        elif idx in vali_idx_set:
            modes = [m for m in all_modes if m != 'train']  # vali, test
        else:
            modes = [all_modes[-1]]  # test only (or vali if no test)

        for mode in modes:
            for txt in ['intersects', 'ints_indices_by_bank']:
                all_parties[from_bank].ctx[mode][None][txt].setdefault(to_bank, []).append(idx)
                all_parties[to_bank].ctx[mode][None][txt].setdefault(from_bank, []).append(idx)

            all_parties[from_bank].ctx[mode][None].setdefault('ints_indices', []).append(idx)
            all_parties[to_bank].ctx[mode][None].setdefault('ints_indices', []).append(idx)


def setup_mappings(manager):
    """Convert intersection indices to local graph indices."""
    all_modes = [m for m in manager.ctx.keys() if m != 'train']
    last_mode = all_modes[-1]  # 'test' or 'vali'

    all_parties = manager.get_parties_for_mode(last_mode)
    for party in all_parties.values():
        dt_indices = party.indices['train_indices'] + party.indices['vali_indices'] + party.indices['test_indices']

        for inner_bank_id, indices in party.ctx[last_mode][None]['intersects'].items():
            all_local = np.where(np.isin(dt_indices, indices))[0]

            # Map train intersects
            if inner_bank_id in party.ctx['train'][None].get('intersects', {}):
                train_length = len(party.ctx['train'][None]['intersects'][inner_bank_id])
                party.ctx['train'][None]['intersects'][inner_bank_id] = all_local[:train_length]

            # Map vali intersects
            if 'vali' in manager.ctx and inner_bank_id in party.ctx['vali'][None].get('intersects', {}):
                vali_length = len(party.ctx['vali'][None]['intersects'][inner_bank_id])
                party.ctx['vali'][None]['intersects'][inner_bank_id] = all_local[:vali_length]

            # Map last mode (test or vali) intersects — all positions
            if last_mode != 'vali':
                party.ctx[last_mode][None]['intersects'][inner_bank_id] = all_local


def get_nodes_overlap(manager, mode='train', batch_num=None):
    """Calculate which nodes are shared between parties."""
    mode_parties = manager.get_parties_for_mode(mode)
    for party in mode_parties.values():

        party.ctx[mode][batch_num]['nodes_overlap'] = defaultdict(list)

        for inner_bank_id, intersects in party.ctx[mode][batch_num]['intersects'].items():
            party.ctx[mode][batch_num]['nodes_idx'][inner_bank_id] = party.procs_data[f'{mode}_data']['df'].edge_index.T[intersects].unique().tolist()

            for node_id in party.ctx[mode][batch_num]['nodes_idx'][inner_bank_id]:
                party.ctx[mode][batch_num]['nodes_overlap'][node_id].append(inner_bank_id)


def setup_non_batching_data(manager):
    """Set up batch labels and parties for non-batching mode."""
    train_size = manager.data['train_data'].shape[0]

    manager.ctx['train'][None]['batch_labels'] = manager.data['train_data'][['From Bank', 'To Bank', 'Is Laundering']]
    manager.ctx['train'][None]['batch_parties'] = manager.parties.keys()

    manager.ctx['vali'][None]['batch_labels'] = manager.data['vali_data'][['From Bank', 'To Bank', 'Is Laundering']][train_size:]
    manager.ctx['vali'][None]['batch_parties'] = manager.get_parties_for_mode('vali').keys()

    if 'test_data' in manager.data:
        vali_size = manager.data['vali_data'].shape[0]
        manager.ctx['test'][None]['batch_labels'] = manager.data['test_data'][['From Bank', 'To Bank', 'Is Laundering']][vali_size:]
        manager.ctx['test'][None]['batch_parties'] = manager.get_parties_for_mode('test').keys()


def setup_vertical(manager, batching=True, batching_mode='neighbor_sample'):
    """Complete setup for vertical federated learning.

    Args:
        manager: The FL manager instance
        batching: Whether to use batching (True) or process all data at once (False)
        batching_mode: Which batch generation strategy to use when batching=True:
            'neighbor_sample' (default): iterative per-party neighbor sampling
            'simple': manager-driven index batching with subgraph() expansion
            'link_neighbor': LinkNeighborLoader on reference party + manual relabeling
    """
    all_modes = ['train', 'vali', 'test'] if 'test_data' in manager.data else ['train', 'vali']

    # Initialize context structures
    init_context(manager)

    # Set up intersection tracking
    setup_intersects(manager)
    setup_mappings(manager)

    # Get ownership mappings, nodes to send, and node overlaps for each mode
    for mode in all_modes:
        parties = manager.get_parties_for_mode(mode).keys()
        get_ownership_mappings(mode, manager, parties, None)
        get_nodes_to_send(mode, manager, parties, None)
        get_nodes_overlap(manager, mode=mode, batch_num=None)

    # Set up batch data
    if not batching:
        setup_non_batching_data(manager)
    elif batching_mode == 'simple':
        from .batching import gen_batch_data_simple
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            gen_batch_data_simple(manager, mode, batch_size=batch_size)
    elif batching_mode == 'link_neighbor':
        from .batching import gen_batch_data_link_neighbor
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            gen_batch_data_link_neighbor(manager, mode, batch_size=batch_size)
    elif batching_mode == 'lazy_link_neighbor':
        from .batching import setup_lazy_batch_loader
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            setup_lazy_batch_loader(manager, mode, batch_size=batch_size)
    else:  # 'neighbor_sample'
        from .batching import gen_batch_data
        batch_size = manager.args['data_parser'].batch_size
        for mode in all_modes:
            gen_batch_data(manager, mode, batch_size=batch_size)
