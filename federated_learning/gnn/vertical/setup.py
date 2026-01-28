"""Setup functions for vertical federated learning.

These functions initialize the intersection tracking, context structures,
and node overlap mappings needed for vertical FL.
"""

import numpy as np
from collections import defaultdict

from .ownership import get_ownership_mappings, get_nodes_to_send


def init_context(manager):
    """Initialize context structures for manager and all parties."""
    manager.ctx = {
        'train': defaultdict(lambda: defaultdict(dict)),
        'eval': defaultdict(lambda: defaultdict(dict))
    }

    for party in manager.sr_parties.values():
        party.ctx = {
            'train': defaultdict(lambda: defaultdict(dict)),
            'eval': defaultdict(lambda: defaultdict(dict))
        }


def setup_intersects(manager):
    """Set up intersection tracking between parties.

    For each cross-bank transaction, records which banks share that transaction
    and tracks the indices for both train and eval modes.
    """
    mask = manager.data['eval_data']['From Bank'] != manager.data['eval_data']['To Bank']
    subset = manager.data['eval_data'][mask]
    train_indices = manager.data['train_data'].index

    for idx, from_bank, to_bank in zip(subset.index, subset['From Bank'].values, subset['To Bank'].values):
        from_bank, to_bank = int(from_bank), int(to_bank)
        modes = ['train', 'eval'] if idx in train_indices else ['eval']

        for mode in modes:
            for txt in ['intersects', 'ints_indices_by_bank']:
                manager.sr_parties[from_bank].ctx[mode][None][txt].setdefault(to_bank, []).append(idx)
                manager.sr_parties[to_bank].ctx[mode][None][txt].setdefault(from_bank, []).append(idx)

            manager.sr_parties[from_bank].ctx[mode][None].setdefault('ints_indices', []).append(idx)
            manager.sr_parties[to_bank].ctx[mode][None].setdefault('ints_indices', []).append(idx)


def setup_mappings(manager):
    """Convert intersection indices to local graph indices."""
    for party in manager.sr_parties.values():
        dt_indices = party.indices['train_indices'] + party.indices['vali_indices'] + party.indices['test_indices']
        for inner_bank_id, indices in party.ctx['eval'][None]['intersects'].items():
            if inner_bank_id in party.ctx['train'][None]['intersects'].keys():
                train_length = len(party.ctx['train'][None]['intersects'][inner_bank_id])
                party.ctx['train'][None]['intersects'][inner_bank_id] = np.where(np.isin(dt_indices, indices))[0][:train_length]
            party.ctx['eval'][None]['intersects'][inner_bank_id] = np.where(np.isin(dt_indices, indices))[0]


def get_nodes_overlap(manager, mode='train', batch_num=None):
    """Calculate which nodes are shared between parties."""
    for party in manager.sr_parties.values():

        party.ctx[mode][batch_num]['nodes_overlap'] = defaultdict(list)

        for inner_bank_id, intersects in party.ctx[mode][batch_num]['intersects'].items():
            party.ctx[mode][batch_num]['nodes_idx'][inner_bank_id] = party.procs_data[f'{mode}_data']['df'].edge_index.T[intersects].unique().tolist()

            for node_id in party.ctx[mode][batch_num]['nodes_idx'][inner_bank_id]:
                party.ctx[mode][batch_num]['nodes_overlap'][node_id].append(inner_bank_id)


def setup_non_batching_data(manager):
    """Set up batch labels and parties for non-batching mode."""
    manager.ctx['train'][None]['batch_labels'] = manager.data['train_data'][['From Bank', 'To Bank', 'Is Laundering']]
    manager.ctx['eval'][None]['batch_labels'] = manager.data['eval_data'][['From Bank', 'To Bank', 'Is Laundering']][manager.ctx['train'][None]['batch_labels'].shape[0]:]
    manager.ctx['train'][None]['batch_parties'] = manager.parties.keys()
    manager.ctx['eval'][None]['batch_parties'] = manager.sr_parties.keys()


def setup_vertical(manager, batching=True):
    """Complete setup for vertical federated learning.

    Args:
        manager: The FL manager instance
        batching: Whether to use batching (True) or process all data at once (False)
    """
    # Initialize context structures
    init_context(manager)

    # Set up intersection tracking
    setup_intersects(manager)
    setup_mappings(manager)

    # Get ownership mappings for non-batched mode
    get_ownership_mappings('train', manager, manager.parties.keys(), None)
    get_ownership_mappings('eval', manager, manager.sr_parties.keys(), None)

    # Get nodes to send for non-batched mode
    get_nodes_to_send('train', manager, manager.parties.keys(), None)
    get_nodes_to_send('eval', manager, manager.sr_parties.keys(), None)

    # Calculate node overlaps
    get_nodes_overlap(manager, mode='train')
    get_nodes_overlap(manager, mode='eval')

    # Set up batch data
    if not batching:
        setup_non_batching_data(manager)
    else:
        from .batching import gen_batch_data
        gen_batch_data(manager, 'train')
        gen_batch_data(manager, 'eval')
