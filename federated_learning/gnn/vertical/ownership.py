"""Ownership mapping functions for vertical federated learning.

These functions determine which nodes/accounts are owned by which party,
and what data needs to be exchanged between parties.
"""

import numpy as np
import torch
from federated_learning.parallel import parallel_party_execute

# batch_banks = parties
# batch_banks = manager.sr_parties.keys()
# batch_num = None
# get_ownership_mappings(mode, manager, parties, None)

def get_ownership_mappings(mode, manager, batch_banks, batch_num, max_workers=None):
    """Get ownership mappings for all banks in a batch."""
    df_ids = manager.data[f'{mode}_data']
    mode_parties = manager.get_parties_for_mode(mode)
    batch_parties = {bid: mode_parties[bid] for bid in batch_banks}

    def _map(bank_id, party):
        graph_df = party.procs_data[f'{mode}_data']['df'] if batch_num is None else party.ctx[mode][batch_num]['graph_data']
        indices = graph_df.edge_attr[:, 0].tolist()
        _get_ownership_mappings(mode, party, bank_id, df_ids, graph_df, indices, batch_num)

    parallel_party_execute(batch_parties, _map, max_workers=max_workers)


def _get_ownership_mappings(mode, party, bank_id, df_ids, graph_df, indices, batch_num):
    """Internal: Calculate ownership mappings for a single party."""
    party_txns = df_ids.iloc[indices, :]

    from_owned = party_txns[party_txns['From Bank'] == bank_id]['from_id']
    to_owned = party_txns[party_txns['To Bank'] == bank_id]['to_id']
    owned_accounts = set(from_owned) | set(to_owned)
    visible_accounts = set(party_txns['from_id']) | set(party_txns['to_id'])

    global_ids = df_ids.iloc[np.array(indices), :][['from_id', 'to_id']]
    local_ids = graph_df.edge_index

    local_to_global = dict(zip(np.array(torch.concat([local_ids[0], local_ids[1]])),
                            np.concatenate([global_ids['from_id'], global_ids['to_id']])))
    global_to_local = {v: k for k, v in local_to_global.items()}
    owned_accounts_local = {global_to_local[node_id] for node_id in owned_accounts}

    party.ctx[mode][batch_num]['owned_accounts'] = owned_accounts
    party.ctx[mode][batch_num]['visible_accounts'] = visible_accounts
    party.ctx[mode][batch_num]['local_to_global'] = local_to_global
    party.ctx[mode][batch_num]['global_to_local'] = global_to_local
    party.ctx[mode][batch_num]['owned_accounts_local'] = owned_accounts_local


def get_nodes_to_send(mode, manager, batch_banks, batch_num):
    """Determine which nodes each party needs to send to other parties."""
    df_ids = manager.data[f'{mode}_data']
    mode_parties = manager.get_parties_for_mode(mode)
    # Reset nodes_to_send for all batch parties so stale entries from previous
    # batches (same batch_num key) don't carry over.
    for bank_id in batch_banks:
        mode_parties[bank_id].ctx[mode][batch_num]['nodes_to_send'] = {}
    processed_pairs = set()
    for bank_id in batch_banks:
        party = mode_parties[bank_id]
        processed_pairs = _get_nodes_to_send(mode, manager, df_ids, party, bank_id, processed_pairs, batch_num)


def _get_nodes_to_send(mode, manager, df_ids, party, bank_id, processed_pairs, batch_num):
    """Internal: Calculate nodes to send for a single party."""
    graph_data = party.procs_data[f'{mode}_data']['df'] if batch_num is None else party.ctx[mode][batch_num]['graph_data']

    for inner_bank_id, ints in party.ctx[mode][batch_num]['intersects'].items():

        pair = tuple(sorted([bank_id, inner_bank_id]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        idx_to_get = graph_data.edge_attr[ints,0].tolist()
        shared_txns = df_ids.iloc[idx_to_get,:][['from_id', 'to_id']]
        shared_txns_set = set(shared_txns.stack())

        my_owned_shared = sorted(shared_txns_set & party.ctx[mode][batch_num]['owned_accounts']) if len(party.ctx[mode][batch_num]['owned_accounts']) > 0 else []
        my_local = [party.ctx[mode][batch_num]['global_to_local'][gid] for gid in my_owned_shared]

        party.ctx[mode][batch_num]['nodes_to_send'][inner_bank_id] = {
            'local_indices': torch.LongTensor(my_local),
            'global_ids': my_owned_shared,
        }

        other_party = manager.get_parties_for_mode(mode)[inner_bank_id]
        other_owned_shared = sorted(shared_txns_set & other_party.ctx[mode][batch_num]['owned_accounts']) if len(other_party.ctx[mode][batch_num]['owned_accounts']) > 0 else []
        other_local = [other_party.ctx[mode][batch_num]['global_to_local'][acc] for acc in other_owned_shared]

        other_party.ctx[mode][batch_num]['nodes_to_send'][bank_id] = {
            'local_indices': torch.LongTensor(list(other_local)),
            'global_ids': list(other_owned_shared),
        }

    return processed_pairs
