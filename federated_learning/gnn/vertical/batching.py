"""Batch generation functions for vertical federated learning."""

import copy
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from data.data_utils import GraphData

from .ownership import get_ownership_mappings, get_nodes_to_send


def gen_seed_values(manager, mode, banks_to_sample, df):
    """Generate seed values for neighbor sampling."""
    for bank_id in banks_to_sample:
        party = manager.parties[bank_id] if mode == 'train' else manager.sr_parties[bank_id]

        mask = (df['From Bank'] == bank_id) | (df['To Bank'] == bank_id)
        party.indexes = np.array(df[mask].index)

        reversed_dict = {v: k for k, v in party.procs_data[f'{mode}_data']['indices_mapping'].items()}

        party.seed_indices = [reversed_dict[idx] for idx in party.indexes]
        party.seed_nodes = party.procs_data[f'{mode}_data']['df'].edge_index[:,party.seed_indices].unique().tolist()
        party.sample_nodes = party.seed_nodes


def gen_loaders(manager, mode, banks_to_sample, layer_num):
    """Generate neighbor loaders for each bank."""
    shuffle_arg = True if mode == 'train' else False

    for bank_id in banks_to_sample:
        party = manager.parties[bank_id] if mode == 'train' else manager.sr_parties[bank_id]

        if len(party.sample_nodes) > 0:
            tmp_loader = NeighborLoader(party.procs_data[f'{mode}_data']['df'],
                                    num_neighbors=[100], input_nodes=party.sample_nodes,
                                                    batch_size=len(party.sample_nodes), shuffle=shuffle_arg,
                                                    transform=None)

            party.tmp_batch = next(iter(tmp_loader))
            party.matching_values = party.tmp_batch.n_id[np.isin(party.tmp_batch.n_id, list(party.ctx[mode][None]['nodes_overlap'].keys()))]
            party.subgraphs_nodes[layer_num] = party.tmp_batch.n_id.tolist()


def send_sample_nodes(manager, banks_to_sample, mode):
    """Send sampled nodes to other parties that share edges."""
    banks_to_add = []
    for bank_id in banks_to_sample:

        party = manager.sr_parties[bank_id]
        for inner_bank_id, nodes_ids in party.ctx[mode][None]['nodes_idx'].items():
            if len(np.array(party.matching_values)[np.isin(party.matching_values, nodes_ids)]) > 0:
                local_nodes_to_send = np.array(party.matching_values)[np.isin(party.matching_values, nodes_ids)]
                nodes_to_send = [party.ctx[mode][None]['local_to_global'][node] for node in local_nodes_to_send]
                manager.sr_parties[inner_bank_id].sample_nodes_to_add[party.bank_id] = nodes_to_send
                if inner_bank_id not in banks_to_sample and inner_bank_id not in banks_to_add:
                    banks_to_add.append(inner_bank_id)

    return banks_to_add


def receive_sample_nodes(manager, banks_to_sample, banks_to_add, layer_num, sample_neighbors, mode):
    """Receive sampled nodes from other parties."""
    for bank_id in (banks_to_sample + banks_to_add):

        party = manager.sr_parties[bank_id]
        party.sample_nodes = []

        for inner_bank_id, nodes in party.sample_nodes_to_add.items():
            mapped_nodes = [party.ctx[mode][None]['global_to_local'][node] for node in nodes]
            party.sample_nodes += mapped_nodes

        if bank_id in banks_to_sample:
            party.sample_nodes += party.tmp_batch.n_id[party.tmp_batch.batch_size:].tolist()

        if layer_num == len(sample_neighbors):
            party.final_subgraph_nodes = []
            for layer_number, nodes in party.subgraphs_nodes.items():
                party.final_subgraph_nodes += nodes
            party.final_subgraph_nodes += party.sample_nodes
            party.final_subgraph_nodes = list(set(party.final_subgraph_nodes))

    return banks_to_sample + banks_to_add


def send_receive_final_nodes(manager, banks_to_sample, mode):
    """Exchange final subgraph nodes between parties."""
    banks_to_add = []
    for bank_id in banks_to_sample:

        party = manager.sr_parties[bank_id]
        for inner_bank_id, nodes_ids in party.ctx[mode][None]['nodes_idx'].items():
            if len(np.array(party.final_subgraph_nodes)[np.isin(party.final_subgraph_nodes, nodes_ids)]) > 0:
                local_nodes_to_send = np.array(party.final_subgraph_nodes)[np.isin(party.final_subgraph_nodes, nodes_ids)]
                nodes_to_send = [party.ctx[mode][None]['local_to_global'][node] for node in local_nodes_to_send]
                manager.sr_parties[inner_bank_id].sample_nodes_to_add[party.bank_id] = nodes_to_send
                if inner_bank_id not in banks_to_sample and inner_bank_id not in banks_to_add:
                    banks_to_add.append(inner_bank_id)

    for bank_id in banks_to_sample:
        party = manager.sr_parties[bank_id]
        for inner_bank_id, nodes in party.sample_nodes_to_add.items():
            mapped_nodes = [party.ctx[mode][None]['global_to_local'][node] for node in nodes]
            party.final_subgraph_nodes += mapped_nodes
        party.final_subgraph_nodes = list(set(party.final_subgraph_nodes))


def gen_batch_graph(manager, all_nodes, mode, batch_num=0, bank_id=0):
    """Generate batch subgraph for a party, filtering out isolated nodes."""
    edge_index_orig = manager.sr_parties[bank_id].procs_data[f'{mode}_data']['df'].edge_index
    edge_attr_orig = manager.sr_parties[bank_id].procs_data[f'{mode}_data']['df'].edge_attr

    # First extract without relabeling to find nodes that actually have edges
    final_edge_index_tmp, _ = subgraph(subset=all_nodes,
        edge_index=edge_index_orig,
        edge_attr=edge_attr_orig,
        relabel_nodes=False)

    # Filter to only nodes that appear in edges (removes isolated nodes)
    used_nodes = sorted(final_edge_index_tmp.unique().tolist())

    # Now create final graph with only used nodes
    final_edge_index, final_edge_attr = subgraph(subset=used_nodes,
        edge_index=edge_index_orig,
        edge_attr=edge_attr_orig,
        relabel_nodes=True)

    final_subgraph = GraphData(
        x=manager.sr_parties[bank_id].procs_data[f'{mode}_data']['df'].x[used_nodes],
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
    )

    manager.sr_parties[bank_id].ctx[mode][batch_num]['graph_data'] = final_subgraph


def get_batch_intersects(manager, banks_to_sample, mode, batch_num):
    """Calculate intersections for a batch."""
    for bank_id in banks_to_sample:

        party = manager.sr_parties[bank_id]
        party.ctx[mode][batch_num]['batch_overlaps'] = np.array(party.ctx[mode][batch_num]['graph_data'].edge_attr[:,0])[np.isin(
                                                                    np.array(party.ctx[mode][batch_num]['graph_data'].edge_attr[:,0]),
                                                                    party.ctx[mode][None]['ints_indices'])]

    for bank_id in banks_to_sample:
        party = manager.sr_parties[bank_id]
        for inner_bank_id, idx in party.ctx[mode][None]['ints_indices_by_bank'].items():
            if np.any(np.isin(idx, party.ctx[mode][batch_num]['batch_overlaps'])):
                party.ctx[mode][batch_num]['intersects'][inner_bank_id] = np.where(np.isin(
                    party.ctx[mode][batch_num]['graph_data'].edge_attr[:,0], idx))[0]
                party.ctx[mode][batch_num]['ints_indices_by_bank'][inner_bank_id] = party.ctx[mode][batch_num]['graph_data'].edge_attr[party.ctx[mode][batch_num]['intersects'][inner_bank_id],0].tolist()


def gen_batch_data(manager, mode, batch_size=8192, sample_neighbors=None):
    """Generate all batch data for a mode (train or eval)."""
    if sample_neighbors is None:
        sample_neighbors = [100, 100]

    df_copy = manager.data[f'{mode}_data'][['From Bank', 'To Bank', 'Is Laundering']]

    if mode == 'train':
        df_copy = df_copy.sample(frac=1)
        parties = manager.parties.values()
        batch_iter = range(0, df_copy.shape[0], batch_size)
    elif mode == 'eval':
        parties = manager.sr_parties.values()
        batch_iter = range(min(manager.indices['eval']), max(manager.indices['eval']), batch_size)

    manager.ctx[mode]['num_batches'] = len(batch_iter)

    for batch_num, batch in enumerate(batch_iter):

        banks_to_sample = df_copy[batch:batch+batch_size][['From Bank', 'To Bank']].stack().unique()
        banks_to_sample = sorted(banks_to_sample)
        manager.ctx[mode][batch_num]['batch_labels'] = df_copy[batch:batch+batch_size]

        gen_seed_values(manager, mode, banks_to_sample, manager.ctx[mode][batch_num]['batch_labels'])

        # Initialize subgraph storage for each party
        for party in parties:
            party.subgraphs_nodes = defaultdict(list)

        for layer_num in range(1, (len(sample_neighbors) + 1)):

            gen_loaders(manager, mode, banks_to_sample, layer_num)

            # Communication between banks
            for party in parties:
                party.sample_nodes_to_add = defaultdict(list)

            banks_to_add = send_sample_nodes(manager, banks_to_sample, mode)
            banks_to_sample = receive_sample_nodes(manager, banks_to_sample, banks_to_add,
                                                   layer_num, sample_neighbors, mode)

        # Exchange final nodes (2 rounds for propagation)
        send_receive_final_nodes(manager, banks_to_sample, mode)
        send_receive_final_nodes(manager, banks_to_sample, mode)

        # Generate batch graphs
        banks_to_use = []

        for bank_id in banks_to_sample:

            gen_batch_graph(manager, manager.sr_parties[bank_id].final_subgraph_nodes, mode, batch_num=batch_num, bank_id=bank_id)
            if manager.sr_parties[bank_id].ctx[mode][batch_num]['graph_data'].edge_attr.shape[0] > 0:
                banks_to_use.append(bank_id)

        manager.ctx[mode][batch_num]['batch_parties'] = copy.deepcopy(banks_to_use)
        get_batch_intersects(manager, manager.ctx[mode][batch_num]['batch_parties'], mode, batch_num)

        get_ownership_mappings(mode, manager, manager.ctx[mode][batch_num]['batch_parties'], batch_num)
        get_nodes_to_send(mode, manager, manager.ctx[mode][batch_num]['batch_parties'], batch_num)
