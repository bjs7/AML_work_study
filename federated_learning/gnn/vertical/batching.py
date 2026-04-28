"""Batch generation functions for vertical federated learning."""

import copy
import numpy as np
import torch
import warnings
from collections import defaultdict
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import subgraph
from data.data_utils import GraphData

from .ownership import get_ownership_mappings, get_nodes_to_send
from federated_learning.parallel import parallel_party_execute

# Silence pyg-lib deprecation warning (show only once)
#warnings.filterwarnings('once', message=".*NeighborSampler.*pyg-lib.*")





def gen_seed_values(manager, mode, banks_to_sample, df):
    """Generate seed values for neighbor sampling."""
    mode_parties = manager.get_parties_for_mode(mode)
    for bank_id in banks_to_sample:
        party = mode_parties[bank_id]

        mask = (df['From Bank'] == bank_id) | (df['To Bank'] == bank_id)
        party.indexes = np.array(df[mask].index)

        reversed_dict = {v: k for k, v in party.procs_data[f'{mode}_data']['indices_mapping'].items()}

        party.seed_indices = [reversed_dict[idx] for idx in party.indexes]
        party.seed_nodes = party.procs_data[f'{mode}_data']['df'].edge_index[:,party.seed_indices].unique().tolist()
        party.sample_nodes = party.seed_nodes


def gen_loaders(manager, mode, banks_to_sample, layer_num):
    """Generate neighbor loaders for each bank."""
    shuffle_arg = True if mode == 'train' else False
    mode_parties = manager.get_parties_for_mode(mode)

    for bank_id in banks_to_sample:
        party = mode_parties[bank_id]

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
    mode_parties = manager.get_parties_for_mode(mode)
    banks_to_add = []
    for bank_id in banks_to_sample:

        party = mode_parties[bank_id]
        for inner_bank_id, nodes_ids in party.ctx[mode][None]['nodes_idx'].items():
            if len(np.array(party.matching_values)[np.isin(party.matching_values, nodes_ids)]) > 0:
                local_nodes_to_send = np.array(party.matching_values)[np.isin(party.matching_values, nodes_ids)]
                nodes_to_send = [party.ctx[mode][None]['local_to_global'][node] for node in local_nodes_to_send]
                mode_parties[inner_bank_id].sample_nodes_to_add[party.bank_id] = nodes_to_send
                if inner_bank_id not in banks_to_sample and inner_bank_id not in banks_to_add:
                    banks_to_add.append(inner_bank_id)

    return banks_to_add


def receive_sample_nodes(manager, banks_to_sample, banks_to_add, layer_num, sample_neighbors, mode):
    """Receive sampled nodes from other parties."""
    mode_parties = manager.get_parties_for_mode(mode)
    for bank_id in (banks_to_sample + banks_to_add):

        party = mode_parties[bank_id]
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
    mode_parties = manager.get_parties_for_mode(mode)
    banks_to_add = []
    for bank_id in banks_to_sample:

        party = mode_parties[bank_id]
        for inner_bank_id, nodes_ids in party.ctx[mode][None]['nodes_idx'].items():
            if len(np.array(party.final_subgraph_nodes)[np.isin(party.final_subgraph_nodes, nodes_ids)]) > 0:
                local_nodes_to_send = np.array(party.final_subgraph_nodes)[np.isin(party.final_subgraph_nodes, nodes_ids)]
                nodes_to_send = [party.ctx[mode][None]['local_to_global'][node] for node in local_nodes_to_send]
                mode_parties[inner_bank_id].sample_nodes_to_add[party.bank_id] = nodes_to_send
                if inner_bank_id not in banks_to_sample and inner_bank_id not in banks_to_add:
                    banks_to_add.append(inner_bank_id)

    for bank_id in banks_to_sample:
        party = mode_parties[bank_id]
        for inner_bank_id, nodes in party.sample_nodes_to_add.items():
            mapped_nodes = [party.ctx[mode][None]['global_to_local'][node] for node in nodes]
            party.final_subgraph_nodes += mapped_nodes
        party.final_subgraph_nodes = list(set(party.final_subgraph_nodes))


def gen_batch_graph(manager, all_nodes, mode, batch_num=0, bank_id=0):
    """Generate batch subgraph for a party, filtering out isolated nodes."""
    mode_parties = manager.get_parties_for_mode(mode)
    edge_index_orig = mode_parties[bank_id].procs_data[f'{mode}_data']['df'].edge_index
    edge_attr_orig = mode_parties[bank_id].procs_data[f'{mode}_data']['df'].edge_attr

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
        x=mode_parties[bank_id].procs_data[f'{mode}_data']['df'].x[used_nodes],
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
    )

    mode_parties[bank_id].ctx[mode][batch_num]['graph_data'] = final_subgraph


def get_batch_intersects(manager, banks_to_sample, mode, batch_num, max_workers=None):
    """Calculate intersections for a batch."""
    mode_parties = manager.get_parties_for_mode(mode)
    batch_parties = {bid: mode_parties[bid] for bid in banks_to_sample}

    def _compute_overlaps(bank_id, party):
        edge_ids = np.array(party.ctx[mode][batch_num]['graph_data'].edge_attr[:, 0])
        party.ctx[mode][batch_num]['batch_overlaps'] = edge_ids[
            np.isin(edge_ids, party.ctx[mode][None]['ints_indices'])]

    parallel_party_execute(batch_parties, _compute_overlaps, max_workers=max_workers)

    def _compute_intersects(bank_id, party):
        # Clear stale entries from previous batch before recomputing
        party.ctx[mode][batch_num]['intersects'] = {}
        party.ctx[mode][batch_num]['ints_indices_by_bank'] = {}
        for inner_bank_id, idx in party.ctx[mode][None]['ints_indices_by_bank'].items():
            if np.any(np.isin(idx, party.ctx[mode][batch_num]['batch_overlaps'])):
                party.ctx[mode][batch_num]['intersects'][inner_bank_id] = np.where(np.isin(
                    party.ctx[mode][batch_num]['graph_data'].edge_attr[:, 0], idx))[0]
                party.ctx[mode][batch_num]['ints_indices_by_bank'][inner_bank_id] = party.ctx[mode][batch_num]['graph_data'].edge_attr[
                    party.ctx[mode][batch_num]['intersects'][inner_bank_id], 0].tolist()

    parallel_party_execute(batch_parties, _compute_intersects, max_workers=max_workers)


def gen_batch_data(manager, mode, batch_size=8192, sample_neighbors=None):

    print(f'testing generating batching for {mode}')

    """Generate all batch data for a mode (train, vali, or test)."""
    if sample_neighbors is None:
        sample_neighbors = [100, 100]

    df_copy = manager.data[f'{mode}_data'][['From Bank', 'To Bank', 'Is Laundering']]
    mode_parties = manager.get_parties_for_mode(mode)

    if mode == 'train':
        df_copy = df_copy.sample(frac=1)
        batch_iter = range(0, df_copy.shape[0], batch_size)
    else:  # 'vali' or 'test'
        batch_iter = range(min(manager.indices[mode]), max(manager.indices[mode]), batch_size)

    manager.ctx[mode]['num_batches'] = len(batch_iter)

    for batch_num, batch in enumerate(batch_iter):

        banks_to_sample = df_copy[batch:batch+batch_size][['From Bank', 'To Bank']].stack().unique()
        banks_to_sample = sorted(banks_to_sample)
        manager.ctx[mode][batch_num]['batch_labels'] = df_copy[batch:batch+batch_size]

        gen_seed_values(manager, mode, banks_to_sample, manager.ctx[mode][batch_num]['batch_labels'])

        # Initialize subgraph storage for each party
        for party in mode_parties.values():
            party.subgraphs_nodes = defaultdict(list)

        for layer_num in range(1, (len(sample_neighbors) + 1)):

            gen_loaders(manager, mode, banks_to_sample, layer_num)

            # Communication between banks
            for party in mode_parties.values():
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

            gen_batch_graph(manager, mode_parties[bank_id].final_subgraph_nodes, mode, batch_num=batch_num, bank_id=bank_id)
            if mode_parties[bank_id].ctx[mode][batch_num]['graph_data'].edge_attr.shape[0] > 0:
                banks_to_use.append(bank_id)

        manager.ctx[mode][batch_num]['batch_parties'] = copy.deepcopy(banks_to_use)
        get_batch_intersects(manager, manager.ctx[mode][batch_num]['batch_parties'], mode, batch_num)

        get_ownership_mappings(mode, manager, manager.ctx[mode][batch_num]['batch_parties'], batch_num)
        get_nodes_to_send(mode, manager, manager.ctx[mode][batch_num]['batch_parties'], batch_num)


def gen_batch_data_simple(manager, mode, batch_size=8192):
    """Alternative batch generation: manager-driven index batching.

    The manager generates batches from global transaction indices. Each party
    extracts its local subgraph by matching global IDs in edge_attr[:,0],
    without iterative neighbor sampling between parties.

    Can be called per-epoch during training for fresh random batches.
    """
    mode_parties = manager.get_parties_for_mode(mode)
    max_workers = getattr(manager.args['fl_parser'], 'max_workers', None)

    if mode == 'train':
        indices = manager.data['train_data'].index.to_numpy().copy()
        np.random.shuffle(indices)
    else:
        indices = manager.indices[mode].to_numpy()

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
            sub_nodes = party_graph.edge_index[:, mask].unique()
            final_edge_index, final_edge_attr = subgraph(
                subset=sub_nodes,
                edge_index=party_graph.edge_index,
                edge_attr=party_graph.edge_attr,
                relabel_nodes=True,
            )
            party.ctx[mode][batch_num]['graph_data'] = GraphData(
                x=party_graph.x[sub_nodes],
                edge_index=final_edge_index,
                edge_attr=final_edge_attr,
            )
            return bank_id

        results = parallel_party_execute(mode_parties, _build_subgraph, max_workers=max_workers)
        banks_to_use = [bid for bid, result in results.items() if result is not None]

        manager.ctx[mode][batch_num]['batch_parties'] = banks_to_use
        get_batch_intersects(manager, banks_to_use, mode, batch_num, max_workers=max_workers)
        get_ownership_mappings(mode, manager, banks_to_use, batch_num, max_workers=max_workers)
        get_nodes_to_send(mode, manager, banks_to_use, batch_num)


def gen_batch_data_link_neighbor(manager, mode, batch_size=8192, sample_neighbors=None):
    """Batch generation using LinkNeighborLoader on the manager's full transaction graph.

    Builds a minimal reference graph from the manager's data (all transactions for
    the mode, with global IDs as edge_attr[:,0]) and uses LinkNeighborLoader to
    generate batches with k-hop neighborhood context. Each party then extracts only
    their edges matching the batch's global IDs, with manual node relabeling — no
    subgraph() call, so only the matched edges are included (not all edges of those nodes).

    batch_labels: seed transactions only (the batch_size edges LinkNeighborLoader
    was asked to predict on).
    Party subgraphs: edges matching seed + neighbor global IDs from the reference.

    Note: the loader reshuffles automatically each time it is iterated (shuffle=True
    for train). Calling this function per-epoch is supported but incurs the full cost
    of get_batch_intersects/get_ownership_mappings/get_nodes_to_send per batch.
    """
    if sample_neighbors is None:
        sample_neighbors = [100, 100]

    mode_parties = manager.get_parties_for_mode(mode)
    df = manager.data[f'{mode}_data']
    df_labels = df[['From Bank', 'To Bank', 'Is Laundering']]

    # Build a minimal manager-level graph for sampling (no party-specific features needed)
    from_ids = torch.tensor(df['from_id'].values, dtype=torch.long)
    to_ids = torch.tensor(df['to_id'].values, dtype=torch.long)
    edge_index = torch.stack([from_ids, to_ids], dim=0)
    global_ids = torch.tensor(df.index.values, dtype=torch.float)
    ref_graph = GraphData(
        x=torch.zeros(edge_index.max().item() + 1, 1),  # dummy node features
        edge_index=edge_index,
        edge_attr=global_ids.unsqueeze(1),              # edge_attr[:,0] = global IDs
    )

    # Pass global transaction IDs as edge_label so each batch reports its seed IDs
    loader = LinkNeighborLoader(
        ref_graph,
        num_neighbors=sample_neighbors,
        edge_label_index=ref_graph.edge_index,
        edge_label=ref_graph.edge_attr[:, 0].float(),
        batch_size=batch_size,
        shuffle=(mode == 'train'),
    )

    manager.ctx[mode]['num_batches'] = len(loader)
    df_index_set = set(df_labels.index)

    for batch_num, batch in enumerate(loader):
        print(batch_num)
        # Seed global IDs → batch_labels (what we predict on)
        seed_global_ids = batch.edge_label.long().numpy()
        valid_seed_ids = seed_global_ids[np.isin(seed_global_ids, list(df_index_set))]
        manager.ctx[mode][batch_num]['batch_labels'] = df_labels.loc[valid_seed_ids]

        # All global IDs in the batch (seed + neighbors) → party matching
        batch_global_ids = batch.edge_attr[:, 0].long()

        banks_to_use = []
        for bank_id, party in mode_parties.items():
            party_graph = party.procs_data[f'{mode}_data']['df']
            mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)

            if mask.sum() == 0:
                continue

            # Manual node relabeling: only keep matched edges, no subgraph() expansion
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
            banks_to_use.append(bank_id)

        manager.ctx[mode][batch_num]['batch_parties'] = banks_to_use
        get_batch_intersects(manager, banks_to_use, mode, batch_num)
        get_ownership_mappings(mode, manager, banks_to_use, batch_num)
        get_nodes_to_send(mode, manager, banks_to_use, batch_num)




def setup_lazy_batch_loader(manager, mode, batch_size=8192, sample_neighbors=None):
    """Set up the lazy LinkNeighborLoader for a mode. Called once at setup time.

    Builds a minimal manager-level graph and creates a LinkNeighborLoader.
    Stores the loader on manager.loaders[mode] so the training loop can iterate
    it each epoch (getting fresh neighbor samples and shuffled batches each time).
    Does NOT pre-compute any batch subgraphs — those are computed on-the-fly
    per batch via process_lazy_batch().
    """
    if sample_neighbors is None:
        sample_neighbors = [100, 100]

    if not hasattr(manager, 'loaders'):
        manager.loaders = {}

    df = manager.data[f'{mode}_data']
    df_labels = df[['From Bank', 'To Bank', 'Is Laundering']]

    from_ids = torch.tensor(df['from_id'].values, dtype=torch.long)
    to_ids = torch.tensor(df['to_id'].values, dtype=torch.long)
    edge_index = torch.stack([from_ids, to_ids], dim=0)
    global_ids = torch.tensor(df.index.values, dtype=torch.float)
    ref_graph = GraphData(
        x=torch.zeros(edge_index.max().item() + 1, 1),
        edge_index=edge_index,
        edge_attr=global_ids.unsqueeze(1),
    )

    if mode == 'train':
        label_index = list(range(0, manager.data['train_data'].shape[0]))
    elif mode == 'vali':
        label_index = list(range(manager.data['train_data'].shape[0], manager.data['vali_data'].shape[0]))
    else:
        label_index = list(range(manager.data['vali_data'].shape[0], manager.data['test_data'].shape[0]))

    loader = LinkNeighborLoader(
        ref_graph,
        num_neighbors=sample_neighbors,
        edge_label_index=ref_graph.edge_index[:, label_index],
        edge_label=ref_graph.edge_attr[label_index, 0].float(),  # global IDs for seed edges only
        batch_size=batch_size,
        shuffle=(mode == 'train'),
    )

    manager.loaders[mode] = loader
    manager.ctx[mode]['num_batches'] = len(loader)
    manager.ctx[mode]['df_labels'] = df_labels


LAZY_BATCH_KEY = 'current_batch'
def process_lazy_batch(manager, mode, batch, mode_parties):
    """Process one loader batch into manager.ctx[mode][LAZY_BATCH_KEY].

    Called per batch per epoch in the training loop. Overwrites the previous
    batch's data, so only one batch worth of subgraphs is in memory at a time.
    """
    max_workers = getattr(manager.args['fl_parser'], 'max_workers', None)

    seed_global_ids = batch.edge_label.long().numpy()
    bl = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels'] = bl[
        bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]

    # Include seed IDs so that seed edges dropped by neighbor sampling are still
    # matched in the party's graph (same logic as batching_masker).
    seed_ids_tensor = torch.tensor(seed_global_ids, dtype=torch.long)
    batch_global_ids = torch.cat([batch.edge_attr[:, 0].long(), seed_ids_tensor]).unique()

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
        party.ctx[mode][LAZY_BATCH_KEY]['graph_data'] = GraphData(
            x=party_graph.x[sub_nodes],
            edge_index=node_remap[matched_edge_index],
            edge_attr=matched_edge_attr,
        )
        return bank_id

    results = parallel_party_execute(mode_parties, _build_subgraph, max_workers=max_workers)
    banks_to_use = [bid for bid, result in results.items() if result is not None]

    manager.ctx[mode][LAZY_BATCH_KEY]['batch_parties'] = banks_to_use
    get_batch_intersects(manager, banks_to_use, mode, LAZY_BATCH_KEY, max_workers=max_workers)
    get_ownership_mappings(mode, manager, banks_to_use, LAZY_BATCH_KEY, max_workers=max_workers)
    get_nodes_to_send(mode, manager, banks_to_use, LAZY_BATCH_KEY)

