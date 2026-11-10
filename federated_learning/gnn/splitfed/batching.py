"""Lazy batch processing for simplified vertical FL.

process_lazy_batch_splitfed is identical to process_lazy_batch from
vertical/batching.py, but omits the three exchange-data setup calls
(get_batch_intersects, get_ownership_mappings, get_nodes_to_send)
that are only needed by the per-layer embedding exchange forward pass.
"""

import torch
from data.data_utils import GraphData
from federated_learning.gnn.fedgraph.batching import LAZY_BATCH_KEY


def build_edge_lookup(mode_parties, mode):
    """(global edge IDs, bank, local edge position) for every edge of every party graph
    of a mode, sorted by global ID. Built once at setup so each batch finds its party
    edges by lookup instead of scanning every party's full graph."""
    gids, banks, positions = [], [], []
    for bank_id, party in mode_parties.items():
        g = party.procs_data[f'{mode}_data']['df'].edge_attr[:, 0].long()
        gids.append(g)
        banks.append(torch.full_like(g, int(bank_id)))
        positions.append(torch.arange(len(g), dtype=torch.long))
    gids = torch.cat(gids)
    order = torch.argsort(gids, stable=True)
    return gids[order], torch.cat(banks)[order], torch.cat(positions)[order]


def _party_edge_positions(edge_lookup, batch_global_ids, banks):
    """{bank: ascending local edge positions} of the batch edges in each bank's graph."""
    sorted_gids, entry_banks, entry_pos = edge_lookup
    left = torch.searchsorted(sorted_gids, batch_global_ids, right=False)
    right = torch.searchsorted(sorted_gids, batch_global_ids, right=True)
    counts = right - left
    total = int(counts.sum())
    if total == 0:
        return {}
    # Expand each batch edge's [left, right) range of lookup entries (an edge sits in
    # one party graph, or two when sender and receiver banks differ).
    starts = torch.repeat_interleave(left, counts)
    offsets = torch.arange(total) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    entries = starts + offsets

    sel_banks, sel_pos = entry_banks[entries], entry_pos[entries]
    keep = torch.isin(sel_banks, torch.tensor(sorted(banks), dtype=torch.long))
    sel_banks, sel_pos = sel_banks[keep], sel_pos[keep]
    if len(sel_banks) == 0:
        return {}

    # Group by bank with positions ascending, i.e. the party graph's own edge order.
    key = sel_banks * (int(sel_pos.max()) + 1) + sel_pos
    order = torch.argsort(key)
    sel_banks, sel_pos = sel_banks[order], sel_pos[order]
    uniq_banks, bank_counts = torch.unique_consecutive(sel_banks, return_counts=True)
    return dict(zip(uniq_banks.tolist(), torch.split(sel_pos, bank_counts.tolist())))


def process_lazy_batch_splitfed(manager, mode, batch, mode_parties):
    """Process one loader batch — subgraph extraction only, no exchange data.

    Called per batch per epoch in the training loop. Overwrites the previous
    batch's data in manager.ctx[mode][LAZY_BATCH_KEY], keeping only one batch
    worth of subgraphs in memory at a time.

    Args:
        manager: The FL manager instance.
        mode: 'train', 'vali', or 'test'.
        batch: A batch object from the LinkNeighborLoader.
        mode_parties: Dict of {bank_id: party} for this mode.
    """
    seed_global_ids = batch.edge_label.long().numpy()
    bl = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels'] = bl[
        bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]

    # Include seed IDs in the lookup set so that seed edges dropped by neighbor
    # sampling are still matched in the party's graph (same logic as batching_masker).
    seed_ids_tensor = torch.tensor(seed_global_ids, dtype=torch.long)
    batch_global_ids = torch.cat([batch.edge_attr[:, 0].long(), seed_ids_tensor]).unique()

    batch_df = manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels']
    seed_banks = set(batch_df['From Bank'].values) | set(batch_df['To Bank'].values)
    seed_mode_parties = {k: v for k, v in mode_parties.items() if k in seed_banks}

    party_positions = _party_edge_positions(
        manager.edge_lookup[mode], batch_global_ids, seed_mode_parties.keys())

    banks_to_use = []
    for bank_id, party in seed_mode_parties.items():
        positions = party_positions.get(int(bank_id))
        if positions is None:
            continue
        party_graph = party.procs_data[f'{mode}_data']['df']
        matched_edge_index = party_graph.edge_index[:, positions]
        sub_nodes = matched_edge_index.reshape(-1).unique()   # sorted, so searchsorted relabels
        party.ctx[mode][LAZY_BATCH_KEY]['graph_data'] = GraphData(
            x=party_graph.x[sub_nodes],
            edge_index=torch.searchsorted(sub_nodes, matched_edge_index),
            edge_attr=party_graph.edge_attr[positions],
        )
        banks_to_use.append(bank_id)
    manager.ctx[mode][LAZY_BATCH_KEY]['batch_parties'] = banks_to_use
    # No get_batch_intersects, get_ownership_mappings, or get_nodes_to_send —
    # the simple forward pass does not need cross-party exchange metadata.
