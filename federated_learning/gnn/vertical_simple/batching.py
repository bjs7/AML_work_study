"""Lazy batch processing for simplified vertical FL.

process_lazy_batch_simple is identical to process_lazy_batch from
vertical/batching.py, but omits the three exchange-data setup calls
(get_batch_intersects, get_ownership_mappings, get_nodes_to_send)
that are only needed by the per-layer embedding exchange forward pass.
"""

import torch
import numpy as np

from data.data_utils import GraphData
from training.parallel import parallel_party_execute
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY


def process_lazy_batch_simple(manager, mode, batch, mode_parties):
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
    max_workers = getattr(manager.args['fl_parser'], 'max_workers', None)

    seed_global_ids = batch.edge_label.long().numpy()
    bl = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels'] = bl[
        bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]

    batch_global_ids = batch.edge_attr[:, 0].long()

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

    # Refilter batch_labels to only transactions covered by at least one party's
    # actual subgraph. The initial filter (mode_party_set) can include transactions
    # that a bank is associated with in df_labels but doesn't have in its local
    # party graph for this specific batch neighborhood.
    covered_ids = set()
    for bank_id in banks_to_use:
        party = mode_parties[bank_id]
        subgraph = party.ctx[mode][LAZY_BATCH_KEY]['graph_data']
        covered_ids.update(subgraph.edge_attr[:, 0].cpu().long().tolist())
    bl = manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels']
    manager.ctx[mode][LAZY_BATCH_KEY]['batch_labels'] = bl[bl.index.isin(covered_ids)]
    # No get_batch_intersects, get_ownership_mappings, or get_nodes_to_send —
    # the simple forward pass does not need cross-party exchange metadata.
