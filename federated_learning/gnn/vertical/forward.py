"""Forward pass functions for vertical federated learning.

These functions handle the multi-party forward pass with embedding exchange.
"""

import numpy as np
import torch


def get_batch_data(manager, mode, batch_num=None, batch_banks=None):
    """Get batch data for a given mode and batch.

    Args:
        manager: The FL manager instance
        mode: 'train', 'vali', or 'test'
        batch_num: Batch number (None for non-batching mode)
        batch_banks: List of bank IDs in the batch

    Returns:
        Dict mapping bank_id to (party, graph_data) tuple
    """
    mode_parties = manager.get_parties_for_mode(mode)
    if batch_num is None:
        return {bank_id: (party, party.procs_data[f'{mode}_data']['df']) for bank_id, party in mode_parties.items()}
    else:
        return {bank_id: (mode_parties[bank_id], mode_parties[bank_id].ctx[mode][batch_num]['graph_data']) for bank_id in batch_banks}


def forward_pass(manager, mode, batch_num, batch_banks, batch_data):
    """Execute forward pass with embedding exchange between parties.

    This is the core vertical FL forward pass that:
    1. Computes initial embeddings at each party
    2. Applies GNN layers with embedding exchange between parties
    3. Collects embeddings and makes predictions

    Args:
        manager: The FL manager instance
        mode: 'train', 'vali', or 'test'
        batch_num: Batch number (None for non-batching mode)
        batch_banks: List of bank IDs participating in this batch
        batch_data: Dict mapping bank_id to (party, graph_data)

    Returns:
        preds_tensor: Predictions tensor
        true_y_tensor: Ground truth labels tensor
    """
    manager.embeddings_indices = {}
    device = manager.device

    # Get the embeddings of nodes and edges (exclude ID column with [:,1:])
    for bank_id in batch_banks:
        party, party_data = batch_data[bank_id]

        # Move graph data to device
        party_data.x = party_data.x.to(device)
        party_data.edge_attr = party_data.edge_attr.to(device)
        party_data.edge_index = party_data.edge_index.to(device)

        party.received_embeddings = {}
        party.current_embeddings = party.model.gnn.emed_features(party_data.x, party_data.edge_attr[:,1:])

    # Apply GNN layers with embedding exchange
    for layer_idx in range(party.model.gnn.num_gnn_layers):

        for bank_id in batch_banks:
            party, party_data = batch_data[bank_id]

            party.current_embeddings = party.model.gnn.apply_gnn_layer(
                party.current_embeddings['nodes'],
                party.current_embeddings['edges'],
                party_data.edge_index,
                layer_idx
            )

        # Send embeddings to other parties
        for bank_id in batch_banks:
            party, party_data = batch_data[bank_id]

            for inner_bank_id, nodes_indices in party.ctx[mode][batch_num]['nodes_to_send'].items():

                # Send embeddings
                node_embeds = party.current_embeddings['nodes'][nodes_indices['local_indices']]
                nodes_dict = dict(zip(nodes_indices['global_ids'], node_embeds))

                idx = party_data.edge_attr[party.ctx[mode][batch_num]['intersects'][inner_bank_id],0].tolist()
                owned_edges = party.ctx[mode][batch_num]['intersects'][inner_bank_id][manager.data[f'{mode}_data'].iloc[idx, :]['From Bank'] == bank_id]

                manager.get_parties_for_mode(mode)[inner_bank_id].received_embeddings[party.bank_id] = {
                    'nodes': nodes_dict,
                    'edges': party.current_embeddings['edges'][owned_edges]
                }

        # Receive and merge embeddings
        for bank_id in batch_banks:
            party, party_data = batch_data[bank_id]

            if len(party.received_embeddings) == 0:
                continue

            nodes_clone = party.current_embeddings['nodes'].clone()
            edges_clone = party.current_embeddings['edges'].clone()

            for inner_bank_id, received in party.received_embeddings.items():
                idx = party_data.edge_attr[party.ctx[mode][batch_num]['intersects'][inner_bank_id],0].tolist()
                not_owned_edges = party.ctx[mode][batch_num]['intersects'][inner_bank_id][manager.data[f'{mode}_data'].iloc[idx, :]['From Bank'] == inner_bank_id]

                for global_id, embedding in received['nodes'].items():
                    if global_id not in party.ctx[mode][batch_num]['global_to_local']:
                        continue
                    local_idx = party.ctx[mode][batch_num]['global_to_local'][global_id]
                    own_acc = party.ctx[mode][batch_num]['owned_accounts'] if isinstance(party.ctx[mode][batch_num]['owned_accounts'], set) else party.ctx[mode][batch_num]['owned_accounts'][0]

                    if global_id not in own_acc:
                        nodes_clone[local_idx] = embedding

                if len(received['edges']) == len(not_owned_edges):
                    edges_clone[not_owned_edges] = received['edges']

            party.current_embeddings['nodes'] = nodes_clone
            party.current_embeddings['edges'] = edges_clone

    # Final preparation of embeddings
    for bank_id in batch_banks:
        party, party_data = batch_data[bank_id]
        party.current_embeddings = party.model.gnn.prep_nodes_edges(
            party.current_embeddings['nodes'],
            party.current_embeddings['edges'],
            party_data.edge_index
        )

    # Collect embeddings for prediction
    embeddings_tensor = {}
    index_to_position = {}

    for bank_id in batch_banks:
        embeddings_tensor[bank_id] = manager.get_parties_for_mode(mode)[bank_id].current_embeddings

        party, party_data = batch_data[bank_id]
        index_to_position[bank_id] = {
            int(idx): pos for pos, idx in enumerate(party_data.edge_attr[:,0])
        }

    # Build prediction tensors
    batch_df = manager.ctx[mode][batch_num]['batch_labels']
    from_banks = batch_df['From Bank'].values.astype(int)
    to_banks = batch_df['To Bank'].values.astype(int)
    true_y = batch_df['Is Laundering'].values

    embed_dim = list(embeddings_tensor.values())[0].shape[1]
    device = list(embeddings_tensor.values())[0].device

    n_samples = manager.ctx[mode][batch_num]['batch_labels'].shape[0]
    indices = batch_df.index.values

    # After per-layer exchange both parties hold identical embeddings for shared
    # transactions, so one party's prep_nodes_edges output is sufficient.
    # Use From Bank as primary; fall back to To Bank when From Bank is absent.
    single_embeds = torch.zeros(n_samples, embed_dim, device=device)

    for bank in np.unique(from_banks):
        if bank not in index_to_position:
            continue
        mask = from_banks == bank
        bank_indices = indices[mask]
        positions = [index_to_position[bank][idx] for idx in bank_indices]
        single_embeds[mask] = embeddings_tensor[bank][positions]

    non_party_from_mask = np.isin(from_banks, list(index_to_position.keys()), invert=True)
    if non_party_from_mask.any():
        for bank in np.unique(to_banks[non_party_from_mask]):
            if bank not in index_to_position:
                continue
            mask = non_party_from_mask & (to_banks == bank)
            bank_indices = indices[mask]
            positions = [index_to_position[bank][idx] for idx in bank_indices]
            single_embeds[mask] = embeddings_tensor[bank][positions]

    preds_tensor = manager.model.gnn.mlp(single_embeds)
    true_y_tensor = torch.tensor(true_y, dtype=torch.long, device=device)

    return preds_tensor, true_y_tensor
