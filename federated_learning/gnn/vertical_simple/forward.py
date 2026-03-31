"""Forward pass for simplified vertical FL.

Rather than exchanging embeddings at every GNN layer, each party runs
all GNN layers locally on its own graph. The manager collects the final
per-edge embeddings from both parties and concatenates them for mlp_vert.

When only one party is present for a transaction, its counterparty slot
stays as zeros (pre-allocated). The model learns that zeros = absent party.

Communication cost: O(1) rounds per epoch (only final embeddings sent),
vs O(num_gnn_layers) rounds in the standard vertical FL approach.
"""

import numpy as np
import torch


def forward_pass_simple(manager, mode, batch_num, batch_banks, batch_data):
    """Run full local GNN per party, collect embeddings, concatenate, predict.

    Args:
        manager: The FL manager instance.
        mode: 'train', 'vali', or 'test'.
        batch_num: Batch key (int, None, or LAZY_BATCH_KEY).
        batch_banks: List of bank IDs participating in this batch.
        batch_data: Dict mapping bank_id -> (party, graph_data).

    Returns:
        preds_tensor: [n_samples, n_classes] logits.
        true_y_tensor: [n_samples] ground-truth labels.
    """
    device = manager.device
    embedding_tensors = {}  # bank_id -> [num_edges, embed_dim] tensor
    index_to_position = {}  # bank_id -> {global_id: position in embedding tensor}

    # --- Step 1: each party runs the full GNN locally (no exchange) ---
    for bank_id in batch_banks:
        party, party_data = batch_data[bank_id]

        party_data.x = party_data.x.to(device)
        party_data.edge_attr = party_data.edge_attr.to(device)
        party_data.edge_index = party_data.edge_index.to(device)

        # Skip the global-ID column ([:,0]) when embedding edge features
        embeddings = party.model.gnn.emed_features(party_data.x, party_data.edge_attr[:, 1:])

        for layer_idx in range(party.model.gnn.num_gnn_layers):
            embeddings = party.model.gnn.apply_gnn_layer(
                embeddings['nodes'],
                embeddings['edges'],
                party_data.edge_index,
                layer_idx,
            )

        final_embeddings = party.model.gnn.prep_nodes_edges(
            embeddings['nodes'],
            embeddings['edges'],
            party_data.edge_index,
        )

        embedding_tensors[bank_id] = final_embeddings
        index_to_position[bank_id] = {
            int(gid): pos for pos, gid in enumerate(party_data.edge_attr[:, 0].cpu())
        }

    # --- Step 2: build per-transaction prediction tensors ---
    batch_df = manager.ctx[mode][batch_num]['batch_labels']
    from_banks = batch_df['From Bank'].values.astype(int)
    to_banks = batch_df['To Bank'].values.astype(int)
    true_y = batch_df['Is Laundering'].values
    n_samples = len(batch_df)
    indices = batch_df.index.values

    embed_dim = list(embedding_tensors.values())[0].shape[1]
    from_embeds = torch.zeros(n_samples, embed_dim, device=device)
    to_embeds = torch.zeros(n_samples, embed_dim, device=device)

    # Vectorized fill: loop over unique banks (same pattern as vertical/forward.py)
    for bank in np.unique(from_banks):
        if bank not in index_to_position:
            continue
        mask = from_banks == bank
        positions = [index_to_position[bank][idx] for idx in indices[mask]]
        from_embeds[mask] = embedding_tensors[bank][positions]

    for bank in np.unique(to_banks):
        if bank not in index_to_position:
            continue
        mask = to_banks == bank
        positions = [index_to_position[bank][idx] for idx in indices[mask]]
        to_embeds[mask] = embedding_tensors[bank][positions]

    # Absent party slots remain zero (pre-allocated above).
    # batch_labels filtering guarantees at least one party is present per transaction.
    # Intra-bank transactions (From Bank == To Bank): zero out the to_embeds slot
    # so the model sees [bank_output | zeros] rather than a duplicated embedding.
    intra_bank_mask = from_banks == to_banks
    if intra_bank_mask.any():
        to_embeds[intra_bank_mask] = 0

    # Concatenate and classify
    embeds = torch.cat([from_embeds, to_embeds], dim=1)
    preds_tensor = manager.model.gnn.mlp_vert(embeds)
    true_y_tensor = torch.tensor(true_y, dtype=torch.long, device=device)

    return preds_tensor, true_y_tensor
