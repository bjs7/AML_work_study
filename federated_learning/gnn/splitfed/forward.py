"""Forward pass for simplified vertical FL.

Rather than exchanging embeddings at every GNN layer, each party runs
all GNN layers locally on its own graph. The manager collects the final
per-edge embeddings from both parties and concatenates them for mlp_vert.

When only one party is present for a transaction, its counterparty slot
stays as zeros (pre-allocated). The model learns that zeros = absent party.

Communication cost: O(1) rounds per epoch (only final embeddings sent),
vs O(num_gnn_layers) rounds in the standard vertical FL approach.

GNN forward approaches
----------------------
_collect_embeddings_sequential: original path, one GPU kernel per party.
_collect_embeddings_batched:    merges all party subgraphs into one PyG
    Batch (disconnected union — no cross-party message passing) and runs
    a single GNN kernel for all parties at once.

BatchNorm note: in train mode BatchNorm computes per-batch statistics.
Sequential mode normalises per-party; batched mode normalises across all
parties combined. In eval mode both are identical (running stats used).
The vertical forward pass is inference, so the model should be in eval
mode during these calls; the test enforces this explicitly.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch


def _collect_embeddings_sequential(batch_banks, batch_data, device, dp_clip, dp_noise_scale):
    """Run GNN forward pass per party sequentially — one GPU call per party.

    Returns:
        embedding_tensors: {bank_id: [num_edges, embed_dim]}
        index_to_position: {bank_id: {global_id: edge_position}}
    """
    embedding_tensors = {}
    index_to_position = {}

    for bank_id in batch_banks:
        party, party_data = batch_data[bank_id]

        party_data.x = party_data.x.to(device)
        party_data.edge_attr = party_data.edge_attr.to(device)
        party_data.edge_index = party_data.edge_index.to(device)

        embeddings = party.model.gnn.emed_features(party_data.x, party_data.edge_attr[:, 1:])
        for layer_idx in range(party.model.gnn.num_gnn_layers):
            embeddings = party.model.gnn.apply_gnn_layer(
                embeddings['nodes'], embeddings['edges'], party_data.edge_index, layer_idx
            )
        final_embeddings = party.model.gnn.prep_nodes_edges(
            embeddings['nodes'], embeddings['edges'], party_data.edge_index
        )

        if dp_clip is not None:
            norms = final_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            final_embeddings = final_embeddings * (dp_clip / norms.clamp(min=dp_clip))
        if dp_noise_scale > 0.0:
            final_embeddings = final_embeddings + torch.randn_like(final_embeddings) * dp_noise_scale

        embedding_tensors[bank_id] = final_embeddings
        index_to_position[bank_id] = {
            int(gid): pos for pos, gid in enumerate(party_data.edge_attr[:, 0].cpu())
        }

    return embedding_tensors, index_to_position


def _collect_embeddings_batched(batch_banks, batch_data, device, dp_clip, dp_noise_scale):
    """Run all party GNN forward passes in one batched GPU call.

    Merges per-party subgraphs into a disconnected PyG Batch so that message
    passing stays within each party's subgraph (no cross-party information
    leak). One GNN call replaces N sequential calls.

    DP clipping is applied to the combined tensor before splitting
    (deterministic, equivalent to per-party clipping). DP noise is applied
    per-party after splitting to preserve independent noise per party.

    All parties must share the same GNN weights (true after FedAvg sync).
    The first party's model is used for the forward pass.

    Returns:
        embedding_tensors: {bank_id: [num_edges, embed_dim]}
        index_to_position: {bank_id: {global_id: edge_position}}
    """
    embedding_tensors = {}
    index_to_position = {}
    gnn_inputs = []
    edge_counts = []

    for bank_id in batch_banks:
        _, party_data = batch_data[bank_id]

        index_to_position[bank_id] = {
            int(gid): pos for pos, gid in enumerate(party_data.edge_attr[:, 0].cpu())
        }

        # Strip global-ID col 0 — GNN only sees real edge features
        gnn_inputs.append(Data(
            x=party_data.x,
            edge_index=party_data.edge_index,
            edge_attr=party_data.edge_attr[:, 1:],
        ))
        edge_counts.append(party_data.num_edges)

    # Assemble on CPU then transfer once — cheaper than N individual transfers
    combined = Batch.from_data_list(gnn_inputs).to(device)

    # All parties share synced GNN weights; use first party's model
    gnn = batch_data[batch_banks[0]][0].model.gnn
    embeddings = gnn.emed_features(combined.x, combined.edge_attr)
    for layer_idx in range(gnn.num_gnn_layers):
        embeddings = gnn.apply_gnn_layer(
            embeddings['nodes'], embeddings['edges'], combined.edge_index, layer_idx
        )
    final_embeddings = gnn.prep_nodes_edges(
        embeddings['nodes'], embeddings['edges'], combined.edge_index
    )

    if dp_clip is not None:
        norms = final_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        final_embeddings = final_embeddings * (dp_clip / norms.clamp(min=dp_clip))

    # Split back per party — Batch preserves insertion order
    per_party = torch.split(final_embeddings, edge_counts)

    for idx, bank_id in enumerate(batch_banks):
        emb = per_party[idx]
        if dp_noise_scale > 0.0:
            emb = emb + torch.randn_like(emb) * dp_noise_scale
        embedding_tensors[bank_id] = emb

    return embedding_tensors, index_to_position


def forward_pass_splitfed(manager, mode, batch_num, batch_banks, batch_data, use_batched=True):
    """Run full local GNN per party, collect embeddings, concatenate, predict.

    Args:
        manager: The FL manager instance.
        mode: 'train', 'vali', or 'test'.
        batch_num: Batch key (int, None, or LAZY_BATCH_KEY).
        batch_banks: List of bank IDs participating in this batch.
        batch_data: Dict mapping bank_id -> (party, graph_data).
        use_batched: If True (default), use the batched GPU forward pass.
            Set False to fall back to the sequential approach (for testing).

    Returns:
        preds_tensor: [n_samples, n_classes] logits.
        true_y_tensor: [n_samples] ground-truth labels.
    """
    device = manager.device
    dp_clip = getattr(manager.args['fl_parser'], 'dp_clip', None)
    dp_noise_scale = getattr(manager.args['fl_parser'], 'dp_noise_scale', 0.0)

    if use_batched:
        embedding_tensors, index_to_position = _collect_embeddings_batched(
            batch_banks, batch_data, device, dp_clip, dp_noise_scale
        )
    else:
        embedding_tensors, index_to_position = _collect_embeddings_sequential(
            batch_banks, batch_data, device, dp_clip, dp_noise_scale
        )

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
