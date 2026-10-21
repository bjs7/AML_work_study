"""Forward pass for RestrictedMerge FL.

For each (From Bank, To Bank) pair in a batch:
1. Get both parties' lazy batch subgraphs (same restriction as SplitFed)
2. For seed transactions shared by both parties, reconcile the sender and
   receiver account nodes so they become ONE node in the merged graph.
3. Merge into a single graph that supports cross-party message passing.
4. Run one GNN over all merged pair-graphs in a single batched GPU call.
5. Classify using the standard single-view mlp head (not mlp_vert).

Experiment motivation
---------------------
SplitFed: 2 independent GNNs (one per party) + mlp_vert (2× embed concatenation)
RestrictedMerge: 1 GNN (per merged pair graph) + mlp (standard embed)

Both see the same restricted subgraph information. Comparing them isolates
whether performance differences from full_info stem from restricted information
(hypothesis: yes) or from the two-view concatenation architecture.
"""

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data, Batch

# Number of merged pair-graphs to process per GPU call.
# Smaller = less peak memory; larger = faster (more GPU parallelism).
# 16 is safe for A100-80GB with lazy_link_neighbor subgraphs.
PAIR_CHUNK_SIZE = 16


def _seed_embeddings(gnn, x, edge_attr, edge_index, positions):
    """Run the GNN on one chunk and return only the seed-edge embeddings."""
    emb = gnn.emed_features(x, edge_attr)
    for layer_idx in range(gnn.num_gnn_layers):
        emb = gnn.apply_gnn_layer(emb['nodes'], emb['edges'], edge_index, layer_idx)
    final = gnn.prep_nodes_edges(emb['nodes'], emb['edges'], edge_index)
    return final[positions]


def _build_merged_graph(gdata_a, gdata_b, seed_gids_set):
    """Merge two party subgraphs via seed-node reconciliation.

    For seed transactions shared by both parties, the sender/receiver accounts
    become ONE node in the merged graph — enabling cross-party message passing.
    All other accounts in B that are not shared seed-nodes become new nodes.

    Args:
        gdata_a: Party A's GraphData (edge_attr[:,0] = global edge IDs, local node IDs)
        gdata_b: Party B's GraphData (same format) or None
        seed_gids_set: set of global edge IDs that are seeds for this (A,B) pair

    Returns:
        merged: PyG Data for the GNN (global-ID column already stripped)
        seed_pos: dict {global_edge_id: edge_position_in_merged}
    """
    if gdata_b is None:
        # Only one party has data: use A's subgraph directly
        a_gids = gdata_a.edge_attr[:, 0].long().tolist()
        seed_pos = {gid: pos for pos, gid in enumerate(a_gids) if gid in seed_gids_set}
        return Data(
            x=gdata_a.x,
            edge_index=gdata_a.edge_index,
            edge_attr=gdata_a.edge_attr[:, 1:],
            num_nodes=gdata_a.x.shape[0],
        ), seed_pos

    N_A = gdata_a.x.shape[0]
    N_B = gdata_b.x.shape[0]

    a_gids = gdata_a.edge_attr[:, 0].long().tolist()
    b_gids = gdata_b.edge_attr[:, 0].long().tolist()

    # Locate seed edges in each party's edge list
    a_seed_pos = {gid: pos for pos, gid in enumerate(a_gids) if gid in seed_gids_set}
    b_seed_pos = {gid: pos for pos, gid in enumerate(b_gids) if gid in seed_gids_set}

    # Default: each B node is a new node offset by N_A
    b_to_merged = torch.arange(N_B, dtype=torch.long) + N_A

    # Reconcile shared seed nodes: B's src/dst → A's src/dst for the same account.
    # The sender and receiver accounts of the seed transaction are the same physical
    # accounts in both parties' graphs, just with different local IDs.
    for gid in seed_gids_set:
        if gid in a_seed_pos and gid in b_seed_pos:
            pa, pb = a_seed_pos[gid], b_seed_pos[gid]
            src_a = int(gdata_a.edge_index[0, pa])
            dst_a = int(gdata_a.edge_index[1, pa])
            src_b = int(gdata_b.edge_index[0, pb])
            dst_b = int(gdata_b.edge_index[1, pb])
            b_to_merged[src_b] = src_a
            b_to_merged[dst_b] = dst_a

    # Compactify: B nodes still at offset (≥ N_A) are genuinely new — renumber them
    # contiguously starting at N_A.
    is_new_b = b_to_merged >= N_A
    new_b_indices = is_new_b.nonzero(as_tuple=True)[0]
    n_new_b = len(new_b_indices)
    b_to_merged[new_b_indices] = torch.arange(n_new_b, dtype=torch.long) + N_A

    # Skip duplicate seed edges already present in A (A's copy is canonical).
    b_dup_mask = torch.zeros(len(b_gids), dtype=torch.bool)
    for gid in a_seed_pos:
        if gid in b_seed_pos:
            b_dup_mask[b_seed_pos[gid]] = True
    b_keep_mask = ~b_dup_mask

    # Remap kept B edges to merged node IDs
    b_edge_index_remapped = b_to_merged[gdata_b.edge_index[:, b_keep_mask]]

    merged_edge_index = torch.cat([gdata_a.edge_index, b_edge_index_remapped], dim=1)
    merged_edge_attr = torch.cat([
        gdata_a.edge_attr[:, 1:],          # strip global-ID col 0
        gdata_b.edge_attr[b_keep_mask][:, 1:],
    ], dim=0)
    merged_x = torch.cat([gdata_a.x, gdata_b.x[new_b_indices]], dim=0)
    N_merged = N_A + n_new_b

    merged = Data(
        x=merged_x,
        edge_index=merged_edge_index,
        edge_attr=merged_edge_attr,
        num_nodes=N_merged,
    )

    # Seed edge positions: A's edges occupy positions 0..len(a_gids)-1 in merged.
    seed_pos = {}
    n_a_edges = len(a_gids)
    for gid in seed_gids_set:
        if gid in a_seed_pos:
            seed_pos[gid] = a_seed_pos[gid]
        elif gid in b_seed_pos:
            # Seed is only in B (rare); find its position among the kept B edges
            b_pos = b_seed_pos[gid]
            pos_in_kept_b = int(b_keep_mask[:b_pos].sum().item())
            seed_pos[gid] = n_a_edges + pos_in_kept_b

    return merged, seed_pos


def forward_pass_restricted_merge(manager, mode, batch_num, batch_banks, batch_data):
    """Merged-pair forward pass for RestrictedMerge.

    Groups batch seeds by (From Bank, To Bank) pair, merges each pair's
    subgraphs into a single connected graph (with seed-node reconciliation),
    runs a single batched GNN forward pass, and classifies with the standard
    mlp head.

    Args:
        manager: FLGNNManagerRestrictedMerge instance.
        mode: 'train', 'vali', or 'test'.
        batch_num: Batch key (typically LAZY_BATCH_KEY).
        batch_banks: List of bank IDs active in this batch.
        batch_data: {bank_id: (party, GraphData)}.

    Returns:
        preds_tensor: [n_samples, n_classes] logits.
        true_y_tensor: [n_samples] ground-truth labels.
    """
    device = manager.device
    batch_df = manager.ctx[mode][batch_num]['batch_labels']
    from_banks = batch_df['From Bank'].values.astype(int)
    to_banks = batch_df['To Bank'].values.astype(int)
    true_y = batch_df['Is Laundering'].values
    seed_gids = batch_df.index.values.astype(int)
    n_samples = len(batch_df)

    # Group seeds by (from_bank, to_bank) pair
    pair_to_seeds = {}
    for i, (fb, tb, gid) in enumerate(zip(from_banks, to_banks, seed_gids)):
        pair = (int(fb), int(tb))
        if pair not in pair_to_seeds:
            pair_to_seeds[pair] = []
        pair_to_seeds[pair].append((i, int(gid)))

    # Build one merged graph per pair; record where each seed ends up
    merged_graphs = []
    sample_graph_idx = [-1] * n_samples
    sample_edge_pos = [-1] * n_samples

    for (bank_a, bank_b), seed_list in pair_to_seeds.items():
        entry_a = batch_data.get(bank_a)
        entry_b = batch_data.get(bank_b)

        if entry_a is None and entry_b is None:
            continue

        # Unpack (party, GraphData) tuples
        raw_a = entry_a[1] if entry_a is not None else None
        raw_b = entry_b[1] if entry_b is not None else None

        # Intra-bank: sender and receiver are the same party — don't double-count
        if bank_a == bank_b:
            raw_b = None

        # Ensure raw_a is the canonical primary subgraph
        if raw_a is None:
            raw_a, raw_b = raw_b, None

        seed_gids_set = {gid for _, gid in seed_list}
        merged, seed_pos = _build_merged_graph(raw_a, raw_b, seed_gids_set)

        graph_idx = len(merged_graphs)
        merged_graphs.append(merged)

        for sample_pos, gid in seed_list:
            if gid in seed_pos:
                sample_graph_idx[sample_pos] = graph_idx
                sample_edge_pos[sample_pos] = seed_pos[gid]

    if not merged_graphs:
        preds = torch.zeros(n_samples, 2, device=device)
        return preds, torch.tensor(true_y, dtype=torch.long, device=device)

    gnn = manager.model.gnn

    graph_to_samples = {}
    for i in range(n_samples):
        if sample_graph_idx[i] >= 0:
            graph_to_samples.setdefault(sample_graph_idx[i], []).append(i)

    # Chunked + checkpointed: activations of all chunks would otherwise be held
    # until backward (merged graphs duplicate hub-bank subgraphs across pairs).
    # Each chunk keeps only its seed embeddings and is recomputed during backward.
    chunk_sample_idx, chunk_embs = [], []
    for chunk_start in range(0, len(merged_graphs), PAIR_CHUNK_SIZE):
        chunk = merged_graphs[chunk_start: chunk_start + PAIR_CHUNK_SIZE]

        samples, positions = [], []
        local_offset = 0
        for j, g in enumerate(chunk):
            for i in graph_to_samples.get(chunk_start + j, []):
                samples.append(i)
                positions.append(local_offset + sample_edge_pos[i])
            local_offset += g.edge_index.shape[1]
        if not samples:
            continue

        combined = Batch.from_data_list(chunk).to(device)
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        args = (gnn, combined.x, combined.edge_attr, combined.edge_index, pos)
        if torch.is_grad_enabled():
            emb = checkpoint(_seed_embeddings, *args, use_reentrant=False)
        else:
            emb = _seed_embeddings(*args)

        chunk_sample_idx.append(torch.tensor(samples, dtype=torch.long, device=device))
        chunk_embs.append(emb)

    embed_dim = 3 * gnn.n_hidden
    sample_embeds = torch.zeros(n_samples, embed_dim, device=device)
    if chunk_embs:
        sample_embeds = sample_embeds.index_put(
            (torch.cat(chunk_sample_idx),), torch.cat(chunk_embs)
        )

    preds_tensor = gnn.mlp(sample_embeds)
    true_y_tensor = torch.tensor(true_y, dtype=torch.long, device=device)

    return preds_tensor, true_y_tensor
