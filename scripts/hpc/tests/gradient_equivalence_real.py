"""
SplitFed gradient equivalence — real AML data, multi-party batches.

Verifies that the explicit multi-party backward pass (detach at embedding
boundary, manager sends ∂L/∂emb back, each party accumulates gradients
and calls emb.backward once) produces identical GNN parameter gradients
to the joint PyTorch backward pass (single computation graph, loss.backward).

Mirrors forward_pass_simple exactly:
- Each batch is a random sample of N_BATCH transactions from all banks.
- Party subgraphs are built once upfront for all banks.
- from_embeds / to_embeds are filled per bank; intra-bank to_embeds = 0.
- N_TESTS independent batches are run with different random seeds.

Runs on CPU (GPU scatter_add is non-deterministic; test requires exact
floating-point reproducibility).

Output: gradient_equivalence_results.csv in the same directory as this script.

Usage:
  python scripts/hpc/tests/gradient_equivalence_real.py
  python scripts/hpc/tests/gradient_equivalence_real.py --n_tests 10 --n_batch 500
"""

import sys
import os

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from configs.paths import get_data_path
from data.data_utils import z_norm
from models.gnn_models import GINe


IBM_EDGE_FEATURES = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
IBM_N_HIDDEN      = 66
IBM_N_LAYERS      = 2


def load_csv(size, ir, n_rows):
    path = os.path.join(get_data_path(), 'AML_work_study',
                        f'formatted_transactions_{size}_{ir}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, nrows=(n_rows if n_rows > 0 else None))
    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()
    return df


def build_full_graph(df):
    max_node   = int(df[['from_id', 'to_id']].to_numpy().max()) + 1
    x          = torch.ones(max_node, 1)
    edge_index = torch.tensor(df[['from_id', 'to_id']].to_numpy().T, dtype=torch.long)
    raw        = torch.tensor(df[IBM_EDGE_FEATURES].to_numpy(), dtype=torch.float)
    global_id  = torch.arange(raw.shape[0], dtype=torch.float).unsqueeze(1)
    edge_attr  = torch.cat([global_id, raw], dim=1)
    y          = torch.tensor(df['Is Laundering'].to_numpy(), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def apply_ibm_fe(graph):
    if graph.x.shape[0] > 1:         graph.x                = z_norm(graph.x)
    if graph.edge_attr.shape[0] > 1: graph.edge_attr[:, 1:] = z_norm(graph.edge_attr[:, 1:])
    return graph


def extract_party_subgraph(full_graph, edge_mask):
    idx            = torch.tensor(np.where(edge_mask)[0], dtype=torch.long)
    sub_edge_index = full_graph.edge_index[:, idx]
    sub_edge_attr  = full_graph.edge_attr[idx]
    sub_y          = full_graph.y[idx]
    all_nodes      = sub_edge_index.reshape(-1).unique()
    remap          = torch.zeros(full_graph.x.shape[0], dtype=torch.long)
    remap[all_nodes] = torch.arange(len(all_nodes), dtype=torch.long)
    return Data(x=full_graph.x[all_nodes],
                edge_index=remap[sub_edge_index],
                edge_attr=sub_edge_attr, y=sub_y)


def build_all_party_subgraphs(full_graph, from_arr, to_arr):
    """Build one subgraph per bank — done once, reused across all test batches."""
    all_banks    = np.unique(np.concatenate([from_arr, to_arr]))
    party_graphs = {}
    for bank_id in all_banks:
        mask = (from_arr == bank_id) | (to_arr == bank_id)
        if mask.sum() > 0:
            party_graphs[int(bank_id)] = extract_party_subgraph(full_graph, mask)
    return party_graphs


def run_gnn(gnn, graph):
    emb = gnn.emed_features(graph.x, graph.edge_attr[:, 1:])
    for i in range(gnn.num_gnn_layers):
        emb = gnn.apply_gnn_layer(emb['nodes'], emb['edges'], graph.edge_index, i)
    return gnn.prep_nodes_edges(emb['nodes'], emb['edges'], graph.edge_index)


def run_batch_test(gnn, party_graphs, batch_df, from_arr_full, to_arr_full, atol, rtol):
    """
    Test gradient equivalence for one random multi-party batch.

    Mirrors forward_pass_simple: from_embeds / to_embeds filled per bank;
    intra-bank transactions keep to_embeds = 0. Gradients are accumulated
    per bank before calling backward once per bank.
    """
    n          = len(batch_df)
    indices    = batch_df.index.values.astype(int)
    from_banks = from_arr_full[indices]
    to_banks   = to_arr_full[indices]
    labels     = torch.tensor(batch_df['Is Laundering'].values, dtype=torch.long)
    loss_fn    = torch.nn.CrossEntropyLoss(reduction='sum')
    intra_mask = from_banks == to_banks

    active_banks = [b for b in np.unique(np.concatenate([from_banks, to_banks]))
                    if b in party_graphs]
    ip = {b: {int(gid): pos for pos, gid in enumerate(party_graphs[b].edge_attr[:, 0].cpu())}
          for b in active_banks}

    # ── Joint path ────────────────────────────────────────────────────────────
    gnn.zero_grad()
    emb       = {b: run_gnn(gnn, party_graphs[b]) for b in active_banks}
    embed_dim = next(iter(emb.values())).shape[1]

    from_embeds = torch.zeros(n, embed_dim)
    to_embeds   = torch.zeros(n, embed_dim)

    for bank in np.unique(from_banks):
        if bank not in ip: continue
        mask = from_banks == bank
        pos  = [ip[bank][idx] for idx in indices[mask]]
        from_embeds[mask] = emb[bank][pos]

    for bank in np.unique(to_banks):
        if bank not in ip: continue
        mask = (to_banks == bank) & ~intra_mask
        if not mask.any(): continue
        pos = [ip[bank][idx] for idx in indices[mask]]
        to_embeds[mask] = emb[bank][pos]
    # intra-bank rows: to_embeds stays 0 (matches forward_pass_simple)

    logits_j = gnn.mlp_vert(torch.cat([from_embeds, to_embeds], dim=1))
    (loss_fn(logits_j, labels) / n).backward()
    grad_joint = {nm: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                  for nm, p in gnn.named_parameters()}

    # ── Manual path ───────────────────────────────────────────────────────────
    gnn.zero_grad()
    emb_m = {b: run_gnn(gnn, party_graphs[b]) for b in active_banks}

    from_proxy = torch.zeros(n, embed_dim)
    to_proxy   = torch.zeros(n, embed_dim)

    for bank in np.unique(from_banks):
        if bank not in ip: continue
        mask = from_banks == bank
        pos  = [ip[bank][idx] for idx in indices[mask]]
        from_proxy[mask] = emb_m[bank][pos].detach()

    for bank in np.unique(to_banks):
        if bank not in ip: continue
        mask = (to_banks == bank) & ~intra_mask
        if not mask.any(): continue
        pos = [ip[bank][idx] for idx in indices[mask]]
        to_proxy[mask] = emb_m[bank][pos].detach()

    from_proxy.requires_grad_(True)
    to_proxy.requires_grad_(True)

    logits_m = gnn.mlp_vert(torch.cat([from_proxy, to_proxy], dim=1))
    (loss_fn(logits_m, labels) / n).backward()

    # Accumulate from + to gradients per bank, then backward once per bank
    grad_per_bank = {b: torch.zeros_like(emb_m[b]) for b in active_banks}

    for bank in np.unique(from_banks):
        if bank not in ip: continue
        mask = from_banks == bank
        pos  = [ip[bank][idx] for idx in indices[mask]]
        for i, p in zip(np.where(mask)[0], pos):
            grad_per_bank[bank][p] += from_proxy.grad[i]

    for bank in np.unique(to_banks):
        if bank not in ip: continue
        mask = (to_banks == bank) & ~intra_mask
        if not mask.any(): continue
        pos = [ip[bank][idx] for idx in indices[mask]]
        for i, p in zip(np.where(mask)[0], pos):
            grad_per_bank[bank][p] += to_proxy.grad[i]

    for bank, grad in grad_per_bank.items():
        emb_m[bank].backward(grad)

    grad_manual = {nm: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                   for nm, p in gnn.named_parameters()}

    # ── Compare ───────────────────────────────────────────────────────────────
    max_diffs = {}
    all_pass  = True
    for nm in grad_joint:
        diff          = (grad_joint[nm] - grad_manual[nm]).abs().max().item()
        max_diffs[nm] = diff
        if not torch.allclose(grad_joint[nm], grad_manual[nm], atol=atol, rtol=rtol):
            all_pass = False

    n_cross = int((~intra_mask).sum())
    return all_pass, max_diffs, n_cross


def main():
    parser = argparse.ArgumentParser(
        description="SplitFed gradient equivalence — real AML data, multi-party batches"
    )
    parser.add_argument('--size',    default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--ir',      default='HI',    choices=['HI', 'LO'])
    parser.add_argument('--n_rows',  default=0,       type=int,
                        help='CSV rows to load (0 = full file)')
    parser.add_argument('--n_tests', default=10,      type=int,
                        help='Number of independent random batches to test')
    parser.add_argument('--n_batch', default=500,     type=int,
                        help='Transactions per batch')
    parser.add_argument('--atol',    default=1e-5,    type=float)
    parser.add_argument('--rtol',    default=1e-4,    type=float)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Loading {args.n_rows or 'all'} rows from "
          f"formatted_transactions_{args.size}_{args.ir}.csv …")
    df       = load_csv(args.size, args.ir, args.n_rows)
    df_train = df.head(int(len(df) * 0.6)).reset_index(drop=True)
    print(f"  Train rows: {len(df_train):,}")

    from_arr = df_train['From Bank'].values.astype(int)
    to_arr   = df_train['To Bank'].values.astype(int)

    print("  Building full graph and applying ibm_fe normalisation …")
    full_graph = apply_ibm_fe(build_full_graph(df_train))

    print("  Building party subgraphs …")
    party_graphs = build_all_party_subgraphs(full_graph, from_arr, to_arr)
    print(f"  {len(party_graphs)} party subgraphs built\n")

    node_dim = 1
    edge_dim = full_graph.edge_attr.shape[1] - 1

    gnn = GINe(
        num_features  =node_dim,
        num_gnn_layers=IBM_N_LAYERS,
        n_classes     =2,
        n_hidden      =IBM_N_HIDDEN,
        edge_updates  =True,
        edge_dim      =edge_dim,
        dropout       =0.0,
        final_dropout =0.0,
        batching      =False,
    )
    gnn.eval()

    print(f"GINe: node_dim={node_dim}, edge_dim={edge_dim}, "
          f"n_hidden={IBM_N_HIDDEN}, n_layers={IBM_N_LAYERS}")
    print(f"Running {args.n_tests} random batches of {args.n_batch} transactions …\n")

    rows        = []
    overall_max = {}

    for t in range(args.n_tests):
        batch_df = df_train.sample(n=args.n_batch, random_state=t)
        ok, max_diffs, n_cross = run_batch_test(
            gnn, party_graphs, batch_df, from_arr, to_arr, args.atol, args.rtol)
        worst  = max(max_diffs.values())
        status = 'PASS' if ok else 'FAIL'
        n_banks = len(set(batch_df['From Bank'].tolist() + batch_df['To Bank'].tolist()))
        print(f"  Batch {t+1:2d}  banks={n_banks:2d}  cross-party={n_cross:4d}  "
              f"{status}  max|Δg|={worst:.2e}")

        rows.append({'batch': t + 1, 'transactions': args.n_batch,
                     'n_banks': n_banks, 'n_cross_party': n_cross,
                     'max_abs_diff': worst, 'result': status})
        for nm, d in max_diffs.items():
            overall_max[nm] = max(overall_max.get(nm, 0.0), d)

    overall_pass = all(r['result'] == 'PASS' for r in rows)
    worst_global = max(overall_max.values())

    print(f"\nOverall: {'ALL PASS' if overall_pass else 'SOME FAILURES'}")
    print(f"Worst max |Δg| across all batches and parameters: {worst_global:.2e}")
    print(f"atol={args.atol}, rtol={args.rtol}")

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, 'gradient_equivalence_results.csv')
    df_out   = pd.DataFrame(rows)
    df_out['worst_global'] = worst_global
    df_out['overall_pass'] = overall_pass
    df_out['size']         = args.size
    df_out['ir']           = args.ir
    df_out['atol']         = args.atol
    df_out['rtol']         = args.rtol
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
