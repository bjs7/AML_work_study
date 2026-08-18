"""
Empirical gradient equivalence test — real AML data, full dataset.

Verifies that the explicit multi-party backward pass (detach at embedding
boundary, manager sends ∂L/∂emb back, each party backpropagates locally)
produces identical GNN parameter gradients to the joint PyTorch backward pass.

Saves per-batch results to gradient_equivalence_results.csv in the same
directory as this script.

Usage:
  python scripts/hpc/tests/gradient_equivalence_real.py
  python scripts/hpc/tests/gradient_equivalence_real.py --size small --ir HI --n_batches 10
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
    df = pd.read_csv(path, nrows=n_rows)
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


def select_bank_pair(df, min_edges=200):
    bank_sizes = pd.concat([df['From Bank'], df['To Bank']]).value_counts()
    valid = set(bank_sizes[bank_sizes >= min_edges].index)
    ab = df[(df['From Bank'] != df['To Bank']) &
            df['From Bank'].isin(valid) & df['To Bank'].isin(valid)]
    if ab.empty:
        raise ValueError(f"No cross-party edges between banks with >= {min_edges} edges.")
    pair = ab.groupby(['From Bank', 'To Bank']).size().idxmax()
    return int(pair[0]), int(pair[1])


def run_gnn(gnn, graph):
    emb = gnn.emed_features(graph.x, graph.edge_attr[:, 1:])
    for i in range(gnn.num_gnn_layers):
        emb = gnn.apply_gnn_layer(emb['nodes'], emb['edges'], graph.edge_index, i)
    return gnn.prep_nodes_edges(emb['nodes'], emb['edges'], graph.edge_index)


def run_batch_test(gnn, graph_A, graph_B, batch_df, atol, rtol):
    labels  = torch.tensor(batch_df['Is Laundering'].values, dtype=torch.long)
    n       = len(batch_df)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    ip_A = {int(gid): pos for pos, gid in enumerate(graph_A.edge_attr[:, 0].cpu())}
    ip_B = {int(gid): pos for pos, gid in enumerate(graph_B.edge_attr[:, 0].cpu())}

    indices     = batch_df.index.values.astype(int)
    A_positions = [ip_A[gid] for gid in indices]
    B_positions = [ip_B[gid] for gid in indices]

    # ── Joint path ────────────────────────────────────────────────────────────
    gnn.zero_grad()
    emb_A_j  = run_gnn(gnn, graph_A)
    emb_B_j  = run_gnn(gnn, graph_B)
    logits_j = gnn.mlp_vert(torch.cat([emb_A_j[A_positions], emb_B_j[B_positions]], dim=1))
    (loss_fn(logits_j, labels) / n).backward()
    grad_joint = {nm: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                  for nm, p in gnn.named_parameters()}

    # ── Manual path ───────────────────────────────────────────────────────────
    gnn.zero_grad()
    emb_A_m    = run_gnn(gnn, graph_A)
    emb_B_m    = run_gnn(gnn, graph_B)
    from_proxy = emb_A_m[A_positions].detach().requires_grad_(True)
    to_proxy   = emb_B_m[B_positions].detach().requires_grad_(True)
    logits_m   = gnn.mlp_vert(torch.cat([from_proxy, to_proxy], dim=1))
    (loss_fn(logits_m, labels) / n).backward()

    grad_A = torch.zeros_like(emb_A_m)
    for i, pos in enumerate(A_positions):
        grad_A[pos] += from_proxy.grad[i]
    emb_A_m.backward(grad_A)

    grad_B = torch.zeros_like(emb_B_m)
    for i, pos in enumerate(B_positions):
        grad_B[pos] += to_proxy.grad[i]
    emb_B_m.backward(grad_B)

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

    return all_pass, max_diffs


def main():
    parser = argparse.ArgumentParser(
        description="SplitFed gradient equivalence — real AML data, full dataset"
    )
    parser.add_argument('--size',      default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--ir',        default='HI',    choices=['HI', 'LO'])
    parser.add_argument('--n_rows',    default=0,       type=int,
                        help='CSV rows to load (0 = full file)')
    parser.add_argument('--n_batch',   default=8192,    type=int,
                        help='Transactions per test batch (default: 8192, matches training)')
    parser.add_argument('--n_batches', default=10,      type=int,
                        help='Number of random batches to test')
    parser.add_argument('--atol',      default=1e-5,    type=float)
    parser.add_argument('--rtol',      default=1e-4,    type=float)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    n_rows = args.n_rows if args.n_rows > 0 else None
    print(f"Loading {n_rows or 'all'} rows from "
          f"formatted_transactions_{args.size}_{args.ir}.csv …")
    df       = load_csv(args.size, args.ir, n_rows)
    df_train = df.head(int(len(df) * 0.6)).reset_index(drop=True)
    print(f"  Train rows: {len(df_train):,}")

    bank_A, bank_B = select_bank_pair(df_train)
    print(f"  Selected banks: A={bank_A}, B={bank_B}")

    full_graph = apply_ibm_fe(build_full_graph(df_train))

    from_arr = df_train['From Bank'].values
    to_arr   = df_train['To Bank'].values
    mask_A   = (from_arr == bank_A) | (to_arr == bank_A)
    mask_B   = (from_arr == bank_B) | (to_arr == bank_B)
    graph_A  = extract_party_subgraph(full_graph, mask_A)
    graph_B  = extract_party_subgraph(full_graph, mask_B)

    ab_mask = (from_arr == bank_A) & (to_arr == bank_B)
    ab_df   = df_train[ab_mask]
    print(f"  Party A edges: {graph_A.edge_attr.shape[0]:,}  nodes: {graph_A.x.shape[0]:,}")
    print(f"  Party B edges: {graph_B.edge_attr.shape[0]:,}  nodes: {graph_B.x.shape[0]:,}")
    print(f"  A→B transactions available: {len(ab_df):,}")

    n_batch = min(args.n_batch, len(ab_df))
    if n_batch < args.n_batch:
        print(f"  Warning: only {len(ab_df)} A→B transactions — capping n_batch at {n_batch}")

    node_dim = graph_A.x.shape[1]
    edge_dim = graph_A.edge_attr.shape[1] - 1

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

    print(f"\nGINe: node_dim={node_dim}, edge_dim={edge_dim}, "
          f"n_hidden={IBM_N_HIDDEN}, n_layers={IBM_N_LAYERS}")
    print(f"Running {args.n_batches} batch(es) of {n_batch} A→B transactions …\n")

    batch_rows  = []
    overall_max = {}

    for b in range(args.n_batches):
        batch_df = ab_df.sample(n=n_batch, random_state=b)
        ok, max_diffs = run_batch_test(gnn, graph_A, graph_B, batch_df, args.atol, args.rtol)
        worst  = max(max_diffs.values())
        status = 'PASS' if ok else 'FAIL'
        print(f"  Batch {b+1}: {status}  worst |diff| = {worst:.2e}")
        batch_rows.append({'batch': b + 1, 'n_transactions': n_batch,
                           'max_abs_diff': worst, 'result': status})
        for nm, d in max_diffs.items():
            overall_max[nm] = max(overall_max.get(nm, 0.0), d)

    overall_pass = all(r['result'] == 'PASS' for r in batch_rows)
    worst_global = max(overall_max.values())

    print(f"\nOverall: {'ALL PASS' if overall_pass else 'SOME FAILURES'}")
    print(f"Worst max |diff| across all batches and parameters: {worst_global:.2e}")
    print(f"atol={args.atol}, rtol={args.rtol}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, 'gradient_equivalence_results.csv')
    df_out = pd.DataFrame(batch_rows)
    df_out['worst_global'] = worst_global
    df_out['overall_pass'] = overall_pass
    df_out['size'] = args.size
    df_out['ir']   = args.ir
    df_out['atol'] = args.atol
    df_out['rtol'] = args.rtol
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
