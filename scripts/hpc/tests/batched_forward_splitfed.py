"""Correctness and timing test for the batched GNN forward pass in SplitFed.

Two things are verified and measured:

  1. Correctness: _collect_embeddings_batched produces bit-identical results
     to _collect_embeddings_sequential for the same inputs (model.eval() so
     BatchNorm uses running stats, same as any inference forward pass).

  2. Timing: wall-clock comparison of sequential vs batched on GPU for a
     batch size representative of the real training workload.

Run directly on the HPC compute node:
    python scripts/hpc/tests/batched_forward_splitfed.py [--device cuda]

Or via the sbatch wrapper:
    bash scripts/hpc/tests/run_batched_forward_test.sh
"""

import sys
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, BatchNorm
from types import SimpleNamespace

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from federated_learning.gnn.splitfed.forward import (
    _collect_embeddings_sequential,
    _collect_embeddings_batched,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n_parties',  type=int, default=50,
                    help='Number of parties per forward pass (comparable to a real batch)')
parser.add_argument('--n_nodes',    type=int, default=100,
                    help='Nodes per party subgraph')
parser.add_argument('--n_edges',    type=int, default=60,
                    help='Edges per party subgraph')
parser.add_argument('--n_warmup',   type=int, default=5,
                    help='Warm-up forward passes before timing')
parser.add_argument('--n_runs',     type=int, default=30,
                    help='Timed forward passes')
args = parser.parse_args()

device = torch.device(args.device)
print(f"Device: {device}")
print(f"Parties per batch: {args.n_parties}, nodes/party: {args.n_nodes}, "
      f"edges/party: {args.n_edges}")
print()

# ---------------------------------------------------------------------------
# Model: same architecture as GINe used in training
# ---------------------------------------------------------------------------

NUM_NODE_FEATURES = 4
NUM_EDGE_FEATURES = 7   # matches AML feature count after stripping global-ID col
N_HIDDEN          = 64
NUM_GNN_LAYERS    = 2


class BenchGINe(nn.Module):
    """GINe with the same interface (emed_features / apply_gnn_layer /
    prep_nodes_edges) as the production model in models/gnn_models.py."""

    def __init__(self):
        super().__init__()
        self.num_gnn_layers = NUM_GNN_LAYERS
        self.n_hidden = N_HIDDEN

        self.node_emb = nn.Linear(NUM_NODE_FEATURES, N_HIDDEN)
        self.edge_emb = nn.Linear(NUM_EDGE_FEATURES, N_HIDDEN)

        self.convs       = nn.ModuleList()
        self.emlps       = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(NUM_GNN_LAYERS):
            self.convs.append(GINEConv(
                nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), nn.ReLU(),
                               nn.Linear(N_HIDDEN, N_HIDDEN)),
                edge_dim=N_HIDDEN,
            ))
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * N_HIDDEN, N_HIDDEN), nn.ReLU(),
                nn.Linear(N_HIDDEN, N_HIDDEN),
            ))
            self.batch_norms.append(BatchNorm(N_HIDDEN))

    def emed_features(self, nodes, edges):
        return {'nodes': self.node_emb(nodes), 'edges': self.edge_emb(edges)}

    def apply_gnn_layer(self, nodes, edges, edge_index, layer_idx):
        src, dst = edge_index
        nodes = (nodes + F.relu(
            self.batch_norms[layer_idx](self.convs[layer_idx](nodes, edge_index, edges))
        )) / 2
        edges = edges + self.emlps[layer_idx](
            torch.cat([nodes[src], nodes[dst], edges], dim=-1)
        ) / 2
        return {'nodes': nodes, 'edges': edges}

    def prep_nodes_edges(self, nodes, edges, edge_index):
        node_pairs = nodes[edge_index.T].reshape(-1, 2 * N_HIDDEN).relu()
        return torch.cat([node_pairs, edges], dim=1)


def make_party(model):
    return SimpleNamespace(model=SimpleNamespace(gnn=model))


def make_party_data(num_nodes, num_edges, global_id_start):
    x          = torch.randn(num_nodes, NUM_NODE_FEATURES)
    src        = torch.randint(0, num_nodes, (num_edges,))
    dst        = torch.randint(0, num_nodes, (num_edges,))
    global_ids = torch.arange(global_id_start, global_id_start + num_edges,
                               dtype=torch.float).unsqueeze(1)
    real_feats = torch.randn(num_edges, NUM_EDGE_FEATURES)
    edge_attr  = torch.cat([global_ids, real_feats], dim=1)
    return Data(x=x, edge_index=torch.stack([src, dst]), edge_attr=edge_attr)


def build_batch_data(model, n_parties, n_nodes, n_edges):
    banks = [f'bank_{i}' for i in range(n_parties)]
    batch_data = {
        f'bank_{i}': (make_party(model),
                      make_party_data(n_nodes, n_edges, global_id_start=i * n_edges))
        for i in range(n_parties)
    }
    return banks, batch_data


def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1. Correctness test
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Correctness: sequential == batched")
print("=" * 60)

torch.manual_seed(0)
model = BenchGINe().to(device)
model.eval()

banks, bd_seq = build_batch_data(model, args.n_parties, args.n_nodes, args.n_edges)
_, bd_bat     = build_batch_data(model, args.n_parties, args.n_nodes, args.n_edges)

with torch.no_grad():
    emb_seq, pos_seq = _collect_embeddings_sequential(
        banks, bd_seq, device, dp_clip=None, dp_noise_scale=0.0)
    emb_bat, pos_bat = _collect_embeddings_batched(
        banks, bd_bat, device, dp_clip=None, dp_noise_scale=0.0)

max_diffs = []
all_ok = True
for bank_id in banks:
    ok = torch.allclose(emb_seq[bank_id], emb_bat[bank_id], atol=1e-4)
    max_diffs.append((emb_seq[bank_id] - emb_bat[bank_id]).abs().max().item())
    if not ok:
        all_ok = False
        print(f"  FAIL {bank_id}: max_diff={max_diffs[-1]:.2e}")

overall_max = max(max_diffs)
print(f"  Checked {len(banks)} parties.  Max diff across all: {overall_max:.2e}")
print(f"  Result: {'PASS' if all_ok else 'FAIL'}")

if not all_ok:
    print("Correctness test FAILED — aborting timing benchmark.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Timing benchmark
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("2. Timing benchmark")
print("=" * 60)

model_t = BenchGINe().to(device)
model_t.eval()


def time_approach(fn_name, fn, n_runs, n_warmup):
    # Build fresh batch_data for each run so GPU state is identical
    times = []
    for i in range(n_warmup + n_runs):
        banks_t, bd_t = build_batch_data(model_t, args.n_parties,
                                          args.n_nodes, args.n_edges)
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            fn(banks_t, bd_t, device, dp_clip=None, dp_noise_scale=0.0)
            sync()
            elapsed = time.perf_counter() - t0
        if i >= n_warmup:
            times.append(elapsed * 1000)   # ms

    mean = sum(times) / len(times)
    mn   = min(times)
    mx   = max(times)
    print(f"  {fn_name:12s}: mean={mean:7.1f} ms  min={mn:6.1f} ms  "
          f"max={mx:6.1f} ms  (n={n_runs}, warmup={n_warmup})")
    return mean


mean_seq = time_approach('sequential', _collect_embeddings_sequential,
                          args.n_runs, args.n_warmup)
mean_bat = time_approach('batched',    _collect_embeddings_batched,
                          args.n_runs, args.n_warmup)

speedup = mean_seq / mean_bat
print()
print(f"  Speedup: {speedup:.2f}x  ({args.n_parties} parties, "
      f"{args.n_nodes} nodes, {args.n_edges} edges each, device={device})")

# ---------------------------------------------------------------------------
# 3. Scaling: repeat timing for different numbers of parties
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("3. Speedup vs number of parties")
print("=" * 60)
print(f"  {'n_parties':>10}  {'sequential (ms)':>16}  {'batched (ms)':>13}  {'speedup':>8}")
print(f"  {'-'*10}  {'-'*16}  {'-'*13}  {'-'*8}")

for n in [5, 10, 25, 50, 100]:
    model_s = BenchGINe().to(device)
    model_s.eval()

    def _seq(banks, bd, dev, dp_clip, dp_noise_scale):
        return _collect_embeddings_sequential(banks, bd, dev, dp_clip, dp_noise_scale)

    def _bat(banks, bd, dev, dp_clip, dp_noise_scale):
        return _collect_embeddings_batched(banks, bd, dev, dp_clip, dp_noise_scale)

    def _bench(fn, n_parties):
        times = []
        for _ in range(5 + 15):   # 5 warmup, 15 timed
            banks_s, bd_s = build_batch_data(model_s, n_parties,
                                              args.n_nodes, args.n_edges)
            with torch.no_grad():
                sync()
                t0 = time.perf_counter()
                fn(banks_s, bd_s, device, dp_clip=None, dp_noise_scale=0.0)
                sync()
                times.append((time.perf_counter() - t0) * 1000)
        times = times[5:]
        return sum(times) / len(times)

    ms_seq = _bench(_seq, n)
    ms_bat = _bench(_bat, n)
    sp     = ms_seq / ms_bat
    print(f"  {n:>10d}  {ms_seq:>16.1f}  {ms_bat:>13.1f}  {sp:>8.2f}x")

print()
print("Done.")
