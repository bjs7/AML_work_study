"""Walk-through and test of the lookup-based party subgraph build (splitfed/batching.py).

Part 1 builds a tiny example (3 banks, 7 transactions) and prints every step:
the party graphs, the lookup table from build_edge_lookup, the positions found by
_party_edge_positions for one batch, and the subgraphs from process_lazy_batch_splitfed,
each compared with the original torch.isin approach.

Part 2 is a randomised stress test (many banks, shuffled party edge order, many
batches) that requires exact equality with the original approach.

No data needed. Run from the repo root:
    python scripts/hpc/tests/test_edge_lookup.py
"""

import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import torch

from data.data_utils import GraphData
from federated_learning.gnn.fedgraph.batching import LAZY_BATCH_KEY
from federated_learning.gnn.splitfed.batching import (
    build_edge_lookup, _party_edge_positions, process_lazy_batch_splitfed)

MODE = 'train'


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def make_party_graph(bank, transactions, rng=None):
    """A party's graph: every transaction where the bank is sender or receiver, with
    local node IDs (accounts renumbered) and edge_attr = [global ID, amount].
    With rng, the edge order is shuffled to show positions need not follow global IDs."""
    rows = [t for t in transactions if bank in (t['from_bank'], t['to_bank'])]
    if rng is not None:
        rows = [rows[i] for i in rng.permutation(len(rows))]
    accounts = list(dict.fromkeys(a for t in rows for a in (t['from_acct'], t['to_acct'])))
    local = {a: i for i, a in enumerate(accounts)}
    return GraphData(
        x=torch.tensor(accounts, dtype=torch.float).unsqueeze(1),   # x = account ID, for readability
        edge_index=torch.tensor([[local[t['from_acct']] for t in rows],
                                 [local[t['to_acct']] for t in rows]], dtype=torch.long),
        edge_attr=torch.tensor([[t['gid'], t['amount']] for t in rows], dtype=torch.float),
    )


def make_parties(transactions, banks, rng=None):
    return {b: SimpleNamespace(procs_data={f'{MODE}_data': {'df': make_party_graph(b, transactions, rng)}},
                               ctx={MODE: defaultdict(dict)})
            for b in banks}


def make_manager(transactions, parties):
    labels = pd.DataFrame({'From Bank': [t['from_bank'] for t in transactions],
                           'To Bank': [t['to_bank'] for t in transactions],
                           'Is Laundering': 0},
                          index=[t['gid'] for t in transactions])
    ctx = {MODE: defaultdict(dict)}
    ctx[MODE]['df_labels'] = labels
    return SimpleNamespace(ctx=ctx, edge_lookup={MODE: build_edge_lookup(parties, MODE)})


def make_batch(seed_gids, sampled_gids):
    """What the LinkNeighborLoader hands over: seed transactions + sampled neighbour edges."""
    return SimpleNamespace(edge_label=torch.tensor(seed_gids, dtype=torch.float),
                           edge_attr=torch.tensor(sampled_gids, dtype=torch.float).unsqueeze(1))


# Original approach, as used before the lookup: torch.isin over each party's full graph.
def reference_positions(party_graph, batch_global_ids):
    mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)
    return mask.nonzero(as_tuple=True)[0]


def reference_subgraph(party_graph, batch_global_ids):
    mask = torch.isin(party_graph.edge_attr[:, 0].long(), batch_global_ids)
    if mask.sum() == 0:
        return None
    matched = party_graph.edge_index[:, mask]
    sub_nodes = matched.reshape(-1).unique()
    remap = torch.zeros(party_graph.x.shape[0], dtype=torch.long)
    remap[sub_nodes] = torch.arange(len(sub_nodes))
    return GraphData(x=party_graph.x[sub_nodes], edge_index=remap[matched],
                     edge_attr=party_graph.edge_attr[mask])


def batch_ids(batch):
    seeds = batch.edge_label.long()
    return torch.cat([batch.edge_attr[:, 0].long(), seeds]).unique()


def same_graph(a, b):
    return all(torch.equal(getattr(a, f), getattr(b, f)) for f in ('x', 'edge_index', 'edge_attr'))


# ---------------------------------------------------------------------------
# Part 1: small example, printed step by step
# ---------------------------------------------------------------------------

def part1():
    print('=' * 78)
    print('PART 1 — small example')
    print('=' * 78)
    T = [  # accounts: 1xx at bank 10, 2xx at bank 20, 3xx at bank 30
        dict(gid=100, from_bank=10, to_bank=20, from_acct=101, to_acct=201, amount=50),
        dict(gid=101, from_bank=10, to_bank=10, from_acct=102, to_acct=101, amount=20),  # intra-bank
        dict(gid=102, from_bank=20, to_bank=30, from_acct=201, to_acct=301, amount=70),
        dict(gid=103, from_bank=30, to_bank=10, from_acct=301, to_acct=102, amount=10),
        dict(gid=104, from_bank=20, to_bank=20, from_acct=202, to_acct=201, amount=30),  # intra-bank
        dict(gid=105, from_bank=10, to_bank=30, from_acct=101, to_acct=302, amount=90),
        dict(gid=106, from_bank=30, to_bank=20, from_acct=302, to_acct=202, amount=40),
    ]
    banks = [10, 20, 30]
    parties = make_parties(T, banks)

    print('\nTransactions (global ID: sender account → receiver account):')
    for t in T:
        kind = ' (intra-bank)' if t['from_bank'] == t['to_bank'] else ''
        print(f"  {t['gid']}: {t['from_acct']} (bank {t['from_bank']}) → {t['to_acct']} (bank {t['to_bank']}){kind}")

    print('\nStep 1 — each party graph (local edge position → global ID, local node IDs):')
    for b in banks:
        g = parties[b].procs_data[f'{MODE}_data']['df']
        accts = g.x[:, 0].long().tolist()
        print(f'  bank {b}: accounts (local node 0..{len(accts) - 1}) = {accts}')
        for p in range(g.edge_index.shape[1]):
            s, d = g.edge_index[:, p].tolist()
            print(f'    position {p}: transaction {int(g.edge_attr[p, 0])}   local {s}→{d}')

    manager = make_manager(T, parties)
    gids, ebanks, epos = manager.edge_lookup[MODE]
    print('\nStep 2 — build_edge_lookup: one entry per (transaction, party graph), sorted by global ID')
    print('  (cross-bank transactions appear twice, intra-bank once)')
    print(f"  {'global ID':>10} {'bank':>6} {'position':>9}")
    for g, b, p in zip(gids.tolist(), ebanks.tolist(), epos.tolist()):
        print(f'  {g:>10} {b:>6} {p:>9}')

    batch = make_batch(seed_gids=[102, 104], sampled_gids=[100, 106])
    ids = batch_ids(batch)
    print(f'\nStep 3 — a batch: seeds 102, 104; sampled neighbours 100, 106 → batch IDs {ids.tolist()}')
    seed_tx = [t for t in T if t['gid'] in (102, 104)]
    seed_banks = {t['from_bank'] for t in seed_tx} | {t['to_bank'] for t in seed_tx}
    print(f'  banks of the seed transactions (only these build a subgraph): {sorted(seed_banks)}')

    found = _party_edge_positions(manager.edge_lookup[MODE], ids, seed_banks)
    print('\nStep 4 — _party_edge_positions vs torch.isin over each full party graph:')
    for b in sorted(seed_banks):
        ref = reference_positions(parties[b].procs_data[f'{MODE}_data']['df'], ids)
        new = found.get(b, torch.empty(0, dtype=torch.long))
        print(f'  bank {b}: lookup {new.tolist()}   isin {ref.tolist()}   '
              f'{"OK" if torch.equal(new, ref) else "MISMATCH"}')
        assert torch.equal(new, ref)

    process_lazy_batch_splitfed(manager, MODE, batch, parties)
    print('\nStep 5 — process_lazy_batch_splitfed subgraphs vs the original build:')
    print(f"  batch_parties = {manager.ctx[MODE][LAZY_BATCH_KEY]['batch_parties']}")
    for b in manager.ctx[MODE][LAZY_BATCH_KEY]['batch_parties']:
        new = parties[b].ctx[MODE][LAZY_BATCH_KEY]['graph_data']
        ref = reference_subgraph(parties[b].procs_data[f'{MODE}_data']['df'], ids)
        print(f'  bank {b}: accounts {new.x[:, 0].long().tolist()}, '
              f'edges {new.edge_index.t().tolist()}, transactions {new.edge_attr[:, 0].long().tolist()}   '
              f'{"OK" if same_graph(new, ref) else "MISMATCH"}')
        assert same_graph(new, ref)
    print('\nPart 1: all steps match the original approach.')


# ---------------------------------------------------------------------------
# Part 2: randomised stress test
# ---------------------------------------------------------------------------

def part2(n_banks=60, n_accounts=3000, n_tx=20000, n_batches=200, seed=0):
    print('\n' + '=' * 78)
    print(f'PART 2 — stress test ({n_banks} banks, {n_accounts} accounts, {n_tx} transactions, '
          f'{n_batches} batches, shuffled party edge order)')
    print('=' * 78)
    rng = np.random.default_rng(seed)
    acct_bank = rng.integers(0, n_banks, n_accounts) * 10 + 10
    T = []
    for gid in range(n_tx):
        a, b = rng.integers(0, n_accounts, 2)
        T.append(dict(gid=1000 + gid, from_bank=int(acct_bank[a]), to_bank=int(acct_bank[b]),
                      from_acct=int(a), to_acct=int(b), amount=float(rng.random())))
    banks = sorted(set(acct_bank.tolist()))
    parties = make_parties(T, banks, rng=rng)
    manager = make_manager(T, parties)

    n_subgraphs, t_ref, t_new = 0, 0.0, 0.0
    all_gids = np.array([t['gid'] for t in T])
    for _ in range(n_batches):
        seeds = rng.choice(all_gids, 256, replace=False)
        sampled = rng.choice(all_gids, 2000, replace=False)
        batch = make_batch(seeds.tolist(), sampled.tolist())
        ids = batch_ids(batch)

        t0 = time.perf_counter()
        process_lazy_batch_splitfed(manager, MODE, batch, parties)
        t_new += time.perf_counter() - t0
        new_banks = manager.ctx[MODE][LAZY_BATCH_KEY]['batch_parties']

        bl = manager.ctx[MODE][LAZY_BATCH_KEY]['batch_labels']
        seed_banks = set(bl['From Bank']) | set(bl['To Bank'])
        t0 = time.perf_counter()
        ref = {b: reference_subgraph(parties[b].procs_data[f'{MODE}_data']['df'], ids) for b in seed_banks}
        t_ref += time.perf_counter() - t0
        ref = {b: g for b, g in ref.items() if g is not None}

        assert set(new_banks) == set(ref), 'different set of parties'
        for b in new_banks:
            assert same_graph(parties[b].ctx[MODE][LAZY_BATCH_KEY]['graph_data'], ref[b]), f'bank {b} differs'
            n_subgraphs += 1

    print(f'  PASS — {n_subgraphs} party subgraphs identical over {n_batches} batches')
    print(f'  time per batch: original (isin, sequential) {t_ref / n_batches * 1000:.1f} ms   '
          f'lookup {t_new / n_batches * 1000:.1f} ms')


if __name__ == '__main__':
    part1()
    part2()
