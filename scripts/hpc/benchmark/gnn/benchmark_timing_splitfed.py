"""Timing breakdown for SplitFed, plus a batched-vs-sequential equivalence check.

Per train batch: loader sampling, party subgraph build, GNN, head, other
(assembly, batching, transfer), backward; peak GPU memory; extrapolated epoch
and full-run time. Shared machinery lives in benchmark_timing_comparison.py.

Equivalence check (--check_batched_equivalence N): on N vali batches, runs the
same model through the batched path (all parties in one GPU call) and the
sequential path (one call per party, the reference a real deployment computes),
then reports PASS/FAIL, embedding / probability differences, label flips, F1 for
both, and the time of each path.

Usage:
  python scripts/hpc/benchmark/gnn/benchmark_timing_splitfed.py \\
      --fl_algo SplitFed --size small --ir HI --ibm_hp --emlps \\
      --batching --batching_mode lazy_link_neighbor --eval_mode system --num_rounds 20 \\
      --n_benchmark_epochs 1 --n_benchmark_batches 30 \\
      --check_batched_equivalence 30 --load_weights <trained seed_k/model.pth>
"""

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

import benchmark_timing_comparison as core

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--check_batched_equivalence', type=int, default=0,
                     help='Check batched == sequential forward on N vali batches, and time both')
_parser.add_argument('--load_weights', type=str, default=None,
                     help='model.pth to load before the equivalence check (e.g. a trained seed)')
_parser.add_argument('--check_subgraph_equivalence', type=int, default=0,
                     help='Check the lookup-based party subgraph build == the original isin-based '
                          'build on N vali batches (exact), and time both')
_args, _ = _parser.parse_known_args()

EQUIV_REL_TOL = 1e-4


# ---------------------------------------------------------------------------
# Reference: the original party subgraph build (torch.isin over each party's full
# graph, threaded), kept as the oracle for the subgraph check. Do not optimise.
# Returns {bank: GraphData} instead of writing party.ctx.
# ---------------------------------------------------------------------------

def _reference_party_subgraphs(manager, mode, batch, mode_parties):
    from data.data_utils import GraphData
    from federated_learning.parallel import parallel_party_execute
    max_workers = getattr(manager.args['fl_parser'], 'max_workers', None)

    seed_global_ids = batch.edge_label.long().numpy()
    bl = manager.ctx[mode]['df_labels'].loc[seed_global_ids]
    mode_party_set = set(mode_parties.keys())
    batch_df = bl[bl['From Bank'].isin(mode_party_set) | bl['To Bank'].isin(mode_party_set)]

    seed_ids_tensor = torch.tensor(seed_global_ids, dtype=torch.long)
    batch_global_ids = torch.cat([batch.edge_attr[:, 0].long(), seed_ids_tensor]).unique()

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
        return GraphData(x=party_graph.x[sub_nodes], edge_index=node_remap[matched_edge_index],
                         edge_attr=matched_edge_attr)

    seed_banks = set(batch_df['From Bank'].values) | set(batch_df['To Bank'].values)
    seed_mode_parties = {k: v for k, v in mode_parties.items() if k in seed_banks}
    results = parallel_party_execute(seed_mode_parties, _build_subgraph, max_workers=max_workers)
    return {bid: g for bid, g in results.items() if g is not None}


def run_subgraph_equivalence_check(manager, n_batches: int):
    mode = 'vali'
    mode_parties = manager.get_parties_for_mode(mode)
    n_done = n_parties = 0
    mismatches = {}
    t_old = t_new = 0.0
    for i, batch in enumerate(manager.loaders[mode]):
        if i >= n_batches:
            break
        n_done += 1
        t0 = time.perf_counter()
        ref = _reference_party_subgraphs(manager, mode, batch, mode_parties)
        t_old += time.perf_counter() - t0

        t0 = time.perf_counter()
        core._orig_fs_bat(manager, mode, batch, mode_parties)
        t_new += time.perf_counter() - t0
        new_banks = manager.ctx[mode][core.LAZY_BATCH_KEY]['batch_parties']

        if set(new_banks) != set(ref):
            mismatches['party set'] = mismatches.get('party set', 0) + 1
            continue
        for bank in new_banks:
            n_parties += 1
            g_new = mode_parties[bank].ctx[mode][core.LAZY_BATCH_KEY]['graph_data']
            for field in ('x', 'edge_index', 'edge_attr'):
                a, b = getattr(ref[bank], field), getattr(g_new, field)
                if a.shape != b.shape or not torch.equal(a, b):
                    mismatches[field] = mismatches.get(field, 0) + 1
                    break

    passed = n_parties > 0 and not mismatches
    print(f"\n{'='*74}")
    print(f"  Party subgraph equivalence check — {'PASS' if passed else 'FAIL'} "
          f"(lookup == isin reference, exact)")
    print(f"  {n_parties} party subgraphs over {n_done} vali batches")
    print(f"{'='*74}")
    if mismatches:
        print(f"  mismatches by first differing field: {mismatches}")
    print(f"  subgraph build per batch  reference {t_old / max(n_done, 1) * 1000:9.1f} ms   "
          f"lookup {t_new / max(n_done, 1) * 1000:9.1f} ms   (×{t_old / max(t_new, 1e-9):.1f})")
    print(f"{'='*74}\n", flush=True)


def run_batched_equivalence_check(manager, n_batches: int, load_weights: str | None):
    if load_weights:
        manager.model.gnn.load_state_dict(torch.load(load_weights, map_location=manager.device))
    manager.model.gnn.eval()
    mlp_vert = manager.model.gnn.mlp_vert

    rel_diffs, prob_b, prob_s, labels = [], [], [], []
    path_times = {True: [], False: []}
    n_parties = []
    with torch.no_grad():
        for i, (batch_key, batch_banks, batch_data) in enumerate(manager._iter_batches('vali', batching=True)):
            if i >= n_batches:
                break
            n_parties.append(len(batch_banks))
            views = {}
            for use_batched in (True, False):   # batched first: sequential moves party data to device in place
                core._cuda_sync(); t0 = time.perf_counter()
                v = core._fs_fwd_mod.build_view_embeddings(
                    manager, 'vali', batch_key, batch_banks, batch_data, use_batched)
                core._cuda_sync(); path_times[use_batched].append(time.perf_counter() - t0)
                emb = torch.cat([v['from_embeds'], v['to_embeds']], dim=1)
                views[use_batched] = (emb, torch.softmax(mlp_vert(emb), dim=1)[:, 1])
            (e_b, p_b), (e_s, p_s) = views[True], views[False]
            nz = e_s.norm(dim=1) > 0
            rel_diffs.append(((e_b - e_s).norm(dim=1)[nz] / e_s.norm(dim=1)[nz]).cpu().numpy())
            prob_b.append(p_b.cpu().numpy()); prob_s.append(p_s.cpu().numpy())
            labels.append(np.asarray(v['true_y'], dtype=int))

    rel = np.concatenate(rel_diffs); pb = np.concatenate(prob_b); ps = np.concatenate(prob_s)
    y = np.concatenate(labels)
    flips = (pb > 0.5) != (ps > 0.5)
    passed = rel.max() < EQUIV_REL_TOL and not flips.any()
    print(f"\n{'='*74}")
    print(f"  Batched vs sequential equivalence check — "
          f"{'PASS' if passed else 'FAIL'} (max rel diff < {EQUIV_REL_TOL:g}, no label flips)")
    print(f"  {len(y)} transactions over {min(n_batches, len(rel_diffs))} vali batches, "
          f"{int(y.sum())} illicit, weights: {load_weights or 'benchmark-trained'}")
    print(f"{'='*74}")
    print(f"  embedding relative diff   mean {rel.mean():.2e}   median {np.median(rel):.2e}   max {rel.max():.2e}")
    print(f"  probability abs diff      mean {np.abs(pb - ps).mean():.6f}   max {np.abs(pb - ps).max():.6f}")
    print(f"  predicted label flips     {int(flips.sum())} of {len(y)}  ({flips.mean():.4%}), "
          f"of which illicit: {int((flips & (y == 1)).sum())}")
    print(f"  F1 batched {f1_score(y, pb > 0.5, zero_division=0):.4f}   "
          f"F1 sequential {f1_score(y, ps > 0.5, zero_division=0):.4f}")
    t_b, t_s = np.mean(path_times[True]) * 1000, np.mean(path_times[False]) * 1000
    print(f"  embedding step per batch  batched {t_b:8.1f} ms   sequential {t_s:8.1f} ms   "
          f"(×{t_s / t_b:.1f}, ~{np.mean(n_parties):.0f} parties/batch, forward only, no backward)")
    print(f"{'='*74}\n", flush=True)


def _after_report(manager, algo):
    if _args.check_subgraph_equivalence > 0:
        run_subgraph_equivalence_check(manager, _args.check_subgraph_equivalence)
    if _args.check_batched_equivalence > 0:
        run_batched_equivalence_check(manager, _args.check_batched_equivalence, _args.load_weights)


if __name__ == '__main__':
    core.run_benchmark(['SplitFed'], after_report=_after_report)
