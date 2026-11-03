"""Timing breakdown for RestrictedMerge, plus a merge-graph equivalence check.

Per train batch: loader sampling, party subgraph build, GNN, merge-graph build
(CPU), head, other (assembly, batching, transfer), backward with the checkpoint
recompute shown separately; peak GPU memory; extrapolated epoch and full-run time.
Shared machinery lives in benchmark_timing_comparison.py; this file adds the
RestrictedMerge-specific timing hooks.

Merge equivalence check (--check_merge_equivalence N): on N vali batches, builds
every (From Bank, To Bank) pair's merged graph with the current vectorised
_build_merged_graph and with the original loop-based implementation kept below as
the reference, and requires exact equality (x, edge_index, edge_attr, num_nodes,
seed positions). Reports PASS/FAIL and the merge time of both.

Usage:
  python scripts/hpc/benchmark/gnn/benchmark_timing_restricted_merge.py \\
      --fl_algo RestrictedMerge --size small --ir HI --ibm_hp --emlps \\
      --batching --batching_mode lazy_link_neighbor --eval_mode system --num_rounds 20 \\
      --n_benchmark_epochs 1 --n_benchmark_batches 30 --check_merge_equivalence 10
"""

import argparse
import time

import torch
from torch_geometric.data import Data

import benchmark_timing_comparison as core
import federated_learning.gnn.restricted_merge.forward as _rm_fwd_mod

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--check_merge_equivalence', type=int, default=0,
                     help='Check vectorised merge == reference merge on N vali batches, and time both')
_parser.add_argument('--check_chunk_sizes', type=int, default=0,
                     help='On N vali batches: check results are identical across PAIR_CHUNK_SIZE '
                          'values (per-pair LayerNorm) and time train fwd+bwd / peak GPU per size')
_parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[16, 64, 256, 1024])
_args, _ = _parser.parse_known_args()

_orig_rm_fwd = _rm_fwd_mod.forward_pass_restricted_merge
_orig_rm_merge = _rm_fwd_mod._build_merged_graph
_orig_seed_index = _rm_fwd_mod._seed_index


def _t_rm_fwd(*a, **kw):
    t = core._active_timer
    if t is None: return _orig_rm_fwd(*a, **kw)
    with t.section('forward_total'): return _orig_rm_fwd(*a, **kw)


def _t_rm_merge(*a, **kw):
    t = core._active_timer
    if t is None: return _orig_rm_merge(*a, **kw)
    with t.section('rm_merge_build'): return _orig_rm_merge(*a, **kw)


def _t_seed_index(*a, **kw):
    t = core._active_timer
    if t is None: return _orig_seed_index(*a, **kw)
    with t.section('rm_merge_build'): return _orig_seed_index(*a, **kw)


# The manager calls rm_forward.forward_pass_restricted_merge and the forward pass calls
# the helpers by module-global lookup, so patching the module attributes is enough.
_rm_fwd_mod.forward_pass_restricted_merge = _t_rm_fwd
_rm_fwd_mod._build_merged_graph = _t_rm_merge
_rm_fwd_mod._seed_index = _t_seed_index


# ---------------------------------------------------------------------------
# Reference: the original loop-based merge (before vectorisation), kept as the
# oracle for the equivalence check. Do not optimise.
# ---------------------------------------------------------------------------

def _build_merged_graph_reference(gdata_a, gdata_b, seed_gids_set):
    if gdata_b is None:
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

    a_seed_pos = {gid: pos for pos, gid in enumerate(a_gids) if gid in seed_gids_set}
    b_seed_pos = {gid: pos for pos, gid in enumerate(b_gids) if gid in seed_gids_set}

    b_to_merged = torch.arange(N_B, dtype=torch.long) + N_A

    for gid in seed_gids_set:
        if gid in a_seed_pos and gid in b_seed_pos:
            pa, pb = a_seed_pos[gid], b_seed_pos[gid]
            src_a = int(gdata_a.edge_index[0, pa])
            dst_a = int(gdata_a.edge_index[1, pa])
            src_b = int(gdata_b.edge_index[0, pb])
            dst_b = int(gdata_b.edge_index[1, pb])
            b_to_merged[src_b] = src_a
            b_to_merged[dst_b] = dst_a

    is_new_b = b_to_merged >= N_A
    new_b_indices = is_new_b.nonzero(as_tuple=True)[0]
    n_new_b = len(new_b_indices)
    b_to_merged[new_b_indices] = torch.arange(n_new_b, dtype=torch.long) + N_A

    b_dup_mask = torch.zeros(len(b_gids), dtype=torch.bool)
    for gid in a_seed_pos:
        if gid in b_seed_pos:
            b_dup_mask[b_seed_pos[gid]] = True
    b_keep_mask = ~b_dup_mask

    b_edge_index_remapped = b_to_merged[gdata_b.edge_index[:, b_keep_mask]]

    merged_edge_index = torch.cat([gdata_a.edge_index, b_edge_index_remapped], dim=1)
    merged_edge_attr = torch.cat([
        gdata_a.edge_attr[:, 1:],
        gdata_b.edge_attr[b_keep_mask][:, 1:],
    ], dim=0)
    merged_x = torch.cat([gdata_a.x, gdata_b.x[new_b_indices]], dim=0)
    N_merged = N_A + n_new_b

    merged = Data(x=merged_x, edge_index=merged_edge_index,
                  edge_attr=merged_edge_attr, num_nodes=N_merged)

    seed_pos = {}
    n_a_edges = len(a_gids)
    for gid in seed_gids_set:
        if gid in a_seed_pos:
            seed_pos[gid] = a_seed_pos[gid]
        elif gid in b_seed_pos:
            b_pos = b_seed_pos[gid]
            pos_in_kept_b = int(b_keep_mask[:b_pos].sum().item())
            seed_pos[gid] = n_a_edges + pos_in_kept_b

    return merged, seed_pos


def _first_difference(old, new):
    (g_old, pos_old), (g_new, pos_new) = old, new
    for field in ('x', 'edge_index', 'edge_attr'):
        a, b = getattr(g_old, field), getattr(g_new, field)
        if a.shape != b.shape or not torch.equal(a, b):
            return field
    if g_old.num_nodes != g_new.num_nodes:
        return 'num_nodes'
    if pos_old != pos_new:
        return 'seed_pos'
    return None


def run_merge_equivalence_check(manager, n_batches: int):
    n_pairs, n_two_party, mismatches = 0, 0, {}
    t_old = t_new = 0.0
    n_done = 0
    for i, (batch_key, batch_banks, batch_data) in enumerate(manager._iter_batches('vali', batching=True)):
        if i >= n_batches:
            break
        n_done += 1
        batch_df = manager.ctx['vali'][batch_key]['batch_labels']
        batch_seed_gids = torch.as_tensor(batch_df.index.values.astype('int64'))
        indices = {}
        for seed_list, (bank_a, graph_a), b in _rm_fwd_mod._pair_inputs(batch_df, batch_data):
            graph_b = b[1] if b is not None else None
            gids = [gid for _, gid in seed_list]

            t0 = time.perf_counter()
            old = _build_merged_graph_reference(graph_a, graph_b, set(gids))
            t_old += time.perf_counter() - t0

            t0 = time.perf_counter()
            for bank, graph in ((bank_a, graph_a),) + ((b,) if b is not None else ()):
                if bank not in indices:
                    indices[bank] = _orig_seed_index(graph, batch_seed_gids)
            new = _orig_rm_merge(graph_a, graph_b, gids,
                                 indices[bank_a], indices[b[0]] if b is not None else None)
            t_new += time.perf_counter() - t0

            n_pairs += 1
            n_two_party += graph_b is not None
            diff = _first_difference(old, new)
            if diff is not None:
                mismatches[diff] = mismatches.get(diff, 0) + 1

    passed = n_pairs > 0 and not mismatches
    print(f"\n{'='*74}")
    print(f"  Merge-graph equivalence check — {'PASS' if passed else 'FAIL'} "
          f"(vectorised == reference, exact)")
    print(f"  {n_pairs} pairs over {n_done} vali batches ({n_two_party} with two parties)")
    print(f"{'='*74}")
    if mismatches:
        print(f"  mismatching pairs by first differing field: {mismatches}")
    print(f"  merge time per batch      reference {t_old / max(n_done, 1) * 1000:9.1f} ms   "
          f"vectorised {t_new / max(n_done, 1) * 1000:9.1f} ms   (×{t_old / max(t_new, 1e-9):.1f})")
    print(f"{'='*74}\n", flush=True)


CHUNK_REL_TOL = 1e-4


def run_chunk_size_check(manager, n_batches: int, sizes: list[int]):
    """Per-pair LayerNorm makes PAIR_CHUNK_SIZE a pure speed/memory setting:
    embeddings must match across sizes. Also times train forward+backward per size."""
    gnn = manager.model.gnn
    captured = {}
    hook = gnn.mlp.register_forward_hook(lambda m, inp, out: captured.__setitem__('emb', inp[0].detach()))
    original_size = _rm_fwd_mod.PAIR_CHUNK_SIZE
    ref_size = sizes[0]
    stats = {s: {'rel': [], 'time': [], 'peak': 0.0, 'oom': False} for s in sizes}

    batches = []
    for i, item in enumerate(manager._iter_batches('vali', batching=True)):
        if i >= n_batches:
            break
        batch_key, batch_banks, batch_data = item
        # Keep each batch's labels: the lazy loader overwrites ctx between batches.
        batches.append((batch_key, batch_banks, batch_data,
                        manager.ctx['vali'][batch_key]['batch_labels']))

    for batch_key, batch_banks, batch_data, labels_df in batches:
        manager.ctx['vali'][batch_key]['batch_labels'] = labels_df
        ref_emb = None
        for size in sizes:
            st = stats[size]
            if st['oom']:
                continue
            _rm_fwd_mod.PAIR_CHUNK_SIZE = size
            try:
                gnn.eval()
                with torch.no_grad():
                    manager.forward_pass('vali', batch_key, batch_banks, batch_data)
                emb = captured['emb']
                if size == ref_size:
                    ref_emb = emb
                else:
                    nz = ref_emb.norm(dim=1) > 0
                    st['rel'].append(((emb - ref_emb).norm(dim=1)[nz] / ref_emb.norm(dim=1)[nz]).max().item())

                gnn.train()
                manager.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                core._cuda_sync(); t0 = time.perf_counter()
                preds, labels = manager.forward_pass('vali', batch_key, batch_banks, batch_data)
                manager.loss_fn(preds, labels).backward()
                core._cuda_sync(); st['time'].append(time.perf_counter() - t0)
                if torch.cuda.is_available():
                    st['peak'] = max(st['peak'], torch.cuda.max_memory_allocated() / 1e9)
            except torch.OutOfMemoryError:
                st['oom'] = True
                manager.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        manager.optimizer.zero_grad()   # no optimizer.step(): the model stays unchanged

    hook.remove()
    _rm_fwd_mod.PAIR_CHUNK_SIZE = original_size
    gnn.eval()

    rels = [r for s in sizes[1:] for r in stats[s]['rel']]
    passed = bool(rels) and max(rels) < CHUNK_REL_TOL
    print(f"\n{'='*74}")
    print(f"  Chunk-size check — {'PASS' if passed else 'FAIL'} "
          f"(embeddings identical across sizes, max rel diff < {CHUNK_REL_TOL:g} vs size {ref_size})")
    print(f"  {len(batches)} vali batches; train forward+backward timed per size (no optimizer step)")
    print(f"{'='*74}")
    print(f"  {'chunk size':>10}  {'max rel diff':>13}  {'fwd+bwd / batch':>16}  {'peak GPU':>9}")
    for s in sizes:
        st = stats[s]
        if st['oom'] and not st['time']:
            print(f"  {s:>10}  {'':>13}  {'OOM':>16}  {'':>9}")
            continue
        rel = '(reference)' if s == ref_size else (f"{max(st['rel']):.2e}" if st['rel'] else 'n/a')
        t = f"{sum(st['time']) / len(st['time']):.2f} s" if st['time'] else 'n/a'
        oom = ' (OOM on some batches)' if st['oom'] else ''
        print(f"  {s:>10}  {rel:>13}  {t:>16}  {st['peak']:>7.1f} GB{oom}")
    print(f"{'='*74}\n", flush=True)


def _after_report(manager, algo):
    if _args.check_merge_equivalence > 0:
        run_merge_equivalence_check(manager, _args.check_merge_equivalence)
    if _args.check_chunk_sizes > 0:
        run_chunk_size_check(manager, _args.check_chunk_sizes, _args.chunk_sizes)


if __name__ == '__main__':
    core.run_benchmark(['RestrictedMerge'], after_report=_after_report)
