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
_args, _ = _parser.parse_known_args()

EQUIV_REL_TOL = 1e-4


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
    if _args.check_batched_equivalence > 0:
        run_batched_equivalence_check(manager, _args.check_batched_equivalence, _args.load_weights)


if __name__ == '__main__':
    core.run_benchmark(['SplitFed'], after_report=_after_report)
