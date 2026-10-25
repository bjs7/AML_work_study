"""Timing benchmark — FedGraph vs FedAvgSplit.

Instruments the training loop at the sub-step level so you can see exactly
where time is spent and whether an explicit multi-GPU implementation (where
each party owns its own computation graph) would be faster.

Key question answered
---------------------
  In the current single-GPU implementation PyTorch handles the joint backward
  pass automatically.  In an explicit multi-GPU implementation each party
  would detach its embedding, send it to the manager, receive gradients back,
  and call emb.backward(received_grad) locally — in parallel across GPUs.

  The script reports:
    • How much forward time is party-GNN vs exchange vs manager-head.
    • A separate explicit-backward mini-benchmark that times the two phases
      of the manual path (manager-only backward, per-party GNN backward) so
      the parallel speedup can be estimated.

Sections timed (main loop)
--------------------------
  batch_setup    subgraph build / intersects / ownership
  party_gnn      emed_features + apply_gnn_layer × L + prep_nodes_edges,
                 summed over all parties (sequential today → parallel on N GPUs)
  exchange       embedding send/receive (FedGraph only; derived:
                 forward_total − party_gnn − manager_head)
  manager_head   mlp() / mlp_vert() (always sequential)
  forward_total  full forward_pass call
  backward       loss.backward() + optimizer.step() (full joint graph today)
  validation     full vali _forward_eval per epoch
  epoch_total    full epoch wall time

Explicit-backward mini-benchmark (appended after main epochs)
--------------------------------------------------------------
  manager_bwd_only   loss.backward() with detached party embeddings
                     — only propagates through mlp_vert
  party_gnn_bwd      emb.backward(received_grad) for one party's GNN
  These two phases map to the multi-GPU protocol and can run in parallel
  for all N parties in the second phase.

Usage
-----
  # Single algorithm:
  python scripts/hpc/benchmark/benchmark_timing_comparison.py \\
      --fl_algo FedGraph --size small --ir HI --ibm_hp --emlps \\
      --batching --batching_mode lazy_link_neighbor --eval_mode system \\
      --n_benchmark_epochs 5

  # Both algorithms in one run (FedGraph + SplitFed):
  python scripts/hpc/benchmark/benchmark_timing_comparison.py \\
      --fl_algo all --size small --ir HI --ibm_hp --emlps \\
      --batching --batching_mode lazy_link_neighbor --eval_mode system \\
      --n_benchmark_epochs 5

This file is also the shared core (timer, timing hooks, epoch loop, report,
run_benchmark) imported by the per-algorithm entry scripts:
  benchmark_timing_splitfed.py          + batched-vs-sequential equivalence check
  benchmark_timing_restricted_merge.py  + merge-graph-build timing

Benchmark-only flags (not forwarded to main parsers):
  --n_benchmark_epochs N    training epochs to time  (default: 5)
  --n_benchmark_batches N   time only N train + N vali batches per epoch and
                            extrapolate full-epoch times (default: 0 = full epochs)
  --fl_algo A [B ...]       one or more algorithms, or "all" (FedGraph + SplitFed)
  --n_explicit_batches N    batches for explicit-backward mini-benchmark
                            (default: 20; set 0 to skip; SplitFed only)
"""

import sys
import os

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    # repo root: scripts/hpc/benchmark/gnn/<this file> -> 4 levels up
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import argparse
import time
import copy
import logging
from collections import defaultdict

import numpy as np
import torch

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos   # noqa: F401
import models.gnn_models              # noqa: F401
from federated_learning.hp_tuning import ibm_gnn
from federated_learning.gnn.fedgraph.batching import LAZY_BATCH_KEY


# ---------------------------------------------------------------------------
# Benchmark-only arguments
# ---------------------------------------------------------------------------

_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument('--n_benchmark_epochs',  type=int, default=5)
_bench_parser.add_argument('--n_benchmark_batches', type=int, default=0,
                            help='Time only this many train (and vali) batches per epoch; '
                                 '0 = full epochs. Full-epoch times are extrapolated.')
_bench_parser.add_argument('--n_explicit_batches',  type=int, default=20)
_bench_parser.add_argument('--fl_algo', nargs='+', default=['SplitFed'],
                            help='One or more of FedGraph, SplitFed, FedAvgSplit; or "all" '
                                 '(FedGraph + SplitFed). For SplitFed / RestrictedMerge breakdowns use '
                                 'benchmark_timing_splitfed.py / benchmark_timing_restricted_merge.py')
_bench_args, _ = _bench_parser.parse_known_args()


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class SectionTimer:
    """Sections synchronise CUDA on entry/exit so GPU work is attributed to the
    section that launched it (otherwise only kernel-launch time is measured)."""

    def __init__(self):
        self.times: dict[str, list[float]] = defaultdict(list)
        self.n_batches = {'train': 0, 'vali': 0}

    def section(self, name: str):
        outer = self
        class _Ctx:
            def __enter__(self_):
                _cuda_sync()
                self_._t = time.perf_counter()
            def __exit__(self_, *_):
                _cuda_sync()
                outer.times[name].append(time.perf_counter() - self_._t)
        return _Ctx()

    def mean_ms(self, name: str) -> float:
        vals = self.times.get(name, [])
        return float(np.mean(vals)) * 1000 if vals else float('nan')

    def per_batch_ms(self, name: str, split: str = 'train') -> float:
        """Total time in a section divided by batches — sections may fire many times per batch."""
        n = self.n_batches[split]
        if not n or name not in self.times:   # section never ran (e.g. merge build for SplitFed)
            return float('nan')
        return float(np.sum(self.times[name])) * 1000 / n

    def n(self, name: str) -> int:
        return len(self.times.get(name, []))


# ---------------------------------------------------------------------------
# Monkey-patches
# A module-level pointer _active_timer selects the currently-recording timer.
# ---------------------------------------------------------------------------

_active_timer: SectionTimer | None = None
# 'bwd' while loss.backward() runs, so GNN calls made by checkpoint recomputation
# (RestrictedMerge) are recorded separately from the forward GNN pass.
_phase: str = 'fwd'


from models.gnn_models import GINe
import federated_learning.gnn.fedgraph.forward   as _fg_fwd_mod
import federated_learning.gnn.splitfed.forward   as _fs_fwd_mod
import federated_learning.gnn.fedgraph.batching  as _fg_bat_mod
import federated_learning.gnn.splitfed.batching  as _fs_bat_mod
import federated_learning.gnn.federated_manager  as _mgr_mod

_orig_emed     = GINe.emed_features
_orig_apply    = GINe.apply_gnn_layer
_orig_prep     = GINe.prep_nodes_edges
_orig_fg_fwd   = _fg_fwd_mod.forward_pass
_orig_fs_fwd   = _fs_fwd_mod.forward_pass_splitfed
_orig_fg_bat   = _fg_bat_mod.process_lazy_batch
_orig_fs_bat   = _fs_bat_mod.process_lazy_batch_splitfed


def _gnn_section():
    return 'party_gnn_recompute' if _phase == 'bwd' else 'party_gnn'

def _t_emed(self, *a, **kw):
    t = _active_timer
    if t is None: return _orig_emed(self, *a, **kw)
    with t.section(_gnn_section()): return _orig_emed(self, *a, **kw)

def _t_apply(self, *a, **kw):
    t = _active_timer
    if t is None: return _orig_apply(self, *a, **kw)
    with t.section(_gnn_section()): return _orig_apply(self, *a, **kw)

def _t_prep(self, *a, **kw):
    t = _active_timer
    if t is None: return _orig_prep(self, *a, **kw)
    with t.section(_gnn_section()): return _orig_prep(self, *a, **kw)

def _t_fg_fwd(*a, **kw):
    t = _active_timer
    if t is None: return _orig_fg_fwd(*a, **kw)
    with t.section('forward_total'):
        return _orig_fg_fwd(*a, **kw)

def _t_fs_fwd(*a, **kw):
    t = _active_timer
    if t is None: return _orig_fs_fwd(*a, **kw)
    with t.section('forward_total'):
        return _orig_fs_fwd(*a, **kw)

def _t_fg_bat(manager, mode, batch, mode_parties):
    t = _active_timer
    if t is None: _orig_fg_bat(manager, mode, batch, mode_parties); return
    with t.section('batch_setup'): _orig_fg_bat(manager, mode, batch, mode_parties)

def _t_fs_bat(manager, mode, batch, mode_parties):
    t = _active_timer
    if t is None: _orig_fs_bat(manager, mode, batch, mode_parties); return
    with t.section('batch_setup'): _orig_fs_bat(manager, mode, batch, mode_parties)


GINe.emed_features    = _t_emed
GINe.apply_gnn_layer  = _t_apply
GINe.prep_nodes_edges = _t_prep

_fg_fwd_mod.forward_pass              = _t_fg_fwd
_fs_fwd_mod.forward_pass_splitfed       = _t_fs_fwd
_fg_bat_mod.process_lazy_batch        = _t_fg_bat
_fs_bat_mod.process_lazy_batch_splitfed = _t_fs_bat
# Patch names referenced inside _iter_batches in federated_manager's namespace
_mgr_mod.process_lazy_batch        = _t_fg_bat
_mgr_mod.process_lazy_batch_splitfed = _t_fs_bat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_mlp_hooks(gnn) -> list:
    """Register forward hooks on gnn.mlp and gnn.mlp_vert to time manager_head.

    mlp/mlp_vert are nn.Sequential instance attributes (not class methods), so
    they cannot be monkey-patched at the class level — hooks are the correct approach.
    Returns the hook handles so the caller can remove them if needed.
    """
    _t_starts: dict[int, float] = {}

    def _pre(module, _input):
        if _active_timer is not None:
            _cuda_sync()
            _t_starts[id(module)] = time.perf_counter()

    def _post(module, _input, _output):
        t = _active_timer
        if t is not None:
            start = _t_starts.pop(id(module), None)
            if start is not None:
                _cuda_sync()
                t.times['manager_head'].append(time.perf_counter() - start)

    handles = []
    for attr in ('mlp', 'mlp_vert'):
        m = getattr(gnn, attr, None)
        if m is not None:
            handles.append(m.register_forward_pre_hook(_pre))
            handles.append(m.register_forward_hook(_post))
    return handles


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main epoch loop
# ---------------------------------------------------------------------------

def _timed_batches(manager, mode, timer, name, max_batches):
    """Yield batches from _iter_batches, timing each fetch (loader sampling + subgraph build)."""
    it = manager._iter_batches(mode, batching=True)
    n = 0
    while not max_batches or n < max_batches:
        with timer.section(name):
            item = next(it, None)
        if item is None:
            timer.times[name].pop()   # the exhausted fetch is not a batch
            return
        n += 1
        yield item


def run_epochs(manager, timer: SectionTimer, n_epochs: int, max_batches: int = 0) -> int:
    """Run N training epochs with timing (optionally capped at max_batches
    train and vali batches each). Returns the timed train batches per epoch."""
    global _active_timer, _phase
    _active_timer = timer
    _register_mlp_hooks(manager.model.gnn)
    batches_per_epoch = 0

    for epoch in range(n_epochs):
        _cuda_sync(); t_epoch = time.perf_counter()
        manager.model.gnn.train()
        n_this = 0

        for batch_key, batch_banks, batch_data in _timed_batches(
                manager, 'train', timer, 'batch_fetch', max_batches):
            manager.optimizer.zero_grad()
            _phase = 'fwd'
            preds, labels = manager.forward_pass('train', batch_key, batch_banks, batch_data)
            loss = manager.loss_fn(preds, labels)
            _phase = 'bwd'
            with timer.section('backward'):
                loss.backward()
                manager.optimizer.step()
            _phase = 'fwd'
            n_this += 1
        timer.n_batches['train'] += n_this

        # Validation: total time only (no inner sections), per batch
        _active_timer = None
        manager.model.gnn.eval()
        _cuda_sync(); t_vali = time.perf_counter()
        n_vali = 0
        with torch.no_grad():
            for batch_key, batch_banks, batch_data in _timed_batches(
                    manager, 'vali', timer, 'vali_fetch', max_batches):
                manager.forward_pass('vali', batch_key, batch_banks, batch_data)
                n_vali += 1
        _cuda_sync(); timer.times['validation'].append(time.perf_counter() - t_vali)
        timer.n_batches['vali'] += n_vali
        _active_timer = timer

        _cuda_sync(); timer.times['epoch_total'].append(time.perf_counter() - t_epoch)
        batches_per_epoch = n_this
        print(f"  Epoch {epoch+1}/{n_epochs} — "
              f"{timer.times['epoch_total'][-1]:.1f}s  ({n_this} train, {n_vali} vali batches)",
              flush=True)

    _active_timer = None
    return batches_per_epoch


# ---------------------------------------------------------------------------
# Explicit-backward mini-benchmark (FedAvgSplit only)
#
# Simulates the multi-GPU protocol:
#   Phase 1: party GNN forward (detached) → proxy embeddings to manager
#   Phase 2: manager forward + loss.backward() → get grad w.r.t. proxies
#            (only propagates through mlp_vert — no party GNN in graph)
#   Phase 3: per-party emb.backward(received_grad)
#            (propagates through each party's own GNN independently)
#
# Phase 2 is always sequential.  Phase 3 can run in parallel across N GPUs.
# ---------------------------------------------------------------------------

def run_explicit_backward_benchmark(manager, n_batches: int) -> dict:
    """Time manager-only backward and per-party GNN backward separately.

    Returns dict with mean times in ms:
      manager_bwd_ms   — Phase 2: loss.backward() through mlp_vert only
      party_gnn_bwd_ms — Phase 3: emb.backward(grad) through one party's GNN
    """
    from federated_learning.gnn.splitfed.batching import process_lazy_batch_splitfed

    manager.model.gnn.train()
    manager_bwd_times   = []
    party_gnn_bwd_times = []
    batch_count = 0

    mode_parties = manager.get_parties_for_mode('train')

    for raw_batch in manager.loaders['train']:
        if batch_count >= n_batches:
            break

        # Reconstruct batch context (without timing patches active)
        _orig_fs_bat(manager, 'train', raw_batch, mode_parties)
        batch_banks = manager.ctx['train'][LAZY_BATCH_KEY]['batch_parties']
        batch_data  = {bank_id: (mode_parties[bank_id],
                                  mode_parties[bank_id].ctx['train'][LAZY_BATCH_KEY]['graph_data'])
                       for bank_id in batch_banks}

        device = manager.device

        # ── Phase 1: party GNN forward (detached) ──────────────────────────
        embedding_tensors = {}   # bank_id → full per-edge embedding [n_edges, D]
        index_to_position = {}   # bank_id → {global_id: local_pos}

        for bank_id in batch_banks:
            party, party_data = batch_data[bank_id]
            party_data.x          = party_data.x.to(device)
            party_data.edge_attr  = party_data.edge_attr.to(device)
            party_data.edge_index = party_data.edge_index.to(device)

            emb = _orig_emed(party.model.gnn, party_data.x, party_data.edge_attr[:, 1:])
            for layer_idx in range(party.model.gnn.num_gnn_layers):
                emb = _orig_apply(party.model.gnn,
                                  emb['nodes'], emb['edges'], party_data.edge_index, layer_idx)
            full_emb = _orig_prep(party.model.gnn,
                                  emb['nodes'], emb['edges'], party_data.edge_index)

            embedding_tensors[bank_id] = full_emb          # still in graph (not detached)
            index_to_position[bank_id] = {
                int(gid): pos for pos, gid in enumerate(party_data.edge_attr[:, 0].cpu())
            }

        # Build proxy tensors: detached slices sent from parties → manager
        import numpy as np
        batch_df    = manager.ctx['train'][LAZY_BATCH_KEY]['batch_labels']
        from_banks  = batch_df['From Bank'].values.astype(int)
        to_banks    = batch_df['To Bank'].values.astype(int)
        true_y      = torch.tensor(batch_df['Is Laundering'].values,
                                   dtype=torch.long, device=device)
        indices     = batch_df.index.values
        n_samples   = len(batch_df)
        embed_dim   = next(iter(embedding_tensors.values())).shape[1]

        from_emb_det = torch.zeros(n_samples, embed_dim, device=device)
        to_emb_det   = torch.zeros(n_samples, embed_dim, device=device)

        for bank in np.unique(from_banks):
            if bank not in index_to_position: continue
            mask = from_banks == bank
            pos  = [index_to_position[bank][idx] for idx in indices[mask]]
            from_emb_det[mask] = embedding_tensors[bank][pos].detach()

        for bank in np.unique(to_banks):
            if bank not in index_to_position: continue
            mask = to_banks == bank
            pos  = [index_to_position[bank][idx] for idx in indices[mask]]
            to_emb_det[mask] = embedding_tensors[bank][pos].detach()

        intra = from_banks == to_banks
        if intra.any():
            to_emb_det[intra] = 0

        from_proxy = from_emb_det.requires_grad_(True)
        to_proxy   = to_emb_det.requires_grad_(True)

        # ── Phase 2: manager forward + backward (mlp_vert only) ────────────
        manager.optimizer.zero_grad()
        combined = torch.cat([from_proxy, to_proxy], dim=1)
        preds    = _orig_mlp_vert(manager.model.gnn, combined)
        loss     = manager.loss_fn(preds, true_y)

        _cuda_sync(); t0 = time.perf_counter()
        loss.backward()   # only propagates through mlp_vert
        _cuda_sync(); manager_bwd_times.append(time.perf_counter() - t0)

        # ── Phase 3: per-party GNN backward ────────────────────────────────
        # Use from_proxy.grad (gradient w.r.t. the from-bank embedding slice)
        # and backpropagate through the first party's full GNN embedding tensor.
        first_bank = batch_banks[0]
        full_emb_A = embedding_tensors[first_bank]
        mask_A     = from_banks == first_bank
        positions_A = [index_to_position[first_bank][idx]
                       for idx in indices[mask_A] if idx in index_to_position[first_bank]]
        if positions_A and from_proxy.grad is not None:
            grad_full = torch.zeros_like(full_emb_A)
            for i, pos in enumerate(positions_A):
                grad_full[pos] += from_proxy.grad[mask_A][i]

            _cuda_sync(); t0 = time.perf_counter()
            full_emb_A.backward(grad_full)
            _cuda_sync(); party_gnn_bwd_times.append(time.perf_counter() - t0)

        batch_count += 1

    return {
        'manager_bwd_ms':   float(np.mean(manager_bwd_times))   * 1000 if manager_bwd_times   else float('nan'),
        'party_gnn_bwd_ms': float(np.mean(party_gnn_bwd_times)) * 1000 if party_gnn_bwd_times else float('nan'),
        'n_batches':        batch_count,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(timer: SectionTimer, algo: str, full_train_batches: int, full_vali_batches: int,
           n_epochs: int, num_rounds: int,
           explicit_bwd: dict | None = None, peak_gpu_gb: float = float('nan')):

    pb = timer.per_batch_ms
    fetch_ms = pb('batch_fetch')
    batch_ms = pb('batch_setup')
    sample_ms = max(0.0, fetch_ms - batch_ms)
    fwd_ms   = pb('forward_total')
    party_ms = pb('party_gnn')
    merge_ms = pb('rm_merge_build')
    head_ms  = pb('manager_head')
    other_ms = max(0.0, fwd_ms - party_ms - head_ms - (0 if np.isnan(merge_ms) else merge_ms))
    bwd_ms   = pb('backward')
    recomp_ms = pb('party_gnn_recompute')
    n_vali   = timer.n_batches['vali']
    vali_batch_ms = float(np.sum(timer.times['validation'])) * 1000 / n_vali if n_vali else float('nan')

    per_batch_total = fetch_ms + fwd_ms + bwd_ms
    train_epoch_s = per_batch_total * full_train_batches / 1000
    vali_epoch_s  = vali_batch_ms * full_vali_batches / 1000
    epoch_ms = (train_epoch_s + vali_epoch_s) * 1000
    vali_ms  = vali_epoch_s * 1000

    def pct(x):
        return 100 * x / per_batch_total if per_batch_total > 0 else 0

    def row(label, ms, show_pct=True, indent=''):
        if np.isnan(ms):
            return
        p = f"  {pct(ms):5.1f}%" if show_pct else ' ' * 9
        print(f"  {indent}{label:<40s}  {ms:9.1f} ms{p}")

    print(f"\n{'='*74}")
    print(f"  {algo}  |  {timer.n_batches['train']} train + {n_vali} vali batches timed  "
          f"|  peak GPU {peak_gpu_gb:.1f} GB")
    print(f"{'='*74}")

    print(f"\n  Per train batch  (% of fetch + forward + backward):")
    row("batch fetch", fetch_ms)
    row("↳ loader neighbor sampling (derived)", sample_ms, indent='  ')
    row("↳ party subgraph build", batch_ms, indent='  ')
    row("forward", fwd_ms)
    row("↳ GNN", party_ms, indent='  ')
    row("↳ merge-graph build (CPU)", merge_ms, indent='  ')
    row("↳ head (mlp / mlp_vert)", head_ms, indent='  ')
    label = "↳ embedding exchange (derived)" if algo == 'FedGraph' else "↳ other: assembly, batching, transfer"
    row(label, other_ms, indent='  ')
    row("backward + optimizer step", bwd_ms)
    row("↳ GNN recompute (checkpointing)", recomp_ms, indent='  ')
    row("total", per_batch_total)
    print(f"\n  Per vali batch (fetch + forward):  {vali_batch_ms:9.1f} ms")

    print(f"\n  Full epoch, extrapolated  ({full_train_batches} train, {full_vali_batches} vali batches):")
    print(f"  {'training':<42s}  {train_epoch_s/60:8.1f} min")
    print(f"  {'validation':<42s}  {vali_epoch_s/60:8.1f} min")
    print(f"  {'epoch total':<42s}  {epoch_ms/60000:8.1f} min")
    est_h = epoch_ms / 1000 * num_rounds / 3600
    print(f"  Extrapolated full run ({num_rounds} rounds):  {est_h:.2f} h")

    # ── Explicit-backward breakdown ───────────────────────────────────────────
    if explicit_bwd and not any(np.isnan(v) for v in explicit_bwd.values() if isinstance(v, float)):
        mgr_bwd  = explicit_bwd['manager_bwd_ms']
        pty_bwd  = explicit_bwd['party_gnn_bwd_ms']
        n_eb     = explicit_bwd['n_batches']

        print(f"\n  Explicit-backward breakdown  (n={n_eb} batches):")
        print(f"  (simulates multi-GPU protocol: parties detach, manager gets grad, "
              f"parties backprop locally)")
        print(f"  {'manager-only backward  (mlp_vert)':<42s}  {mgr_bwd:8.1f} ms")
        print(f"  {'per-party GNN backward  (1 party)':<42s}  {pty_bwd:8.1f} ms")
        print(f"  {'joint backward  (current, all parties)':<42s}  {bwd_ms:8.1f} ms")

        print(f"\n  Forward + backward parallelism estimate  "
              f"(if N parties each on their own GPU):")
        print(f"  {'N':>3}   fwd speedup   bwd speedup   epoch est (s)   full run est (h)")
        print(f"  {'-'*66}")
        n_parties_fwd = max(1, round(party_ms / (party_ms / max(timer.n('party_gnn') //
                                                                  max(timer.n('forward_total'), 1), 1))))
        for n in [2, 4, 8]:
            # Forward: party_gnn runs in parallel → 1/N of sequential time
            new_fwd   = fwd_ms - party_ms + party_ms / n
            # Backward: manager_bwd sequential + party_gnn_bwd in parallel
            # Current joint backward ≈ mgr_bwd + N_parties × pty_bwd (estimated)
            n_pty_est = max(2, round((bwd_ms - mgr_bwd) / max(pty_bwd, 1e-6)))
            new_bwd   = mgr_bwd + pty_bwd  # manager seq + 1 party parallel
            fwd_su    = (batch_ms + fwd_ms + bwd_ms) / (batch_ms + new_fwd + new_bwd)
            new_epoch = epoch_ms / 1000 * (batch_ms + new_fwd + new_bwd) / per_batch_total
            new_epoch_vali = new_epoch + (vali_ms / 1000) * (1 - (batch_ms + new_fwd + new_bwd) / per_batch_total)
            new_h     = new_epoch * num_rounds / 3600
            fwd_sp    = fwd_ms / new_fwd
            bwd_sp    = bwd_ms / new_bwd
            print(f"  {n:>3}   {fwd_sp:9.2f}×   {bwd_sp:9.2f}×   {new_epoch:11.1f}   {new_h:14.2f}")

    # ── Simple parallelism estimate (forward only, when no explicit bwd data) ─
    elif party_ms > 0:
        print(f"\n  Parallelism estimate  (forward only; "
              f"party_gnn = {pct(party_ms):.1f}% of per-batch total):")
        print(f"  {'N':>3}   per-batch speedup   epoch est (s)   full run est (h)")
        print(f"  {'-'*56}")
        for n in [2, 4, 8]:
            new_fwd   = fwd_ms - party_ms + party_ms / n
            new_bwd   = bwd_ms  # unchanged (no backward breakdown)
            sp        = per_batch_total / (batch_ms + new_fwd + new_bwd)
            new_epoch = epoch_ms / 1000 / sp
            new_h     = new_epoch * num_rounds / 3600
            print(f"  {n:>3}   {sp:14.2f}×   {new_epoch:11.1f}   {new_h:14.2f}")

    print(f"{'='*74}\n")


# ---------------------------------------------------------------------------
# Setup one manager
# ---------------------------------------------------------------------------

def setup_manager(parsers, df, scaler_encoders, laundering_values_vali,
                  laundering_values_test, algo: str):
    p = copy.deepcopy(parsers)
    p['fl_parser'].fl_algo = algo
    utils.set_seed(1)
    manager = Manager.get_algo_class(p)
    # Party feature engineering modifies df/scalers in place, so each algorithm gets its own copy.
    manager.setup_parties(copy.deepcopy(df), p, copy.deepcopy(scaler_encoders),
                          copy.deepcopy(laundering_values_vali))
    batching_mode = p['data_parser'].batching_mode
    manager.setup(batching=True, batching_mode=batching_mode)
    manager.setup_model(ibm_gnn, laundering_values_test)
    return manager, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(algos, after_report=None):
    """Load data once, then time each algorithm. after_report(manager, algo) runs
    algorithm-specific extras (e.g. the SplitFed equivalence check) after the report."""
    utils.logger_setup()
    logging.getLogger().setLevel(logging.WARNING)

    parsers, df, scaler_encoders = utils.setup_get_data()
    num_rounds    = parsers['fl_parser'].num_rounds
    n_epochs      = _bench_args.n_benchmark_epochs
    n_expl        = _bench_args.n_explicit_batches

    lv_vali, lv_test = dfn.prep_laundering_dfs(
        parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

    max_batches = _bench_args.n_benchmark_batches

    for algo in algos:
        print(f"\n{'#'*74}")
        print(f"#  {algo}")
        print(f"{'#'*74}", flush=True)

        t_setup = time.perf_counter()
        manager, algo_parsers = setup_manager(
            parsers, df, scaler_encoders, lv_vali, lv_test, algo)

        print(f"  device={manager.device}, "
              f"batching_mode={algo_parsers['data_parser'].batching_mode}, "
              f"eval_mode={algo_parsers['data_parser'].eval_mode}, "
              f"setup {time.perf_counter() - t_setup:.0f}s", flush=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # No tracemalloc here: it records every Python allocation and inflated the
        # Python-heavy sections (merge build, assembly) several-fold.
        timer = SectionTimer()
        run_epochs(manager, timer, n_epochs, max_batches)
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float('nan')

        # Explicit-backward mini-benchmark (FedAvgSplit only — clean split boundary)
        explicit_bwd = None
        if algo == 'FedAvgSplit' and n_expl > 0:
            print(f"\n  Running explicit-backward mini-benchmark "
                  f"({n_expl} batches) …")
            explicit_bwd = run_explicit_backward_benchmark(manager, n_expl)

        report(timer, algo, manager.ctx['train']['num_batches'], manager.ctx['vali']['num_batches'],
               n_epochs, num_rounds, explicit_bwd, peak_gpu_gb)

        if after_report is not None:
            after_report(manager, algo)

        del manager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    algos = ['FedGraph', 'SplitFed'] if _bench_args.fl_algo == ['all'] else _bench_args.fl_algo
    run_benchmark(algos)


if __name__ == '__main__':
    main()
