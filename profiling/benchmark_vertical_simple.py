"""Benchmark simplified vertical FL batching: timing and memory over 1-2 epochs.

Unlike the full vertical benchmark, this variant skips get_batch_intersects,
get_ownership_mappings, and get_nodes_to_send — the simple forward pass does
not perform per-layer embedding exchange, so none of those are called.

Usage (from project root):
    python -m profiling.benchmark_vertical_simple --fl_algo FedGraphSimple --batching \
        --batching_mode lazy_link_neighbor --size small --ir HI --testing

The --testing flag limits data to 5% and runs 2 epochs. Remove it and set
--n_benchmark_epochs to run more epochs on full data.

Reports:
  - Per-batch timing: subgraph build, total
  - Per-epoch timing
  - Peak RSS memory (tracemalloc)
  - Extrapolated full-run estimate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tracemalloc
import copy
import logging
from collections import defaultdict
import numpy as np

import utils
import data.data_functions as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos
import models.gnn_models
from federated_learning.gnn.vertical_simple import batching as simple_bat
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

class SectionTimer:
    """Accumulates elapsed times per named section."""

    def __init__(self):
        self.times = defaultdict(list)

    def section(self, name):
        outer = self
        class _Ctx:
            def __enter__(self):
                self._t = time.perf_counter()
            def __exit__(self, *_):
                outer.times[name].append(time.perf_counter() - self._t)
        return _Ctx()

    def report(self, n_full_batches=None, n_full_epochs=None):
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)
        for name, vals in self.items():
            arr = np.array(vals)
            print(f"  {name:<30s}  n={len(arr):4d}  "
                  f"mean={arr.mean()*1000:7.1f}ms  "
                  f"std={arr.std()*1000:6.1f}ms  "
                  f"total={arr.sum():.2f}s")
        if n_full_batches and 'process_lazy_batch_simple' in self.times:
            per_batch = np.mean(self.times['process_lazy_batch_simple'])
            print(f"\n  Extrapolated full run:")
            print(f"    Batches per epoch (full): {n_full_batches}")
            if n_full_epochs:
                est = per_batch * n_full_batches * n_full_epochs
                print(f"    Epochs: {n_full_epochs}")
                print(f"    Estimated training time: {est/60:.1f} min")
        print("="*60)

    def items(self):
        return self.times.items()


timer = SectionTimer()


# ---------------------------------------------------------------------------
# Monkey-patch key functions with timing wrappers
# ---------------------------------------------------------------------------

_real_process_lazy_batch_simple = simple_bat.process_lazy_batch_simple


def _fully_timed_process_lazy_batch_simple(manager, mode, batch, mode_parties):
    with timer.section('process_lazy_batch_simple'):
        _real_process_lazy_batch_simple(manager, mode, batch, mode_parties)


simple_bat.process_lazy_batch_simple = _fully_timed_process_lazy_batch_simple


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    utils.logger_setup()
    logging.getLogger().setLevel(logging.WARNING)  # suppress info spam

    parsers, df, scaler_encoders = utils.setup_get_data()

    laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(
        parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

    manager = Manager.get_algo_class(parsers)
    manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)

    batching_mode = parsers['data_parser'].batching_mode
    print(f"\nBenchmarking batching_mode='{batching_mode}', "
          f"testing={parsers['data_parser'].testing}, "
          f"max_workers={parsers['fl_parser'].max_workers}")

    # Setup model and vertical structures
    from federated_learning.gnn.vertical_simple import setup

    setup.setup_vertical_simple(manager, batching=True, batching_mode=batching_mode)
    manager.setup_model(ibm_gnn, laundering_values_test)

    n_batches = manager.ctx['train']['num_batches']
    print(f"Batches per epoch (this run): {n_batches}")

    # --- Start memory tracking ---
    tracemalloc.start()

    epochs = 2
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        manager.model.gnn.train()

        if batching_mode == 'lazy_link_neighbor':
            mode_parties = manager.get_parties_for_mode('train')
            for batch in manager.loaders['train']:
                simple_bat.process_lazy_batch_simple(manager, 'train', batch, mode_parties)
                batch_banks = manager.ctx['train'][LAZY_BATCH_KEY]['batch_parties']
                batch_key = LAZY_BATCH_KEY
                batch_data = manager.get_batch_data('train', batch_key, batch_banks)

                manager.optimizer.zero_grad()
                with timer.section('forward_pass'):
                    preds, labels = manager.forward_pass('train', batch_key, batch_banks, batch_data)
                with timer.section('backward_pass'):
                    loss = manager.loss_fn(preds, labels)
                    loss.backward()
                    manager.optimizer.step()

        else:  # simple / link_neighbor / neighbor_sample — batches pre-computed
            for batch_key in range(manager.ctx['train']['num_batches']):
                batch_banks = manager.ctx['train'][batch_key]['batch_parties']
                batch_data = manager.get_batch_data('train', batch_key, batch_banks)

                manager.optimizer.zero_grad()
                with timer.section('forward_pass'):
                    preds, labels = manager.forward_pass('train', batch_key, batch_banks, batch_data)
                with timer.section('backward_pass'):
                    loss = manager.loss_fn(preds, labels)
                    loss.backward()
                    manager.optimizer.step()

        epoch_time = time.perf_counter() - t_epoch
        timer.times['epoch'].append(epoch_time)
        print(f"  Epoch {epoch+1}/{epochs} — {epoch_time:.1f}s  "
              f"({n_batches} batches)")

    # --- Memory snapshot ---
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nMemory: current={current_mem/1e6:.1f} MB  peak={peak_mem/1e6:.1f} MB")

    # Extrapolate to full run
    from configs.configs import epochs as full_epochs
    full_batches_est = n_batches  # already sized to data (testing=5% gives proportional batches)
    timer.report(n_full_batches=full_batches_est, n_full_epochs=full_epochs)

    # Per-batch breakdown summary
    if timer.times:
        total_per_batch = np.mean(timer.times.get('process_lazy_batch_simple', [0]))
        print(f"\n  process_lazy_batch_simple breakdown (approx):")
        print(f"    subgraph build:      {total_per_batch*1000:.1f}ms  (no exchange data)")
        print(f"    total per batch:     {total_per_batch*1000:.1f}ms")


if __name__ == '__main__':
    run_benchmark()
