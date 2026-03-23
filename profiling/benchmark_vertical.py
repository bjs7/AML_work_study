"""Benchmark vertical FL batching: timing and memory over 1-2 epochs.

Usage (from project root):
    python -m profiling.benchmark_vertical --fl_algo FedGraph --batching \
        --batching_mode lazy_link_neighbor --size small --ir HI --testing

The --testing flag limits data to 5% and runs 2 epochs. Remove it and set
--n_benchmark_epochs to run more epochs on full data.

Reports:
  - Per-batch timing: subgraph build, intersects, ownership, nodes_to_send, total
  - Per-epoch timing
  - Peak RSS memory (tracemalloc)
  - Extrapolated full-run estimate
"""

import sys
import time
import tracemalloc
import copy
import logging
from collections import defaultdict
import numpy as np
import pandas as pd

import utils
import data.data_functions as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos
import models.gnn_models
from data.raw_data_processing import get_data
from configs.configs import split_perc
from configs.paths import get_data_path
from federated_learning.gnn.vertical import batching as bat
from federated_learning.gnn.vertical import ownership as own
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY
from training.utils import ibm_gnn


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
        if n_full_batches and 'process_lazy_batch' in self.times:
            per_batch = np.mean(self.times['process_lazy_batch'])
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

_orig_get_batch_intersects = bat.get_batch_intersects
_orig_get_ownership = own.get_ownership_mappings
_orig_get_nodes = own.get_nodes_to_send


def _timed_get_batch_intersects(*args, **kwargs):
    with timer.section('get_batch_intersects'):
        return _orig_get_batch_intersects(*args, **kwargs)


def _timed_get_ownership(*args, **kwargs):
    with timer.section('get_ownership_mappings'):
        return _orig_get_ownership(*args, **kwargs)


def _timed_get_nodes(*args, **kwargs):
    with timer.section('get_nodes_to_send'):
        return _orig_get_nodes(*args, **kwargs)


bat.get_batch_intersects = _timed_get_batch_intersects
own.get_ownership_mappings = _timed_get_ownership
own.get_nodes_to_send = _timed_get_nodes

# Also patch the references imported inside batching.py
bat.get_ownership_mappings = _timed_get_ownership
bat.get_nodes_to_send = _timed_get_nodes


_real_process_lazy_batch = bat.process_lazy_batch


def _fully_timed_process_lazy_batch(manager, mode, batch, mode_parties):
    with timer.section('process_lazy_batch'):
        _real_process_lazy_batch(manager, mode, batch, mode_parties)


bat.process_lazy_batch = _fully_timed_process_lazy_batch


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
    from federated_learning.gnn.vertical import setup

    batching_mode = parsers['data_parser'].batching_mode
    setup.setup_vertical(manager, batching=True, batching_mode=batching_mode)
    manager.setup_model(ibm_gnn, laundering_values_test)

    n_batches = manager.ctx['train']['num_batches']
    print(f"Batches per epoch (this run): {n_batches}")

    # --- Start memory tracking ---
    tracemalloc.start()

    epochs = 2
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        manager.model.gnn.train()

        mode_parties = manager.get_parties_for_mode('train')
        for batch in manager.loaders['train']:
            bat.process_lazy_batch(manager, 'train', batch, mode_parties)

            batch_banks = manager.ctx['train'][LAZY_BATCH_KEY]['batch_parties']
            batch_data = manager.get_batch_data('train', LAZY_BATCH_KEY, batch_banks)

            manager.optimizer.zero_grad()
            with timer.section('forward_pass'):
                preds, labels = manager.forward_pass('train', LAZY_BATCH_KEY, batch_banks, batch_data)

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
        total_per_batch = np.mean(timer.times.get('process_lazy_batch', [0]))
        sub_total = sum(
            np.mean(timer.times.get(k, [0]))
            for k in ['get_batch_intersects', 'get_ownership_mappings', 'get_nodes_to_send']
        )
        print(f"\n  process_lazy_batch breakdown (approx):")
        print(f"    subgraph build:      {(total_per_batch - sub_total)*1000:.1f}ms")
        for k in ['get_batch_intersects', 'get_ownership_mappings', 'get_nodes_to_send']:
            v = np.mean(timer.times.get(k, [0]))
            print(f"    {k:<30s} {v*1000:.1f}ms")
        print(f"    total per batch:     {total_per_batch*1000:.1f}ms")


if __name__ == '__main__':
    run_benchmark()
