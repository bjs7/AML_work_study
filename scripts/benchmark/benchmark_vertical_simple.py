"""Benchmark simplified vertical FL batching: timing and memory over N epochs.

Unlike the full vertical benchmark, this variant skips get_batch_intersects,
get_ownership_mappings, and get_nodes_to_send — the simple forward pass does
not perform per-layer embedding exchange, so none of those are called.

Usage (training benchmark):
    python -m scripts.benchmark.benchmark_vertical_simple --fl_algo FedGraphSimple --batching \
        --batching_mode lazy_link_neighbor --size small --ir HI --emlps --ibm_hp \
        --eval_mode comparable --testing

Usage (inference-only with pre-trained weights):
    python -m scripts.benchmark.benchmark_vertical_simple --fl_algo FedGraphSimple --batching \
        --batching_mode lazy_link_neighbor --size small --ir HI --emlps --ibm_hp \
        --eval_mode system --inference_only --load_weights /path/to/seed_1/model.pth

Benchmark-specific flags (not passed to main parsers):
  --n_benchmark_epochs N   Number of training epochs to time (default: 2)
  --inference_only         Skip training; load weights and time test inference only
  --load_weights PATH      Path to model.pth (required with --inference_only)

Reports:
  - Per-batch timing: subgraph build, forward pass, backward pass
  - Per-epoch timing
  - Peak RSS memory (tracemalloc)
  - Extrapolated full-run estimate (training mode only)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import time
import tracemalloc
import copy
import logging
from collections import defaultdict
import numpy as np
import torch

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos
import models.gnn_models
from federated_learning.gnn.vertical_simple import batching as simple_bat
from federated_learning.gnn.vertical.batching import LAZY_BATCH_KEY
from federated_learning.hp_tuning import ibm_gnn


# ---------------------------------------------------------------------------
# Benchmark-specific arguments (parsed separately, not passed to main parsers)
# ---------------------------------------------------------------------------

_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument('--n_benchmark_epochs', type=int, default=2,
                            help='Number of training epochs to time (default: 2)')
_bench_parser.add_argument('--inference_only', action='store_true',
                            help='Skip training; load weights and time test inference only')
_bench_parser.add_argument('--load_weights', type=str, default=None,
                            help='Path to model.pth to load for --inference_only')
_bench_args, _ = _bench_parser.parse_known_args()


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
    logging.getLogger().setLevel(logging.WARNING)

    parsers, df, scaler_encoders = utils.setup_get_data()

    laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(
        parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

    manager = Manager.get_algo_class(parsers)
    manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)

    batching_mode = parsers['data_parser'].batching_mode
    eval_mode = getattr(parsers['data_parser'], 'eval_mode', 'comparable')

    print(f"\nBenchmarking batching_mode='{batching_mode}', eval_mode='{eval_mode}', "
          f"testing={parsers['data_parser'].testing}, "
          f"max_workers={parsers['fl_parser'].max_workers}, "
          f"inference_only={_bench_args.inference_only}")

    from federated_learning.gnn.vertical_simple import setup
    setup.setup_vertical_simple(manager, batching=True, batching_mode=batching_mode)
    manager.setup_model(ibm_gnn, laundering_values_test)

    # -----------------------------------------------------------------------
    # Inference-only mode: load pre-trained weights, time test evaluation
    # -----------------------------------------------------------------------
    if _bench_args.inference_only:
        if _bench_args.load_weights is None:
            raise ValueError("--load_weights <path> is required with --inference_only")

        print(f"\nLoading weights from: {_bench_args.load_weights}")
        state_dict = torch.load(_bench_args.load_weights, map_location=manager.device)
        manager.model.gnn.load_state_dict(state_dict)
        manager.model.gnn.eval()

        n_test_batches = manager.ctx['test']['num_batches']
        print(f"Test batches: {n_test_batches}")

        tracemalloc.start()
        t_start = time.perf_counter()
        with timer.section('test_inference'):
            manager._forward_eval('test', batching=True)
        t_total = time.perf_counter() - t_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nTest inference time: {t_total:.2f}s  ({t_total/60:.1f} min)")
        print(f"Memory: current={current_mem/1e6:.1f} MB  peak={peak_mem/1e6:.1f} MB")
        timer.report()
        return

    # -----------------------------------------------------------------------
    # Training benchmark: time forward + backward over N epochs
    # -----------------------------------------------------------------------
    n_batches = manager.ctx['train']['num_batches']
    epochs = _bench_args.n_benchmark_epochs
    print(f"Batches per epoch: {n_batches}  |  Benchmarking {epochs} epoch(s)")

    tracemalloc.start()

    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        manager.model.gnn.train()

        if batching_mode == 'lazy_link_neighbor':
            mode_parties = manager.get_parties_for_mode('train')
            for batch in manager.loaders['train']:
                simple_bat.process_lazy_batch_simple(manager, 'train', batch, mode_parties)
                batch_banks = manager.ctx['train'][LAZY_BATCH_KEY]['batch_parties']
                batch_data = manager.get_batch_data('train', LAZY_BATCH_KEY, batch_banks)

                manager.optimizer.zero_grad()
                with timer.section('forward_pass'):
                    preds, labels = manager.forward_pass('train', LAZY_BATCH_KEY, batch_banks, batch_data)
                with timer.section('backward_pass'):
                    loss = manager.loss_fn(preds, labels)
                    loss.backward()
                    manager.optimizer.step()

        else:
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
        print(f"  Epoch {epoch+1}/{epochs} — {epoch_time:.1f}s  ({n_batches} batches)")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nMemory: current={current_mem/1e6:.1f} MB  peak={peak_mem/1e6:.1f} MB")

    num_rounds = parsers['fl_parser'].num_rounds
    timer.report(n_full_batches=n_batches, n_full_epochs=num_rounds)


if __name__ == '__main__':
    run_benchmark()
