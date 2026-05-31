#!/usr/bin/env python3
"""Inter-stage aggregation for parallel GNN tuning.

Two modes:
  --stage N   Read stage N results, pick top 5, narrow search intervals,
              generate stage N+1 configs. Run after stage N array completes.
  --final     Read stage 2 results, pick the best HP, save it to the tuned
              HP file (configs/tuned_hyperparams/gnn/...). Run after stage 2.

Usage (called by run_aggregate.sh):
  python aggregate_stage.py --stage 1 --fl_algo full_info --model GINe --size small ...
  python aggregate_stage.py --final  --fl_algo full_info --model GINe --size small ...
"""
import argparse
import glob
import json
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.normpath(os.path.join(_here, '..', '..', '..')),             # local: scripts/hpc/gnn_tuning/
    os.path.normpath(os.path.join(_here, '..', '..', 'AML_work_study')), # HPC:   batch_jobs/gnn_tuning/
]:
    if os.path.exists(os.path.join(_candidate, 'utils.py')):
        sys.path.insert(0, _candidate)
        break

import utils
from federated_learning.hp_tuning import hyper_sampler
from federated_learning.gnn.manager_mixin import GNNMixinManager
from configs.paths import get_tuning_configs, get_full_info_hp_path, get_data_path


def get_work_dir(parsers):
    size = parsers['data_parser'].size
    ir = parsers['data_parser'].ir
    eval_mode = getattr(parsers['data_parser'], 'eval_mode', 'system')
    if get_data_path() == '/data/leuven/362/vsc36278':
        base = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    return os.path.join(base, 'configs', 'gnn_tuning_work', f'{size}_{ir}_{eval_mode}')


def load_stage_results(work_dir, stage):
    pattern = os.path.join(work_dir, f'stage{stage}_results', 'hp_*.json')
    results = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            results.append(json.load(f))
    return results


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--stage', type=int, choices=[1, 2], default=None)
    pre_parser.add_argument('--final', action='store_true')
    our_args, remaining = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    parsers = utils.parser_all()
    work_dir = get_work_dir(parsers)

    if our_args.final:
        results = load_stage_results(work_dir, 2)
        if not results:
            raise RuntimeError(f"No stage 2 results found in {work_dir}/stage2_results/")

        best = max(results, key=lambda r: r['f1'])
        hp = best['hyperparameters']

        hp_path = get_full_info_hp_path(parsers)
        os.makedirs(os.path.dirname(hp_path), exist_ok=True)
        with open(hp_path, 'w') as f:
            json.dump(hp, f, indent=4)
        print(f"Best HP (F1={best['f1']:.4f}, idx={best['hp_idx']}) saved to {hp_path}")

    else:
        stage = our_args.stage
        results = load_stage_results(work_dir, stage)

        # Load config file to verify we got all results
        configs_path = os.path.join(work_dir, f'stage{stage}_configs.json')
        with open(configs_path) as f:
            expected_n = len(json.load(f))
        if len(results) < expected_n:
            raise RuntimeError(
                f"Expected {expected_n} stage {stage} results, found {len(results)}. "
                "Check if any array tasks failed before aggregating."
            )

        results_sorted = sorted(results, key=lambda r: r['f1'], reverse=True)
        top5 = results_sorted[:5]

        print(f"Stage {stage} top 5:")
        for r in top5:
            print(f"  idx={r['hp_idx']:3d}  F1={r['f1']:.4f}  {r['hyperparameters']}")

        top5_hps = [r['hyperparameters'] for r in top5]
        sample_space = GNNMixinManager._get_search_space(top5_hps)
        print(f"Narrowed search space: {sample_space}")

        random.seed(parsers['data_parser'].seed)
        tuning_configs = get_tuning_configs(parsers)
        x_0 = tuning_configs[parsers['data_parser'].scenario][parsers['data_parser'].size]['x_0']
        hp_list = [hyper_sampler(parsers['fl_parser'], None, sample_space) for _ in range(x_0)]

        out_path = os.path.join(work_dir, f'stage{stage + 1}_configs.json')
        with open(out_path, 'w') as f:
            json.dump(hp_list, f, indent=2)
        print(f"Generated {len(hp_list)} stage {stage + 1} configs → {out_path}")


if __name__ == '__main__':
    main()
