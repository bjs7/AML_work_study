#!/usr/bin/env python3
"""Evaluate one GNN HP config and save its validation F1.

Called by run_hp_eval.sh for each SLURM array task. Loads the HP config at
index --hp_idx from the stage JSON, trains the full_info model once, and
writes the result to the stage results directory.

Usage (via SLURM array job, not directly):
  python eval_single_hp.py \
      --stage 1 --hp_idx 3 \
      --fl_algo full_info --model GINe --size small --ir HI --eval_mode system --batching
"""
import argparse
import copy
import json
import logging
import os
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
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos   # populates registry
import models.gnn_models              # registers GNN model classes
from configs.paths import get_data_path


def get_work_dir(parsers):
    size = parsers['data_parser'].size
    ir = parsers['data_parser'].ir
    eval_mode = getattr(parsers['data_parser'], 'eval_mode', 'system')
    if get_data_path() == '/data/leuven/362/vsc36278':
        base = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    return os.path.join(base, 'configs', 'gnn_tuning_work', f'{size}_{ir}_{eval_mode}')


def main():
    # Parse our own args first, strip them from sys.argv so utils.parser_all() works cleanly
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    pre_parser.add_argument('--hp_idx', type=int, required=True)
    our_args, remaining = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    utils.logger_setup()
    parsers, df, scaler_encoders = utils.setup_get_data()

    laundering_values_vali, _ = dfn.prep_laundering_dfs(
        parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])}
    )

    work_dir = get_work_dir(parsers)
    configs_path = os.path.join(work_dir, f'stage{our_args.stage}_configs.json')
    with open(configs_path) as f:
        hp_configs = json.load(f)
    hp = hp_configs[our_args.hp_idx]

    logging.info("Stage %d | HP idx %d | config: %s", our_args.stage, our_args.hp_idx, hp)

    # Set up manager and party directly (bypasses setup_parties to avoid triggering tuning)
    manager = Manager.get_algo_class(parsers)
    manager._add_party(None, df, parsers, scaler_encoders)
    manager.set_mode('tuning')  # after _add_party so mode propagates to the party
    next(iter(manager.parties.values())).prep_data()

    # tuning_loop with a single-element list = exactly one HP evaluation
    _, _, f1 = manager.tuning_loop([hp], laundering_values_vali)

    result = {'hp_idx': our_args.hp_idx, 'f1': f1, 'hyperparameters': hp}
    results_dir = os.path.join(work_dir, f'stage{our_args.stage}_results')
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f'hp_{our_args.hp_idx:03d}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info("Stage %d | HP idx %d | F1=%.4f → %s", our_args.stage, our_args.hp_idx, f1, result_path)


if __name__ == '__main__':
    main()