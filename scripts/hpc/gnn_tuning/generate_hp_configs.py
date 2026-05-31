#!/usr/bin/env python3
"""Generate stage 1 HP configs for parallel GNN tuning and save to a JSON file.

Run from the project root (or let the submit script call it):
  python scripts/hpc/gnn_tuning/generate_hp_configs.py \
      --fl_algo full_info --model GINe --size small --ir HI --eval_mode system

Prints N_CONFIGS=<n> to stdout so the submit script can derive the array bounds.
"""
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
from configs.paths import get_tuning_configs, get_data_path


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
    parsers = utils.parser_all()

    random.seed(parsers['data_parser'].seed)

    tuning_configs = get_tuning_configs(parsers)
    x_0 = tuning_configs[parsers['data_parser'].scenario][parsers['data_parser'].size]['x_0']

    hp_list = [hyper_sampler(parsers['fl_parser'], None) for _ in range(x_0)]

    work_dir = get_work_dir(parsers)
    os.makedirs(work_dir, exist_ok=True)

    out_path = os.path.join(work_dir, 'stage1_configs.json')
    with open(out_path, 'w') as f:
        json.dump(hp_list, f, indent=2)

    print(f"Generated {len(hp_list)} stage 1 HP configs → {out_path}")
    print(f"N_CONFIGS={len(hp_list)}", flush=True)


if __name__ == '__main__':
    main()
