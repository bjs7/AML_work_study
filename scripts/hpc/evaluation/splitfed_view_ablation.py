"""Test-time view ablation for trained SplitFed models (no training).

For each seed_k/model.pth in --run_dir, rebuilds the SplitFed test pipeline exactly
as in training (set_seed -> setup -> setup_model), loads the validation-selected
weights and runs the GNN once over the test split. From the same sender/receiver
embeddings, mlp_vert is applied three times:

    both           [s | r]   (regular SplitFed prediction)
    sender_only    [s | 0]
    receiver_only  [0 | r]

Ablation is applied to cross-bank transactions only; intra-bank rows keep their
regular [s | 0] input in every condition and are flagged with is_intra. All three
conditions share the same sampled test subgraphs, so differences are purely due
to the missing view.

Output per seed, kept apart from the training output (seed_k/ is never written to):
    <run_dir>/evaluation/seed_k/predictions_evalseed{S}.pkl   (DataFrame indexed by global transaction ID)
    <run_dir>/evaluation/seed_k/eval_config_evalseed{S}.json  (source model, seeds, flags, date)
Pass the same model flags as the training run, e.g.

    python scripts/hpc/evaluation/splitfed_view_ablation.py \
        --fl_algo SplitFed --model GINe --size small --ir HI --batching --ibm_hp --emlps \
        --eval_mode system --batching_mode lazy_link_neighbor \
        --run_dir <experiments/.../FedGraphSimple/GINe__emlps/batching__ibm_hp/<run_id>>

Extra flags: --run_dir (required), --seeds 1 2 3 4 (default: every seed_* with a
model.pth), --eval_seed S (sampling seed; default: the model's training seed).
"""

import sys
import os

_hpc_repo = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study'
if os.path.exists(_hpc_repo):
    sys.path.insert(0, _hpc_repo)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import copy
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

import utils
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos  # noqa: F401 — registers FL algorithms
import models.gnn_models  # noqa: F401 — registers GNN architectures
from federated_learning.gnn.splitfed.forward import build_view_embeddings
from inference import probs_to_binary

logger = logging.getLogger(__name__)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--run_dir', type=str, required=True)
_parser.add_argument('--seeds', type=int, nargs='+', default=None)
_parser.add_argument('--eval_seed', type=int, default=None)
_args, _ = _parser.parse_known_args()

CONDITIONS = ('both', 'sender_only', 'receiver_only')


def _seed_dirs(run_dir, seeds):
    found = {int(p.parent.name.split('_')[1]): p for p in run_dir.glob('seed_*/model.pth')}
    if seeds is not None:
        missing = set(seeds) - set(found)
        if missing:
            raise FileNotFoundError(f"No model.pth for seeds {sorted(missing)} in {run_dir}")
        found = {s: found[s] for s in seeds}
    if not found:
        raise FileNotFoundError(f"No seed_*/model.pth found in {run_dir}")
    return dict(sorted(found.items()))


def _evaluate_views(manager):
    """One GNN pass over the test split; mlp_vert applied per ablation condition."""
    mlp_vert = manager.model.gnn.mlp_vert
    use_batched = manager.args['data_parser'].batching
    frames = []

    with torch.no_grad():
        for batch_key, batch_banks, batch_data in manager._iter_batches('test', use_batched):
            v = build_view_embeddings(manager, 'test', batch_key, batch_banks, batch_data, use_batched)
            s, r = v['from_embeds'], v['to_embeds']
            is_intra = v['from_banks'] == v['to_banks']
            cross = torch.as_tensor(~is_intra, device=s.device).unsqueeze(1)

            inputs = {
                'both': torch.cat([s, r], dim=1),
                'sender_only': torch.cat([s, torch.where(cross, torch.zeros_like(r), r)], dim=1),
                'receiver_only': torch.cat([torch.where(cross, torch.zeros_like(s), s), r], dim=1),
            }
            frame = pd.DataFrame({
                'from_bank': v['from_banks'], 'to_bank': v['to_banks'],
                'true_y': v['true_y'].astype(int), 'is_intra': is_intra,
                'has_sender': v['has_from'], 'has_receiver': v['has_to'],
            }, index=v['batch_labels'].index)
            for cond, x in inputs.items():
                probs = torch.softmax(mlp_vert(x), dim=1)[:, 1].cpu().numpy()
                frame[f'prob_{cond}'] = probs
                frame[f'pred_{cond}'] = probs_to_binary(probs)
            frames.append(frame)

    return pd.concat(frames)


def _summary(df):
    rows = []
    for subset, mask in [('all', np.ones(len(df), dtype=bool)), ('cross_bank', ~df['is_intra'].values)]:
        sub = df[mask]
        for cond in CONDITIONS:
            if subset == 'all' and cond != 'both':
                continue
            y, p = sub['true_y'], sub[f'pred_{cond}']
            rows.append({
                'subset': subset, 'condition': cond, 'n': len(sub), 'n_illicit': int(y.sum()),
                'f1': f1_score(y, p, zero_division=0),
                'precision': precision_score(y, p, zero_division=0),
                'recall': recall_score(y, p, zero_division=0),
            })
    return pd.DataFrame(rows)


def main():
    utils.logger_setup()
    run_dir = Path(_args.run_dir)
    seed_dirs = _seed_dirs(run_dir, _args.seeds)

    parsers, df, scaler_encoders = utils.setup_get_data()
    if parsers['fl_parser'].fl_algo != 'SplitFed':
        raise ValueError("View ablation is defined for --fl_algo SplitFed only")

    laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(
        parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

    manager = Manager.get_algo_class(parsers)
    tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)
    manager.set_mode('training')
    manager._prep_parties_data()

    batching_mode = getattr(parsers['data_parser'], 'batching_mode', 'neighbor_sample')

    for seed, model_path in seed_dirs.items():
        eval_seed = _args.eval_seed if _args.eval_seed is not None else seed
        logger.info("Seed %d: loading %s (eval_seed=%d)", seed, model_path, eval_seed)

        # Same order as GNNMixinManager.train -> _train, minus the training loop.
        utils.set_seed(eval_seed)
        manager.setup(batching=parsers['data_parser'].batching, batching_mode=batching_mode)
        manager.setup_model(tuned_hp, copy.deepcopy(laundering_values_test))
        manager.model.gnn.load_state_dict(torch.load(model_path, map_location=manager.device))
        manager.model.gnn.eval()

        preds = _evaluate_views(manager)
        summary = _summary(preds)

        out_dir = run_dir / 'evaluation' / f'seed_{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'predictions_evalseed{eval_seed}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump({'predictions': preds, 'summary': summary,
                         'model_path': str(model_path), 'eval_seed': eval_seed}, f)
        (out_dir / f'eval_config_evalseed{eval_seed}.json').write_text(json.dumps({
            'model_path': str(model_path),
            'training_seed': seed,
            'eval_seed': eval_seed,
            'conditions': list(CONDITIONS),
            'argv': sys.argv[1:],
            'created': datetime.now().isoformat(timespec='seconds'),
        }, indent=4))

        original = model_path.parent / 'metrics_laundering_values.pkl'
        if original.exists():
            with open(original, 'rb') as f:
                logger.info("Seed %d: original test F1 = %s", seed, pickle.load(f)['metrics'].get('f1'))
        logger.info("Seed %d summary:\n%s", seed, summary.to_string(index=False, float_format='%.4f'))
        logger.info("Saved %s", out_path)


if __name__ == '__main__':
    main()
