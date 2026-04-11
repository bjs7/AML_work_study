import configs.configs as config
import os
import json
import logging
from pathlib import Path
import torch
import pickle
import utils
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Path construction helper (shared by build_save_dir and legacy save_results)
# =============================================================================

def _build_path(manager):
    """Return (save_direc_str, str_folder, data_folder, model_tuning_configs)."""
    str_testing = 'testing' if manager.args['data_parser'].testing else ''
    fl_algo = manager.args['fl_parser'].fl_algo

    if fl_algo == 'FedGraph':
        algo_subfolder = manager.args['fl_parser'].aggregation
    elif fl_algo in ('FedAvg', 'FedProx'):
        fl_p = manager.args['fl_parser']
        weighting = getattr(fl_p, 'weighting', 'proportional')
        C = getattr(fl_p, 'client_fraction', 1.0)
        E = getattr(fl_p, 'num_local_epochs', 1)
        mu = getattr(fl_p, 'mu', 0.0)
        R = getattr(fl_p, 'num_rounds', 100)
        algo_subfolder = f'{weighting}_C{C}_E{E}'
        if R != 100:
            algo_subfolder += f'_R{R}'
        if mu > 0:
            algo_subfolder += f'_mu{mu}'
        validate_every = getattr(fl_p, 'validate_every', 1)
        if validate_every != 1:
            algo_subfolder += f'_ve{validate_every}'
    else:
        algo_subfolder = ''

    eval_mode = getattr(manager.args['data_parser'], 'eval_mode', 'system')

    save_direc = os.path.join(config.save_direc_training, str_testing,
                              manager.args['data_parser'].size + '_' + manager.args['data_parser'].ir,
                              f'split_{config.split_perc[0]}_{config.split_perc[1]}',
                              eval_mode, fl_algo, algo_subfolder)

    str_folder = manager.args['fl_parser'].model
    model_tuning_configs = utils.get_tuning_configs(manager.args).get(manager.args['data_parser'].scenario)

    if manager.args['fl_parser'].model_type == 'gnn':
        for key, value in vars(manager.args['gnn_parser']).items():
            if value:
                str_folder += f'__{key}'

    data_flags = ['batching', 'batchnorm', 'ibm_fe', 'ibm_hp', 'use_global_stats']
    data_settings = [flag for flag in data_flags if getattr(manager.args['data_parser'], flag)]
    batching_mode = getattr(manager.args['data_parser'], 'batching_mode', 'lazy_link_neighbor')
    if manager.args['data_parser'].batching and batching_mode != 'lazy_link_neighbor':
        data_settings.append(f'bm_{batching_mode}')
    if manager.args['data_parser'].normalize_currency:
        data_settings.append('normalize_currency')
    if manager.args['data_parser'].bank_filter:
        data_settings.append(f'bank_filter_{manager.args["data_parser"].bank_filter}')
    if manager.args['data_parser'].loss_ratio is not None:
        data_settings.append(f'loss_ratio_{manager.args["data_parser"].loss_ratio}')
    if manager.args['data_parser'].batch_size != 8192:
        data_settings.append(f'batch_size_{manager.args["data_parser"].batch_size}')
    data_folder = '__'.join(data_settings) if data_settings else 'default'

    return save_direc, str_folder, data_folder, model_tuning_configs


# =============================================================================
# Pre-training setup: create output dir and save static files
# =============================================================================

def build_save_dir(manager, hyperparams):
    """Create the experiment output directory before training starts.

    Saves static files (config, hyperparameters) immediately so they are
    available even if the job crashes mid-training. Returns the Path so the
    caller can store it for incremental per-seed saves.

    Args:
        manager: The FL manager instance.
        hyperparams: Tuned hyperparameters dict.

    Returns:
        Path: The created experiment directory.
    """
    save_direc, str_folder, data_folder, model_tuning_configs = _build_path(manager)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = Path(save_direc) / str_folder / data_folder / run_id
    folder_path.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment directory created: %s", folder_path)

    (folder_path / 'model_tuning_configs.json').write_text(json.dumps(model_tuning_configs, indent=4))
    (folder_path / 'experiment_config.json').write_text(
        json.dumps(create_experiment_config(manager), indent=4))

    fl_algo = manager.args['fl_parser'].fl_algo
    if fl_algo != 'individual':
        with open(folder_path / 'hyper_parameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=4)

    return folder_path


# =============================================================================
# Per-seed incremental save
# =============================================================================

def save_seed_result(save_dir, seed, result, manager):
    """Save one seed's results immediately after it completes.

    Args:
        save_dir: Path returned by build_save_dir.
        seed: The seed integer (1-indexed).
        result: The dict returned by _train() for this seed.
        manager: The FL manager instance.
    """
    seed_folder = Path(save_dir) / f'seed_{seed}'
    seed_folder.mkdir(parents=True, exist_ok=True)

    fl_algo = manager.args['fl_parser'].fl_algo
    model_type = manager.args['fl_parser'].model_type

    if model_type == 'gnn':
        if fl_algo == 'individual':
            with open(seed_folder / 'metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': result['metrics'],
                             'laundering_values': result['laundering_values'],
                             'party_performance': result['party_performance']}, f)
            with open(seed_folder / 'models_hyperparameters.pkl', 'wb') as f:
                pickle.dump({'models_hyperparameters': result['models']}, f)
        elif fl_algo == 'full_info':
            with open(seed_folder / 'metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': result['metrics'],
                             'laundering_values': result['laundering_values'],
                             'best_vali_f1': result.get('best_vali_f1')}, f)
            torch.save(result['model'], seed_folder / 'model.pth')
        else:  # FedAvg, FedProx, FedGraph, FedGraphSimple, etc.
            with open(seed_folder / 'metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': result['metrics'],
                             'laundering_values': result['laundering_values'],
                             'best_vali_f1': result.get('best_vali_f1'),
                             'party_performance': result.get('party_performance')}, f)
            torch.save(result['weights'], seed_folder / 'model.pth')

    elif model_type == 'booster':
        with open(seed_folder / 'metrics_laundering_values.pkl', 'wb') as f:
            pickle.dump({'metrics': result['metrics'],
                         'laundering_values': result['laundering_values']}, f)

    logger.info("Seed %d saved - F1: %.4f", seed, result['metrics']['f1'])


# =============================================================================
# Final save: aggregation (seeds already written incrementally)
# =============================================================================

def save_results(results, hyperparams, manager):

    logger.info("="*80)
    logger.info("Saving experiment results")
    logger.info("="*80)

    save_dir = getattr(manager, 'save_dir', None)

    if save_dir is None:
        # Fallback: manager.save_dir was not set (e.g. code path not yet updated).
        # Fall back to the original behaviour: create dir and save seeds now.
        save_dir = build_save_dir(manager, hyperparams)
        for seed, result in results.items():
            save_seed_result(save_dir, seed, result, manager)
    else:
        logger.info("Save directory: %s", save_dir)

    # Aggregated results across all completed seeds
    aggregated_results = aggregate_seed_results(results)
    with open(save_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated_results, f, indent=4)

    logger.info("Aggregated results: Mean F1=%.4f (±%.4f), Best F1=%.4f (seed %d)",
                aggregated_results['f1']['mean'],
                aggregated_results['f1']['std'],
                aggregated_results['best_f1'],
                aggregated_results['best_seed'])
    logger.info("Results saved successfully to: %s", save_dir)
    logger.info("="*80)


def create_experiment_config(manager):
    """Create experiment configuration dictionary, including only fields
    relevant to the current model type and FL algorithm."""
    fl_algo = manager.args['fl_parser'].fl_algo
    model_type = manager.args['fl_parser'].model_type

    fl_cfg = {
        "fl_algo": fl_algo,
        "scenario": manager.args['data_parser'].scenario,
    }

    # FedAvg / FedProx specific
    if fl_algo in ('FedAvg', 'FedProx'):
        fl_p = manager.args['fl_parser']
        fl_cfg.update({
            "weighting": getattr(fl_p, 'weighting', 'proportional'),
            "client_fraction": getattr(fl_p, 'client_fraction', 1.0),
            "num_local_epochs": getattr(fl_p, 'num_local_epochs', 1),
            "num_rounds": getattr(fl_p, 'num_rounds', 100),
            "mu": getattr(fl_p, 'mu', 0.0),
        })

    # GNN-only FL fields
    if model_type == 'gnn':
        fl_cfg["aggregation"] = manager.args['fl_parser'].aggregation
        fl_cfg["validate_every"] = getattr(manager.args['fl_parser'], 'validate_every', 1)

    model_cfg = {
        "model_type": model_type,
        "model_name": manager.args['fl_parser'].model,
    }
    if model_type == 'gnn':
        model_cfg["gnn_settings"] = {
            key: value for key, value in vars(manager.args['gnn_parser']).items()}

    experiment_configs = {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "size": manager.args['data_parser'].size,
            "ir": manager.args['data_parser'].ir,
            "split": list(config.split_perc),
            "eval_mode": getattr(manager.args['data_parser'], 'eval_mode', 'system'),
            "testing": manager.args['data_parser'].testing,
            "ibm_fe": manager.args['data_parser'].ibm_fe,
            "batchnorm": manager.args['data_parser'].batchnorm,
            "normalize_currency": manager.args['data_parser'].normalize_currency,
            "bank_filter": manager.args['data_parser'].bank_filter,
            "loss_ratio": manager.args['data_parser'].loss_ratio,
            "batch_size": manager.args['data_parser'].batch_size,
            "batching_mode": getattr(manager.args['data_parser'], 'batching_mode', 'lazy_link_neighbor'),
        },
        "fl": fl_cfg,
        "model": model_cfg,
        "training": {
            "seeds": manager.args['data_parser'].testing_seeds,
            "epochs": config.epochs if model_type != 'booster' else None,
        }
    }

    return experiment_configs


def aggregate_seed_results(results_by_seed):
    """Aggregate metrics across all seeds."""
    metric_names = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc']
    aggregated = {}

    for metric in metric_names:
        values = [results_by_seed[seed]['metrics'][metric] for seed in sorted(results_by_seed.keys())]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    best_seed = max(results_by_seed.keys(), key=lambda s: results_by_seed[s]['metrics']['f1'])
    aggregated['best_seed'] = best_seed
    aggregated['best_f1'] = results_by_seed[best_seed]['metrics']['f1']

    return aggregated
