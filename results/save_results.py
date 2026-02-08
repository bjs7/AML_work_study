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

def save_results(results, hyperparams, manager):

    logger.info("="*80)
    logger.info("Saving experiment results")
    logger.info("="*80)

    # get general folders
    str_testing = 'testing' if manager.args['data_parser'].testing else ''
    fl_algo = manager.args['fl_parser'].fl_algo

    # Build algo-specific subfolder
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
    else:
        algo_subfolder = ''

    eval_mode = getattr(manager.args['data_parser'], 'eval_mode', 'system')

    save_direc = os.path.join(config.save_direc_training, str_testing,
                            manager.args['data_parser'].size + '_' + manager.args['data_parser'].ir,
                            f'split_{config.split_perc[0]}_{config.split_perc[1]}',
                            eval_mode,
                            fl_algo,
                            algo_subfolder)

    str_folder = manager.args['fl_parser'].model
    model_tuning_configs = utils.get_tuning_configs(manager.args).get(manager.args['data_parser'].scenario)

    # get gnn folder
    if manager.args['fl_parser'].model_type == 'gnn':
        for key, value in vars(manager.args['gnn_parser']).items():
            if value:
                str_folder += f'__{key}'


    # get booster folder
    elif manager.args['fl_parser'].model_type == 'booster':
        x_0_fi, r_0_fi = model_tuning_configs.get('full_info').get(manager.args['data_parser'].size).get('x_0'), model_tuning_configs.get('full_info').get(manager.args['data_parser'].size).get('r_0')

    # add data flags to folder name
    data_flags = ['batching', 'batchnorm', 'ibm_fe', 'ibm_hp', 'use_global_stats'] #'add_ids',
    data_settings = [flag for flag in data_flags if getattr(manager.args['data_parser'], flag)]
    if manager.args['data_parser'].normalize_currency:
        data_settings.append('normalize_currency')
    if manager.args['data_parser'].bank_filter:
        data_settings.append(f'bank_filter_{manager.args["data_parser"].bank_filter}')
    if manager.args['data_parser'].loss_ratio is not None:
        data_settings.append(f'loss_ratio_{manager.args["data_parser"].loss_ratio}')
    data_folder = '__'.join(data_settings) if data_settings else 'default'

    # create the folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_direc = os.path.join(save_direc, str_folder, data_folder, run_id)
    folder_path = Path(save_direc)
    folder_path.mkdir(parents=True, exist_ok=True)

    logger.info("Save directory: %s", save_direc)
    logger.info("Data settings: %s", data_folder if data_settings else "default (no special flags)")

    # save tuning configs
    file_path = folder_path / 'model_tuning_configs.json'
    file_path.write_text(json.dumps(model_tuning_configs, indent=4))
    logger.debug("Saved model tuning configs")

    # save experiment configs
    experiment_config = create_experiment_config(manager)
    file_path = folder_path / 'experiment_config.json'
    file_path.write_text(json.dumps(experiment_config, indent=4))
    logger.debug("Saved experiment config")

    # save the results
    if manager.args['fl_parser'].fl_algo == 'full_info':
        logger.info("Saving Full Info results")
        save_full_info(save_direc, results, hyperparams, manager)
    elif manager.args['fl_parser'].fl_algo != 'individual':
        logger.info("Saving Federated Learning results")
        save_FL(save_direc, results, hyperparams, manager)
    else:
        logger.info("Saving Individual bank results")
        save_individual(save_direc, results, manager)

    logger.info("Results saved successfully to: %s", save_direc)
    logger.info("="*80)


def save_FL(save_direc, results, hyperparams, manager):

    # save hyperparameters
    file_path = os.path.join(save_direc, 'hyper_parameters.json')
    with open(file_path, 'w') as file:
        json.dump(hyperparams, file, indent=4)

    if manager.args['fl_parser'].model_type == 'gnn':

        for seed, seed_result in results.items():

            seed_folder = os.path.join(save_direc, f'seed_{seed}')
            folder_path = Path(seed_folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            with open(f'{seed_folder}/metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': seed_result['metrics'], 'laundering_values': seed_result['laundering_values'],
                             'best_vali_f1': seed_result.get('best_vali_f1'),
                             'party_performance': seed_result.get('party_performance')}, f)

            torch.save(seed_result['weights'], f'{seed_folder}/model.pth')

        aggregated_results = aggregate_seed_results(results)
        file_path = os.path.join(save_direc, 'aggregated_results.json')
        with open(file_path, 'w') as file:
            json.dump(aggregated_results, file, indent=4)


def save_individual(save_direc, results, manager):


    if manager.args['fl_parser'].model_type == 'gnn':

        logger.info("Saving results for %d seeds", len(results))
        for seed, seed_result in results.items():

            seed_folder = os.path.join(save_direc, f'seed_{seed}')
            folder_path = Path(seed_folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            with open(f'{seed_folder}/metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': seed_result['metrics'], 
                             'laundering_values': seed_result['laundering_values'],
                             'party_performance': seed_result['party_performance']}, f)

            with open(f'{seed_folder}/models_hyperparameters.pkl', 'wb') as f:
                pickle.dump({'models_hyperparameters': seed_result['models']}, f)

            logger.debug("Seed %d saved - F1: %.4f", seed, seed_result['metrics']['f1'])

        aggregated_results = aggregate_seed_results(results)
        file_path = os.path.join(save_direc, 'aggregated_results.json')
        with open(file_path, 'w') as file:
            json.dump(aggregated_results, file, indent=4)

        logger.info("Aggregated results: Mean F1=%.4f (±%.4f), Best F1=%.4f (seed %d)",
                   aggregated_results['f1']['mean'],
                   aggregated_results['f1']['std'],
                   aggregated_results['best_f1'],
                   aggregated_results['best_seed'])


def save_full_info(save_direc, results, hyperparams, manager):

    # save hyperparameters
    file_path = os.path.join(save_direc, 'hyper_parameters.json')
    with open(file_path, 'w') as file:
        json.dump(hyperparams, file, indent=4)
    logger.debug("Saved hyperparameters")

    # save seeds and the results
    if manager.args['fl_parser'].model_type == 'gnn':

        logger.info("Saving results for %d seeds", len(results))
        for seed, seed_result in results.items():

            seed_folder = os.path.join(save_direc, f'seed_{seed}')
            folder_path = Path(seed_folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            with open(f'{seed_folder}/metrics_laundering_values.pkl', 'wb') as f:
                pickle.dump({'metrics': seed_result['metrics'], 'laundering_values': seed_result['laundering_values'],
                             'best_vali_f1': seed_result.get('best_vali_f1')}, f)

            torch.save(seed_result['model'], f'{seed_folder}/model.pth')

            logger.debug("Seed %d saved - F1: %.4f", seed, seed_result['metrics']['f1'])

        aggregated_results = aggregate_seed_results(results)
        file_path = os.path.join(save_direc, 'aggregated_results.json')
        with open(file_path, 'w') as file:
            json.dump(aggregated_results, file, indent=4)

        logger.info("Aggregated results: Mean F1=%.4f (±%.4f), Best F1=%.4f (seed %d)",
                   aggregated_results['f1']['mean'],
                   aggregated_results['f1']['std'],
                   aggregated_results['best_f1'],
                   aggregated_results['best_seed'])


def create_experiment_config(manager):
    """Create comprehensive experiment configuration dictionary.

    Args:
        manager: The FL manager instance

    Returns:
        dict: Complete experiment configuration
    """
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
            "loss_ratio": manager.args['data_parser'].loss_ratio
        },
        "fl": {
            "fl_algo": manager.args['fl_parser'].fl_algo,
            "scenario": manager.args['data_parser'].scenario,
            "aggregation": manager.args['fl_parser'].aggregation,
            "weighting": getattr(manager.args['fl_parser'], 'weighting', 'proportional'),
            "client_fraction": getattr(manager.args['fl_parser'], 'client_fraction', 1.0),
            "num_local_epochs": getattr(manager.args['fl_parser'], 'num_local_epochs', 1),
            "num_rounds": getattr(manager.args['fl_parser'], 'num_rounds', 100),
            "mu": getattr(manager.args['fl_parser'], 'mu', 0.0)
        },
        "model": {
            "model_type": manager.args['fl_parser'].model_type,
            "model_name": manager.args['fl_parser'].model
        },
        "training": {
            "seeds": 4,  # You could make this configurable
            #"epochs": manager.args['data_parser'].epochs if hasattr(manager.args['data_parser'], 'epochs') else None
            "epochs": config.epochs if manager.args['fl_parser'].model_type != 'booster' else None
        }
    }

    # Add GNN-specific settings if applicable
    if manager.args['fl_parser'].model_type == 'gnn':
        experiment_configs["model"]["gnn_settings"] = {key: value for key, value in
                                                       vars(manager.args['gnn_parser']).items()}
    return experiment_configs

def aggregate_seed_results(results_by_seed):

    """Aggregate metrics across all seeds."""
    metric_names = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc']  # Add all metrics you track
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
    
    # Find best seed
    best_seed = max(results_by_seed.keys(), key=lambda s: results_by_seed[s]['metrics']['f1'])
    aggregated['best_seed'] = best_seed
    aggregated['best_f1'] = results_by_seed[best_seed]['metrics']['f1']
    
    return aggregated

