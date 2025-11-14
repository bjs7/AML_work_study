import configs.configs as config
import os
import json
from pathlib import Path
import torch
import pickle
import utils

def save_results(results, hyperparams, manager):

    str_testing = 'testing' if manager.args['data_parser'].testing else ''
    save_direc = os.path.join(config.save_direc_training, str_testing,
                            manager.args['data_parser'].size + '_' + manager.args['data_parser'].ir,
                            f'split_{config.split_perc[0]}_{config.split_perc[1]}',
                            manager.args['fl_parser'].fl_algo)

    str_folder = manager.args['fl_parser'].model
    model_tuning_configs = utils.get_tuning_configs(manager.args).get(manager.args['data_parser'].scenario)

    if manager.args['fl_parser'].model_type == 'gnn':
        for key, value in vars(manager.args['gnn_parser']).items():
            if value:
                model_tuning_configs[key] = value
                str_folder += f'__{key}'

    elif manager.args['fl_parser'].model_type == 'booster':
        x_0_fi, r_0_fi = model_tuning_configs.get('full_info').get(manager.args['data_parser'].size).get('x_0'), model_tuning_configs.get('full_info').get(manager.args['data_parser'].size).get('r_0')

    save_direc = os.path.join(save_direc, str_folder)
    folder_path = Path(save_direc)
    file_path = folder_path / 'model_tuning_configs.json'
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(model_tuning_configs, indent=4))


    if manager.args['fl_parser'].fl_algo != 'individual':
        save_FL(save_direc, results, hyperparams, manager)
    else:
        save_individual(save_direc, results, manager)


def save_FL(save_direc, results, hyperparams, manager):

    with open(save_direc + '/metrics_laundering_values.pkl', 'wb') as f:
        pickle.dump({'metrics': results['metrics'], 'laundering_values': results['laundering_values']}, f)

    if manager.args['fl_parser'].model_type == 'gnn':
        torch.save(results['w'], save_direc + '/model.pth')
    elif manager.args['fl_parser'].model_type == 'booster':
        print('save booster')

    if manager.args['fl_parser'].fl_algo != 'individual':
        file_path = os.path.join(save_direc, 'hyper_parameters.json')
        with open(file_path, 'w') as file:
            json.dump(hyperparams, file, indent=4)


def save_individual(save_direc, results, manager):

    with open(save_direc + '/metrics_laundering_values.pkl', 'wb') as f:
        pickle.dump({'metrics': results['metrics'], 'laundering_values': results['laundering_values']}, f)

    with open(save_direc + '/models_hyperparameters.pkl', 'wb') as f:
        pickle.dump({'models_hyperparameters': results['models']}, f)
