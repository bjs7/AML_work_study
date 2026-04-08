import os
import json


def get_data_path():
    local_path = "/home/nam_07/projects"
    hpc_path = "/data/leuven/362/vsc36278"

    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(hpc_path):
        return hpc_path
    else:
        raise FileNotFoundError("Neither data path exists: local_path or hpc_path")


def get_tuning_configs(parsers):

    tuning_configs = 'tuning_configs_for_testing' if parsers['data_parser'].testing else 'tuning_configs'

    if get_data_path() == '/data/leuven/362/vsc36278':
        folder = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study/configs/' + tuning_configs + '.json'
    else:
        folder = 'configs/' + tuning_configs + '.json'

    with open(folder, 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(parsers['fl_parser'].model_type)


def get_full_info_hp_path(parsers, model=None):
    """Return path where full_info tuned booster HPs are saved/loaded.

    Args:
        parsers: dict with 'data_parser' and 'fl_parser'.
        model: override the model name in the path (e.g. 'xgboost' when
               SecureBoost wants to reuse xgboost-tuned HPs).
    """
    model_name = model or parsers['fl_parser'].model
    size = parsers['data_parser'].size
    ir   = parsers['data_parser'].ir

    if get_data_path() == '/data/leuven/362/vsc36278':
        base = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study/configs/tuned_hyperparams'
    else:
        base = 'configs/tuned_hyperparams'

    return os.path.join(base, 'booster', model_name, f'{size}_{ir}.json')
