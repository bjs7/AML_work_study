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
