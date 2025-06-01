import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random
import json


model_types = {
    'GINe': 'graph',
    'xgboost': 'booster',
    'light_gbm': 'booster'
}

data_types = {
    'graph': 'graph_data',
    'booster': 'regular_data'
}

file_types = {
    'graph': 'pth',
    'booster': 'ubj' #pkl
}

def get_data_path():
    local_path = "/home/nam_07/projects"
    hpc_path = "/data/leuven/362/vsc36278"
    
    # Check which path exists
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(hpc_path):
        return hpc_path
    else:
        raise FileNotFoundError("Neither data path exists: local_path or hpc_path")


def get_model_configs(args):

    with open('model_configs.json', 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(args.model)

def get_tuning_configs(args):

    tuning_configs = 'tuning_configs_for_testing' if args.testing else 'tuning_configs'

    if get_data_path() == '/data/leuven/362/vsc36278':
        folder = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study/configs/' + tuning_configs + '.json'
    else:
        folder = 'configs/' + tuning_configs + '.json'

    with open(folder, 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(args.model_type)


def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )



def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--ports", action='store_true')

    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=0, type=int, help="Set seed for reproducability")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--specific_banks', default=[], type=parse_banks, help='Used if specific banks are to be studied')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    
    return parser


def parse_banks(value):

    value = value.strip()

    if value.startswith('[') and value.endswith(']') and ':' in value:
        start, end = value[1:-1].split(':')
        return list(range(int(start), int(end)))

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError('Invalid format')


def parse_data_split(value):
    
    value = value.strip()

    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list) and all(isinstance(i, (int, float)) for i in parsed_value):
            return parsed_value
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid format for --data_split. Use [0.6, 0.2]")


def set_seed(seed: int = 0, log = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if log:
        logging.info(f"Random seed set as {seed}")

