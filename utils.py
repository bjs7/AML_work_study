import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import json
import random


# --------------------------------------------------------------------------------------------------
# dictionary holders

model_types = {
    'GINe': 'gnn',
    'xgboost': 'booster',
    'light_gbm': 'booster',
    'regression': 'regression'
}

data_types = {
    'gnn': 'graph_data',
    'booster': 'regular_data',
    'regression': 'regular_data'
}

file_types = {
    'gnn': 'pth',
    'booster': 'ubj'
}

# --------------------------------------------------------------------------------------------------
# kept here tmp. Gonna need to update relevant banks file later.

import configs.configs as config

def load_relevant_banks(args):

    save_direc = config.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    
    split = config.split_perc[0:2]
    str_folder = f'{args.size}_' + args.ir + f'__split_{split[0]}_{split[1]}.json'
    file_location = os.path.join(save_direc, str_folder)

    with open(file_location, 'r') as file:
        relevant_banks = json.load(file)

    return relevant_banks

def get_relevant_banks(args):
    relevant_banks = load_relevant_banks(args).get(args.banks)
    return relevant_banks.get('fr_banks'), relevant_banks.get('sr_banks')



# --------------------------------------------------------------------------------------------------
# util functions

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


def get_tuning_configs(parsers):

    tuning_configs = 'tuning_configs_for_testing' if parsers['data_parser'].testing else 'tuning_configs'

    if get_data_path() == '/data/leuven/362/vsc36278':
        folder = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study/configs/' + tuning_configs + '.json'
    else:
        folder = 'configs/' + tuning_configs + '.json'

    with open(folder, 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(parsers['fl_parser'].model_type)


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


def get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim):

    if parsers['data_parser'].data_type == 'graph_data':
        bank_dim = train_data['df']['x'].shape[0]
    else:
        bank_dim = train_data['x'].shape[0]

    if bank_dim < smallest_dim:
        smallest_bank = bank
        smallest_dim = bank_dim

    return smallest_bank, smallest_dim

# --------------------------------------------------------------------------------------------------
# parser function for all the parsers

def parser_all():

    parsers = [
        ("fl_parser", fl_parser()),
        ("data_parser", data_parser()),
        ("gnn_parser", gnn_parser())
    ]

    remaining_args = sys.argv[1:]
    all_parsers = {}

    for name, parser in parsers:
        try:
            parsed, remaining_args = parser.parse_known_args(remaining_args)
            all_parsers[name.lower()] = parsed
        except Exception as e:
            all_parsers[name.lower()] = None

    if remaining_args:
        print(f"Unparsed arguments: {remaining_args}")

    all_parsers['fl_parser'].model_type = model_types.get(all_parsers['fl_parser'].model)
    all_parsers['fl_parser'].data_type = data_types.get(all_parsers['fl_parser'].model_type)
    all_parsers['data_parser'].data_type = data_types.get(all_parsers['fl_parser'].model_type)

    all_parsers['data_parser'].scenario = 'individual_banks' if all_parsers['fl_parser'].fl_algo != 'full_info' else 'full_info'

    return all_parsers


# fl parser
def fl_parser():

    parser = argparse.ArgumentParser(description="main args for fl")
    parser.add_argument('--fl_algo', default='FedGD', type=str)
    parser.add_argument('--model', default='GINe', type=str)
    #parser.add_argument('--regu', default=)
    
    return parser

# data parser
def data_parser():

    parser = argparse.ArgumentParser(description="args for data configs and utils")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--overwrite', action='store_true')

    # utils
    parser.add_argument('--testing', action='store_true')
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=0, type=int, help="Set seed for reproducability")

    return parser

# gnn parser
def gnn_parser():

    parser = argparse.ArgumentParser(description="args for gnn models")
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--ports", action='store_true')
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")

    return parser
