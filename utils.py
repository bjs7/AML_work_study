import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import json
import random
import pandas as pd
import copy

#from data.get_indices_type_data import get_indices_bdt
#from federated_learning.fl_base import Manager, Party
#import data.data_functions as dfn
#from data.raw_data_processing import get_data
#from configs.configs import split_perc

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
# util functions


def setup_get_data():

    parsers = parser_all()
    set_seed(parsers['data_parser'].seed, True)
    
    df = pd.read_csv(f"{get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

    if parsers['data_parser'].testing and parsers['fl_parser'].fl_algo == 'full_info':
        print('set set of data')
        df = pd.concat([df.iloc[0:50000,:], df.iloc[3000000:3050000,:], df.iloc[5000000:5050000,:]])

    df, scaler_encoders  = get_data(df, parsers['data_parser'], split_perc = split_perc)

    return parsers, df, scaler_encoders


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
        format="%(asctime)s [%(levelname)-5.5s] %(name)-20s - %(message)s",
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
    parser.add_argument('--fl_algo', default='FedAvg', type=str)
    parser.add_argument('--model', default='GINe', type=str)
    #parser.add_argument('--model', default='xgboost', type=str)
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
    parser.add_argument('--train_for_final', action='store_true', help='Use train data for final training instead of vali')
    parser.add_argument('--ibm_fe', action='store_true', help='Set to True if the feature engineering should be 1:1 with the IBM paper')
    parser.add_argument('--ibm_hp', action='store_true', help='Set to True if the IBM hyperparameters should be used')

    # utils
    parser.add_argument('--testing', action='store_true')
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=1, type=int, help="Set seed for reproducability")

    return parser

# gnn parser
def gnn_parser():

    parser = argparse.ArgumentParser(description="args for gnn models")
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--ports", action='store_true')
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")

    return parser


def init_parties(df, parsers, manager, scaler_encoders = None):

    if parsers['fl_parser'].fl_algo == 'full_info':

        bank_indices = get_indices_bdt(df, bank=None)
        train_data, vali_data, test_data = dfn.fl_get_data(parsers, df, bank_indices)
        tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
        Party.get_algo_class(parsers = parsers, bank_id=None, data=tmp_data, 
                                indices=bank_indices, manager=manager, 
                                scaler_encoders=scaler_encoders)
    else:           
        fr_banks, sr_banks = get_relevant_banks(data_parser)
        relevant_banks = fr_banks #+ sr_banks
        relevant_banks = relevant_banks[0:4] + relevant_banks[5:10]

        for bank in relevant_banks:

            bank_indices = get_indices_bdt(df, bank=bank)
            train_data, vali_data, test_data = dfn.fl_get_data(parsers, df, bank_indices)
            tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
            Party.get_algo_class(parsers = parsers, bank_id=bank, data=tmp_data, 
                                indices=bank_indices, manager=manager, 
                                scaler_encoders=scaler_encoders)

        manager._num_parties = len(manager.parties)



def add_banks_to_manager(parsers, banks, manager, df, scaler_encoders, tuned_hp = None):

    # this part here is only used for individual banks settings
    if tuned_hp is not None:
        best_tuned_hp = max(tuned_hp.values(), key=lambda x: x['f1_score'])['hyperparameters']
        tuned_hp = {bank_id: entry['hyperparameters'] for bank_id, entry in tuned_hp.items()}

    for bank in banks:
        manager._add_party(bank, df, parsers, copy.deepcopy(scaler_encoders))
        
        if tuned_hp is not None:
            tuned_hp[bank] = best_tuned_hp
    
    manager._num_parties = len(manager.parties)

    return tuned_hp


