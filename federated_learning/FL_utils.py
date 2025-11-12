import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random
import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np




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
    parser.add_argument('--model', default='GINe', type=str) # regression
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




# --------------------------------------------------------------------------------------------------
# util functions


def get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim):

    if parsers['data_parser'].data_type == 'graph_data':
        bank_dim = train_data['df']['x'].shape[0]
    else:
        bank_dim = train_data['x'].shape[0]

    if bank_dim < smallest_dim:
        smallest_bank = bank
        smallest_dim = bank_dim

    return smallest_bank, smallest_dim


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




# --------------------------------------------------------------------------------------------------
# tuning functions

def get_tuning_configs(parsers):

    tuning_configs = 'tuning_configs_for_testing' if parsers['data_parser'].testing else 'tuning_configs'

    if get_data_path() == '/data/leuven/362/vsc36278':
        folder = '/data/leuven/362/vsc36278/AML_work_study/AML_work_study/configs/' + tuning_configs + '.json'
    else:
        folder = 'configs/' + tuning_configs + '.json'

    with open(folder, 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(parsers['fl_parser'].model_type)


def hyper_sampler(args, num_nodes = None, sample_intervals = None):

    if args.model == 'xgboost':
        
        device = None
        if get_data_path == "/data/leuven/362/vsc36278":
            device = "cuda"

        parameters = {
            "num_rounds": random.randint(10, 1000),
            "params": {
                "objective":  "binary:logistic",
                "eval_metric": "logloss",
                
                "max_depth": random.randint(1, 15),
                "learning_rate": random.uniform(10**(-2.5), 10**(-1)),
                "lambda": random.uniform(10**(-2), 10**(2)),
                "scale_pos_weight": random.uniform(1, 10),
                "colsample_bytree": random.uniform(0.5, 1.0),
                "subsample": random.uniform(0.5, 1.0),
                "tree_method": "hist",
                "device": device,
                "random_state": 1
                }
            }
    
    elif args.model == 'light_gbm':
        
        if args.size == 'large':

            parameters = {
                'num_rounds': random.randint(32, 512),
                'params': {

                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',

                'num_leaves': random.randint(32, 256),
                'learning_rate':  random.uniform(0.001, 0.01),
                'lambda_l2': random.uniform(0.01, 0.5),
                'scale_pos_weight': random.uniform(1,10),
                'lambda_l1': random.uniform(10**(0.01), 10**(0.5)),
                "tree_method": "hist",
                "device": "gpu", 
                'random_state': 1,
                'verbose': -1
                }

            }
        
        else:
            parameters = {
                'num_rounds': random.randint(10, 1000),
                'params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',

                'num_leaves': random.randint(1, 16384),
                'learning_rate':  random.uniform(10**(-2.5), 10**(-1)),
                'lambda_l2': random.uniform(10**(-2), 10**(2)),
                'scale_pos_weight': random.uniform(1,10),
                'lambda_l1': random.uniform(10**(0.01), 10**(0.5)),
                "tree_method": "hist",
                "device": "gpu",
                'random_state': 1,
                'verbose': -1
                }

            }

    #args.model
    elif args.model == 'GINe':

        if not sample_intervals:
            hid_em_size_interval = [16, 72]
            lr_interval = [0.005, 0.05]
            gnn_layer_interval = [2, 4]
            dropout_interval = [0, 0.5]
            w_ce2_interval = [6,8]
        else:
            hid_em_size_interval = sample_intervals.get('hid_em_size_interval')
            lr_interval = sample_intervals.get('lr_interval')
            gnn_layer_interval = sample_intervals.get('gnn_layer_interval')
            dropout_interval = sample_intervals.get('dropout_interval')
            w_ce2_interval = sample_intervals.get('w_ce2_interval')

        parameters = {
            'hidden_embedding_size': random.randint(hid_em_size_interval[0], hid_em_size_interval[1]),
            'learning rate': random.uniform(lr_interval[0], lr_interval[1]),
            'gnn_layers': random.randint(gnn_layer_interval[0], gnn_layer_interval[1]),
            'dropout': random.uniform(dropout_interval[0], dropout_interval[1]),
            'w_ce1': 1.0000182882773443,
            'w_ce2': random.uniform(w_ce2_interval[0], w_ce2_interval[1])
        }

    return parameters






