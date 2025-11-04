
import os
import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_functions as data_funcs
import tuning as tune
import train_models as tr_models
import inference_saving.save_load_models as slm
import xgboost as xgb
import numpy as np

import json
import inference_saving.inference_functions as pu
from sklearn.metrics import f1_score
import torch
from relevant_banks import get_relevant_banks

import trainer_utils as tu
import process_data_type as pdt


# --------------------------------------------------------------------------------------------------------------------------------------

def get_parser_performance():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--specific_banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    
    return parser

# --------------------------------------------------------------------------------------------------------------------------------------
# Setup

parser = get_parser_performance()
args, unknown = parser.parse_known_args()

df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
raw_data = dp.get_data(df, split_perc = configs.split_perc)


# Create dataframe for true values, full info, individual banks etc. --------------------------------------------------
args.model = 'xgboost'
bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = None)
train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
laundering_values = pd.DataFrame( {'indices': bank_indices['test_indices'], 
                                   'true y': test_data['x']['Is Laundering'],
                                   'predicted_full_info': test_data['x']['Is Laundering'].shape[0] * [None],
                                   'predicted_individual_banks': test_data['x']['Is Laundering'].shape[0] * [0]
                                   })



# --------------------------------------------------------------------------------------------------------------------------------------
# Full info

args, unknown = parser.parse_known_args()

# get folders
#args.model = 'xgboost'
args.model = 'GINe'
args.scenario = 'individual_banks'
bank = 5
main_folder, model_settings = pu.get_main_folder(args)
#tmp_folder, model_parameters = pu.get_indi_para(main_folder)
tmp_folder, model_parameters = pu.get_indi_para(main_folder, bank=5)


# get bank indices, data and apply feature engineering to data
#test_data = pu.get_indices_feat_data(args, raw_data)
test_data, test_indices = pu.get_indices_feat_data(args, raw_data, bank = 5)

# Import the model
model = pu.get_model(args, test_data, model_parameters, model_settings, tmp_folder)

# make predictions
predictions, f1_values = pu.get_predictions(args, model, test_data, model_settings, tmp_folder)

#laundering_values['predicted_full_info'] = predictions

# --------------------------------------------------------------------------------------------------------------------------------------
# Individual banks

#/home/nam_07/miniconda3/envs/multignn/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1757: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no true nor predicted samples. Use `zero_division` parameter to control this behavior.
#  _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

# relevant indices where y = 1
relevant_indices = laundering_values.iloc[np.where(laundering_values['true y'] == 1)[0],:]['indices']

# Get relevant banks
fr_banks, sr_banks = get_relevant_banks(args)
banks = fr_banks + sr_banks
all_test_indices = []
for bank in banks:
    bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
    tmp_indices =  bank_indices.get('test_indices')
    all_test_indices += tmp_indices

laund_indices = len(set(all_test_indices) & set(relevant_indices))
amount_launderings = len(relevant_indices)
percent = laund_indices / amount_launderings
print(f'{laund_indices} observations with laundering in test split out of {amount_launderings}, {percent}')

bank = 5
#fr_banks
#banks
for bank in [0,2,4,5,7]:
    main_folder, model_settings = pu.get_main_folder(args)
    tmp_folder, model_parameters = pu.get_indi_para(main_folder, bank=bank)

    if not model_parameters:
        continue

    # get data
    test_data, test_indices = pu.get_indices_feat_data(args, raw_data, bank = bank)

    # Import the model
    model = pu.get_model(args, test_data, model_parameters, model_settings, tmp_folder)

    # make predictions
    predictions, f1_values = pu.get_predictions(args, model, test_data, model_settings, tmp_folder, f1_values)

    # create dataframe with predictions and indices    
    tmp_preidctions = pd.DataFrame({'original_indices': test_indices, 'predictions': predictions})

    # join with main dataframe
    indices_to_update = tmp_preidctions['original_indices'][np.where(tmp_preidctions['predictions'] == 1)[0]]
    laundering_values.loc[indices_to_update,'predicted_individual_banks'] = 1

    

"""
pred3 = model(test_data['df'].x, 
            test_data['df'].edge_index, 
            test_data['df'].edge_attr, 
            test_data['df'].edge_index, 
            test_data['df'].edge_attr, 
            index_mask = False)

pred4 = model(test_data['df'].x, 
            test_data['df'].edge_index, 
            test_data['df'].edge_attr, 
            test_data['df'].edge_index, 
            test_data['df'].edge_attr, 
            index_mask = True)
"""