# packages
import utils
import logging
import argparse
import pandas as pd
from models.base import Model
from data.raw_data_processing import get_data
from configs.configs import split_perc
from relevant_banks import get_relevant_banks
from relevant_banks import load_relevant_banks
import xgboost as xgb
from data.feature_engi import feature_engi_regular_data

# packages for FL
import threading
import queue
import time
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import federated_learning.FL_utils as fl_utils
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
from data.get_indices_type_data import get_indices_bdt
import numpy as np
from models.base import Model, InferenceModel
import copy
import training.gnn_utils as gu
import models.gnn_models as gnn_m
import training.hyperparams as tune_u
import federated_learning.data_functions as dfn

import federated_learning.models

from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos

# remove files once no longer needed?
# check om der skal der skal samples 2 droprates?
# residuals i gnn model?

# still missing to put a lot of the stuff / data on CUDA

# setup -----------------------------------------------

parsers = fl_utils.parser_all()
fl_parser, data_parser, gnn_parser = parsers['fl_parser'], parsers['data_parser'], parsers['gnn_parser']
parsers['data_parser'].testing = True

utils.logger_setup()
utils.set_seed(data_parser.seed, True)


# -------------

# need to check that individual saves predictions/laundering values correct

parsers['fl_parser'].fl_algo = 'individual'

# -------------

# Get data ---------------------------------------------------------------------------------------
# is it actually necessary to carry bank indices with the graph data? Or can one just use only graph data
# for gnn models? Think I can, or only have the indices with one, reset of columns can be dropped
df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{data_parser.size}' + f'_{data_parser.ir}' + '.csv')
raw_data = get_data(df, data_parser, split_perc = split_perc)
scaler_encoders = dfn.extract_enc_cats(df)


# check get_relevant_banks functions etc. to make sure everything works as intended
fr_banks, sr_banks = get_relevant_banks(data_parser)
# bank 7 does not have any 1's in validation set, therefore might wanna remove it? But
# then would also be need for first calculations? Like full/individual calculations
relevant_banks = fr_banks #+ sr_banks
relevant_banks = relevant_banks[0:4] + relevant_banks[5:10]


# ------------------------------------------------------------------------------------------------
# laundering values ------------------------------ 
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(data_parser, copy.deepcopy(raw_data))

# -------------------------------------------------------------------------------------------------
# Setup for manager and parties ------------------------------------------------------

# fl setup --------------------------

# Get the manager
manager = Manager.get_algo_class(parsers)
smallest_bank = None
smallest_dim = 1e10



# one can split the GNNMixingParty up, or keep it, but part of it into
# seperate parts, one for FL and one for individual

# there are several functions that could be simplified. Check from top to bottom to find them

# still missing to include more of the features, like sent/received, currency and amount.
# though need to figure out how to deal with it, when applying feature engineering


# optimize, fl_get_data function and the two functions that it uses?

# 39735 is max number of nodes, with 271928 edges, 371380 edges, and 452751 edges, bank 68
# Get the parties, and their data
for bank in relevant_banks:

    bank_indices = get_indices_bdt(raw_data, bank=bank)
    train_data, vali_data, test_data = dfn.fl_get_data(parsers, raw_data, bank_indices)
    tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
    Party.get_algo_class(parsers = parsers, bank_id=bank, data=tmp_data, 
                         indices=bank_indices, manager=manager, 
                         scaler_encoders=scaler_encoders)


manager._num_parties = len(manager.parties)

#smallest_bank, smallest_dim = fl_utils.get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim)
#manager._smallest_bank = smallest_bank

# check that up_data / update_nodes is not used after feature engineering
# Need to check how to "extract" the right indices from GNN when batching. 
# Does having target edges make sense?

# -------------------------------------------------------------------------------------------------

# after tuning, there should be 'cleaned out' in the different variables, dataframe, etc.
# in order to save space
# and just to make sure the stuff from tuning that shouldn't be carried over, isn't carried over
# reset global_w etc.

# design for no FL
manager.set_mode('tuning')
self = manager
manager.mode

laundering_values = laundering_values_vali
tuned_hp = manager.tuning(laundering_values_vali)


#manager.__mro__


# train
laundering_values = laundering_values_test
self.set_mode('training')

hyperparameters = tuned_hp[14]
party = manager.parties[14]

party.model
manager.parties[0].model


results = self.train(tuned_hp, laundering_values_test)


# once training is done for individual, one needs to combine values and run inference


# -------------------------------------------------------------------------------------------------

manager.parties[14].model.gnn.state_dict()
#save_direc = config.save_direc_training
#save_direc = os.path.join(config.save_direc_training, manager.args['fl_parser'].model)


# might not need to attached split_perc to parser?

import configs.configs as config
import os
import json
from pathlib import Path
import torch
import pickle


def save_results(results, hyperparams, manager):

    str_testing = 'testing' if manager.args['data_parser'].testing else ''
    save_direc = os.path.join(config.save_direc_training, str_testing,
                            manager.args['data_parser'].size + '_' + manager.args['data_parser'].ir,
                            f'split_{config.split_perc[0]}_{config.split_perc[1]}',
                            manager.args['fl_parser'].fl_algo)

    str_folder = manager.args['fl_parser'].model
    model_tuning_configs = fl_utils.get_tuning_configs(manager.args).get(manager.args['data_parser'].scenario)

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

    with open(save_direc + '/metrics.pkl', 'wb') as f:
        pickle.dump({'metrics': results['metrics']}, f)

    with open(save_direc + '/models_hyperparameters.pkl', 'wb') as f:
        pickle.dump({'models_hyperparameters': results['models']}, f)








# need to get the right save direction. Need to find out if it should be saved pth or something else
# and need to check that I can load the parameters
# I am not saving the whole













# Load back
with open(save_direc + '/metrics_laundering_values.pkl', 'rb') as f:
    data = pickle.load(f)
    test1 = data['metrics']
    test2 = data['laundering_values']







file_path = folder_path / 'laundering_values.json'
with open(file_path, 'w') as f:
    json.dump(results, f)



gnn_params = torch.load('test.pth')
checkpoint = torch.load('test.pth')



gnn_params = {name: param.data for name, param in manager.parties[0].model.gnn.named_parameters()}
torch.save(gnn_params, 'gnn_only.pth')

# Loading them back
gnn_params = torch.load('gnn_only.pth')
gnn.load_state_dict(gnn_params, strict=False)


list(test_values['w'])

type(test_values['w'])

type(manager.parties[0].model.gnn.state_dict())

manager.parties[0].model.gnn.load_state_dict(test_values['w'])

list(manager.parties[0].model.gnn.state_dict())

names = []
for t, params in manager.parties[0].model.gnn.named_parameters():
    names.append(t)



manager.parties[14].get_eval_data()
manager.parties[14].get_eval_indices()



for key, value in results.items():
    print(key)


for bank_id, party in manager.parties.items():
    print(bank_id)












folder_name = f'bank_{manager.parties[0].bank_id}'



save_direc = os.path.join(save_direc, str_folder)
if manager.args['fl_parser'].fl_algo == 'full_info':
    folder_path = Path(save_direc)
    file_path = folder_path / 'model_settings.json'
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(m_settings, indent=4))




save_direc = os.path.join(save_direc, folder_name)

if not os.path.exists(save_direc):
    os.makedirs(save_direc, exist_ok=True)


file_name = 1

# save the model
file_type = utils.file_types.get(manager.args['fl_parser'].model_type)
file_name = os.path.join(save_direc, file_name + f'.{file_type}')
if manager.args['fl_parser'].model_type == 'graph':
    #torch.save(model.state_dict(), file_name)
    torch.save(model, file_name)
elif manager.args['fl_parser'].model_type == 'booster':
    #joblib.dump({"model": model, "scaler": scaler}, file_name) if scaler is not None else model.save_model(file_name)
    model.save_model(file_name)
if q == 1:
    file_path = os.path.join(save_direc, 'hyper_parameters.json')
    with open(file_path, 'w') as file:
        json.dump(hyper_params, file, indent=4)



    


    






from functools import singledispatch

@singledispatch
def select_data_for_feat_engi(tr_data, val_data, tune_values = None):
    raise TypeError("Unsupported type")

@select_data_for_feat_engi.register
def _(x: int):
    return 0
    








# --------------------------------------------

# Start the manager and parties
manager.start()
for bank in relevant_banks:
    manager.parties[bank].start()



# Send initial model etc.


# simple training

#tmpmodel.coef_




# Tuning ------------------------------------------------------















# Training ------------------------------------------------------


# Save model(s) ------------------------------------------------------


# Inference ------------------------------------------------------



# need feature engineering functions etc.




"""


for hyperparams in hyperparameters_tuning:
    
    # initiate models:
    manager.init_models()

    # set initi parameters
    manager.global_w = init_w

    # here one should do such that the parties find the data they will be using
    # training set initial parameters and get relevant data
    manager.send_global_w_params()


    # One could keep the processed data outside the parties in a dict
    # But for fl settings, they probably need to hold it themselves anyway.
    for bank_id, party in manager.parties.items():
        party.prep_data_tuning()

    # if reg or graph epochs is used. Or also is for decision trees, yes?
    # just update in one, and then another for sending to manager?
    # think seperate is better in order to change for FL settings later
    
    for i in range(0, epochs):

        for bank_id, party in parties.items():
            
            # little unsure if I should update parameters like this, or if 
            # it should be kept inside the party
            party.model.update_w(party.procs_data['tr_data']['x'], party.procs_data['tr_data']['y'])
            party.send_local_w(manager)
        
        # manage update global_w
        manager.update_global_w()

        # send global_w
        manager.send_global_w()

        # I need to reset laundering_values_vali or something every time new parameters are tested
        # inference / status
        if (i+1) % 20 == 0:

            # reset preditcions
            laundering_values_vali['predictions_fl'] = 0
            
            for bank_id, party in parties.items():
                update_vali(party, laundering_values_vali)

            tmp_metrics = hf.metrics(laundering_values_vali['true_y'], laundering_values_vali['predictions_fl'])

            if tmp_metrics['f1'] > best_f1:
                best_w = manager.global_w
                best_hyperparameters = hyperparams
                best_matrics = tmp_metrics
                best_f1 = tmp_metrics['f1']
                best_preditcions = laundering_values_vali['predictions_fl']






"""





"""













    @abstractmethod
    def process_messages(self): #_receive_handle
        pass


    def get_data():
        return 0
    
    def process_messages(self):
        return 0
    
    def send_messages(self, recipient: str, content: Any):
        #msg = Message(self.bank_id, recipient, content)
        return 0

    def broadcast(self, content: Any):
        return 0
        #return super().process_messages()

        




"""