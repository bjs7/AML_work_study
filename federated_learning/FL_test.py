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

# setup -----------------------------------------------

parsers = fl_utils.parser_all()
fl_parser, data_parser, gnn_parser = parsers['fl_parser'], parsers['data_parser'], parsers['gnn_parser']
parsers['data_parser'].testing = True

utils.logger_setup()
utils.set_seed(data_parser.seed, True)


# -------------

# I would like to change to gnn instead of graph, but for now need to stick
# to graph, because too much to change

# for graph. just tmp

parsers['fl_parser'].model_type = 'graph'
parsers['fl_parser'].model = 'GINe'
parsers['fl_parser'].scenario = 'individual_banks'
parsers['fl_parser'].data_type = 'graph_data'
parsers['data_parser'].data_type = 'graph_data'
parsers['gnn_parser'].scenario = 'individual_banks'


# -------------

# Get data ---------------------------------------------------------------------------------------
df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{data_parser.size}' + f'_{data_parser.ir}' + '.csv')
raw_data = get_data(df, fl_parser, split_perc = split_perc)
scalar_encoders = dfn.extract_enc_cats(df)
fr_banks, sr_banks = get_relevant_banks(data_parser)
# bank 7 does not have any 1's in validation set, therefore might wanna remove it? But
# then would also be need for first calculations? Like full/individual calculations
relevant_banks = fr_banks #+ sr_banks
relevant_banks = relevant_banks[0:4] + relevant_banks[5:10]


# WRITE LOGISTIC AS MATRIX / CONVEX FUNCTION FORM?


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


# 39735 is max number of nodes, with 271928 edges, 371380 edges, and 452751 edges, bank 68
# Get the parties, and their data
for bank in relevant_banks:

    bank_indices = get_indices_bdt(raw_data, bank=bank)
    train_data, vali_data, test_data = dfn.fl_get_data(parsers, raw_data, bank_indices)

    smallest_bank, smallest_dim = fl_utils.get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim)

    tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
    Party.get_algo_class(parsers = parsers, bank_id=bank, data=tmp_data, 
                         indices=bank_indices, manager=manager, 
                         scalar_encoders=scalar_encoders)


manager._num_parties = len(manager.parties)
manager._smallest_bank = smallest_bank



# -------------------------------------------------------------------------------------------------

manager.set_mode('tuning')
manager.mode

self = manager

laundering_values = laundering_values_vali

# -------------------------------------------------------------------------------


tuned_values = manager.tuning(laundering_values_vali)



# after tuning, there should be 'cleaned out' in the different variables, dataframe, etc.
# in order to save space
# and just to make sure the stuff from tuning that shouldn't be carried over, isn't carried over
# reset global_w etc.

# tuning for gnn
manager.set_mode('tuning')



# -------------------------------------------------------------------------------------------------
# training --------------------------------------------------------

# when training gnn, I also need to "seed", 4 or 5 different seeds and then pick the best model

manager.set_mode('training')
manager.mode

# training
laundering_values = laundering_values_test
#hf.fl_training(Manager, laundering_values)

results = manager.train(tuned_values, laundering_values)


list(results)

results['w']


type(manager.parties[0].model.gnn.state_dict())


# save the model, and results, also inference

import configs.configs as config
import os
import json

folder_name = f'bank_{manager.parties[0].bank_id}'

#save_direc = config.save_direc_training
save_direc = os.path.join(config.save_direc_training, manager.args['fl_parser'].model)
save_direc = os.path.join(save_direc, manager.args['data_parser'].size + '_' + manager.args['data_parser'].ir)

# might not need to attached split_perc to parser?
str_folder = f'split_{config.split_perc[0]}_{config.split_perc[1]}__'

# Need to include time?
m_settings = fl_utils.get_tuning_configs(manager.args)

if manager.args['fl_parser'].model_type == 'graph':

    empls_boolean = manager.args['gnn_parser'].emlps
    m_settings['emlps'] = empls_boolean
    mask_indexing, transforming = m_settings.get('model_settings').get('index_masking'), m_settings.get('model_settings').get('transforming_of_time')
    str_folder += f'EU_{empls_boolean}__' + f'transforming_of_time_{transforming}__' + f'mask_indexing_{mask_indexing}'


elif manager.args['fl_parser'].model_type == 'booster':
    x_0_fi, r_0_fi = m_settings.get('full_info').get(manager.args['data_parser'].size).get('x_0'), m_settings.get('full_info').get(manager.args['data_parser'].size).get('r_0')


save_direc = os.path.join(save_direc, str_folder)
if manager.args['fl_parser'].scenario == 'full_info':
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