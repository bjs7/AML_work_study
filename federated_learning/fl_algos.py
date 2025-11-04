
# packages
import utils
import logging
import argparse
import warnings
import pandas as pd
from data.feature_engi import feature_engi_regular_data, feature_engi_graph_data

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
from data.get_indices_type_data import get_booster_data
import torch


import numpy as np

from federated_learning.fl_base import Manager, Party



# -------------------------------------------
# FL Algos ----------------------------------
# -------------------------------------------

# FedAVG ------------------------------------------------------




# FedGD -------------------------------------------------------

# this is just for FedGD, but right now I have built for fedavg?
class FedGD_party(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)
    

class FedGD_manager(Manager):

    def get_adjacency_matrix(self):
        return 0


# ------------------------------------------------------------------------------------------


# -------------------------------------------
# Mixer classes -----------------------------
# -------------------------------------------

# Regression ---------------------------------------------------------------

class RegressionMixinParty:

    def feature_engineering(self, train_data, vali_data):

        # I need to double check that it process the data right, that it makes a copy etc.
        train_data = feature_engi_regular_data(train_data, self.scalar_encoders)
        vali_data = feature_engi_regular_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, vali_data

    def prep_data_tuning(self):
        # if regular or graph, then it should only process once, and this should be skipped
        # need to figure out how
        tr_processed, val_processed = self.feature_engineering(self.data['train_data'], self.data['vali_data'])
        self.procs_data = {'train_data': tr_processed, 'vali_data': val_processed}

    def prep_data_training(self):
        
        tr_data = self._prep_helper()
        tr_processed, te_processed = self.feature_engineering(tr_data, self.data['test_data'])
        self.procs_data = {'train_data': tr_processed, 'test_data': te_processed}

    def _prep_helper(self):

        train_data = {
            'x': pd.concat([self.data['train_data']['x'], self.data['vali_data']['x']]).reset_index(drop=True),
            'y': np.concatenate([self.data['train_data']['y'], self.data['vali_data']['y']])
            }

        return train_data
    
    def update_local_w(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']

        # can add FL-specifics here

        for epoch in range(num_local_epochs):
            self.model.update_w(tr_data['x'], tr_data['y'])

    def send_local_w(self, manager):
        manager.parties_w[self.bank_id] = self.model.current_w
    
    def get_eval_data(self):
        return self.procs_data['vali_data']['x'] if self.mode == 'tuning' else self.procs_data['test_data']['x']
    




# regression might also be viable for boosting trees. Will see
class RegressionMixinManager:

    MODEL_REGISTRY = FL_REG_MODEL_REGISTRY

    def _get_num_features(self):
        self._num_features = feature_engi_regular_data(self.parties[self._smallest_bank].data['train_data'], self.parties[self._smallest_bank].scalar_encoders)['x'].shape[1]
        
    def init_hyperparams(self):

        self._get_num_features()
        self._init_w = np.random.normal(size = self._num_features)
        hp_list = [{'params': {'eta': ele}} for ele in [0.1, 0.05, 0.05/2, 0.01, 0.01/2]]

        return hp_list
    
    def get_global_w(self):
        self.global_w = self._init_w
    
    def _get_reg_model_class(self):
        
        #using a registery
        model_name = self.args['fl_parser'].model
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown algo type: {model_name}")
        return self.MODEL_REGISTRY[model_name]
        

    def init_models(self, hyperparams):

        # need to add something such that this only run once
        # or maybe keep, such that parameters are sure to reset

        # for now eta is fixed. Also, could add such that global loss is 
        # obtained as the sum of all losses. Then this could use to continuesly
        # optimize on eta to get better values of it
        
        model = self._get_reg_model_class()
        #models = {bank_id: model(hyperparams['params']['eta']) for bank_id in self.parties.keys()}
        for bank_id, party in self.parties.items():
            party.model = model(hyperparams['params']['eta'])
            #party.model = models[bank_id]
    
    def update_global_w(self):
        self.global_w = sum(self.parties_w.values()) / len(self.parties_w)

    # tuning for regular
    def tuning(self, laundering_values_vali):

        # probably need to first init hyparameters here
        hyperparameters_tuning = self.init_hyperparams()

        # for boost it would want it inside the hyperparameter loop.
        # however, if spliting and processing several datapoints at once, then
        # maybe it could be avoided, however too much memory usage?
        # So don't do that, find solution to only do once for reg and gnn?
        # will see at booster trees

        # preferable this should be a 'reuseable loop' for all the models
        # potentially this could be placed inside tuning_loop, though that
        # might mostly be relevant for boost trees

        # --------------------------------
        for bank_id, party in self.parties.items():
            party.prep_data_tuning()
        
        results, _ = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)

        return results    


# GNN -------------------------------------------------------------------


class GNNMixingParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._get_batch_configs()

    def _get_batch_configs(self):

        # uses test data to decide whether batching or not, as should be consistent and be the same for train, validation and testing. 
        # Should check for research on batch trained networks generalizing to full sample.
        #graph density should potentially be incorporated here and adjuset/set batch size, num neighbos based on that. #num_neighbors = [20, 20, 10, 5] #num_neighbors = [20, 15, 5, 5]
        if self.data['test_data']['df'].num_nodes >= 250e3:
            self.tr_configs['num_neighbors'] = [5, 4, 3, 2]
            self.tr_configs['batch_size'] = 2048 #2048 #4096 8192

        elif self.data['test_data']['df'].num_nodes >= 100e3:
            self.tr_configs['num_neighbors'] = [5, 4, 3, 2]
            self.tr_configs['batch_size'] = 1024 #512  

        else:
            self.tr_configs['num_neighbors'] = None
            self.tr_configs['batch_size'] = 0


    def prep_data_tuning(self):
        # if regular or graph, then it should only process once, and this should be skipped
        # need to figure out how
        tr_processed, val_processed = self.feature_engineering(self.data['train_data'], self.data['vali_data'])
        self.procs_data = {'train_data': tr_processed, 'vali_data': val_processed}


    def prep_data_training(self):
        
        #tr_data = self._prep_helper()
        tr_processed, te_processed = self.feature_engineering(self.data['vali_data'], self.data['test_data'])
        self.procs_data = {'train_data': tr_processed, 'test_data': te_processed}


    def feature_engineering(self, train_data, vali_data):

        # I need to double check that it process the data right, that it makes a copy etc.
        train_data = feature_engi_graph_data(train_data, self.args['gnn_parser'], self.scalar_encoders)
        vali_data = feature_engi_graph_data(vali_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

        return self._adjust_data_dimension(train_data, vali_data)


    def _adjust_data_dimension(self, tr_data, val_test_data):

        m_settings = fl_utils.get_tuning_configs(self.manager.args).get('model_settings')
        columns_to_drop = []
        
        if not self.tr_configs['batch_size']:
            columns_to_drop.append(0)
        if not m_settings['include_time']:
            columns_to_drop.append(1)
        
        # potentially combine/use _get_num_features here, 
        # could be better to combine/use those such that it is juts once

        num_cols = tr_data['df'].edge_attr.shape[1]
        mask = torch.ones(num_cols, dtype=bool)
        mask[columns_to_drop] = False

        tr_data['df'].edge_attr = tr_data['df'].edge_attr[:, mask]
        val_test_data['df'].edge_attr = val_test_data['df'].edge_attr[:, mask]

        return tr_data, val_test_data
    

    def update_local_w(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']['df']

        # can add FL-specifics here

        for epoch in range(num_local_epochs):
            self.model.update_w(tr_data)

    def send_local_w(self, manager):
        # in theory this functions could be dropped and the manager could just collect the parameters itself
        # though here one would apply some encrypt?
        manager.parties_w[self.bank_id] = {param: value.data.clone() for param, value in self.model.gnn.named_parameters()}
    
    def get_eval_data(self):
        return {'gd_data': self.procs_data['vali_data']['df'], 'pred_indices': self.procs_data['vali_data']['pred_indices']} if self.mode == 'tuning' else {'gd_data': self.procs_data['test_data']['df'], 'pred_indices': self.procs_data['test_data']['pred_indices']}
    
    
        

    


from training.gnn_utils import gnn_m
import training.hyperparams as tune_u


# maybe move get_gnn out of GNN, and into manager 100%, and then send to parties?
class GNN(ABC):

    def __init__(self, manager, hyperparams):
        super().__init__()
        self._get_gnn_loss_optimizer(manager, hyperparams)
        # I could potentially also ahve the data in here, if it is just pointers
        # but keep it out for now

    def _init_model_configs(self):
        return 0

    def _get_gnn_loss_optimizer(self, manager, hyperparams):
        self.gnn = get_gnn(manager, hyperparams)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=hyperparams['params'].get('learning rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparams['params'].get('w_ce1'), hyperparams['params'].get('w_ce2')])) 

    def update_w(self, gd_data):

        self.gnn.train()
        self.optimizer.zero_grad()

        pred = self.gnn(gd_data.x, gd_data.edge_index, gd_data.edge_attr, 
                            gd_data.edge_index, gd_data.edge_attr)
        loss = self.loss_fn(pred, gd_data.y)

        loss.backward()
        self.optimizer.step()
    
    def predict_binary(self, graph_data):

        gd_data = graph_data.get('gd_data')
        pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            #data.to(device)
            pred = self.gnn(gd_data.x, gd_data.edge_index, gd_data.edge_attr, 
                    gd_data.edge_index, gd_data.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred
            
        return pred.argmax(dim=-1)


            


        


        


from federated_learning.registry import GNN_REGISTRY

# place this inside the GNNMxiingManager? Probably. And maybe not actually

# potentially split this up, so that it is in two, like one where parameters are extracted once
# and then used to get the model in the register.
# need to sort how to init about parties and whether they need batching or not, and
# also how to adjust the data set size, such that copy can be avoided.

def get_gnn(manager, hyperparams):

    m_settings = fl_utils.get_tuning_configs(manager.args).get('model_settings')

    e_dim_adjust = 1 if m_settings.get('include_time') else 2
    node_features = manager.parties[manager._smallest_bank].data['train_data']['df'].x.shape[1] # switch between tuning / training?
    e_dim = (manager._num_features - e_dim_adjust)

    # add the GNN_REGISTRY as an attribute to the manager?
    model_name = manager.args['fl_parser'].model
    if model_name not in GNN_REGISTRY:
        raise ValueError(f"Unknown algo type: {model_name}")
    gnn_init = GNN_REGISTRY[model_name]

    arguments = {'num_features': node_features, 'num_gnn_layers': hyperparams['params'].get('gnn_layers'),
                 'n_classes': 2, 'n_hidden': hyperparams['params'].get('hidden_embedding_size'),
                 'residual': False, 'edge_updates': manager.args['gnn_parser'].emlps, 
                 'edge_dim': e_dim, 'dropout': hyperparams['params'].get('dropout'), 
                 'final_dropout': hyperparams['params'].get('dropout')}
    
    gnn = gnn_init(**arguments)
    
    return gnn



class GNNMixingManager:

    #MODEL_REGISTRY = FL_REG_MODEL_REGISTRY

    def test(self):
        return 0
    
    def _get_num_features(self):
        #n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
        self._num_features = feature_engi_graph_data(self.parties[self._smallest_bank].data['train_data'], self.args['gnn_parser'], self.parties[self._smallest_bank].scalar_encoders)['df'].edge_attr.shape[1]

    def init_hyperparams(self, sample_intervals = None):

        # this works, but potentially I should change such that it request from a party how many
        # features are in the end, however, this can wait till more actual fl is implemented

        self._get_num_features()
        x_0 = fl_utils.get_tuning_configs(self.args)['individual_banks']['small']['x_0']
        hp_list = [fl_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    
    def init_models(self, hyperparams):
        
        # when init here, one also have to assign whether the party needs batching or not
        for bank_id, party in self.parties.items():
            #print(bank_id)
            party.model = GNN(self, hyperparams)

        # Currently no register for gnn models
        # should potentially make that, just as there is for reg

        # need to 'save' parameters from model 1 as starting, or not, because the models will vary
        # Here get the relevant parameters etc. or it can be placed in get_global_w

    def _get_relevant_parameters(self):

        # for now only learnable parameters are being shared, using non-learnable
        # can potentially be done once encryption is started to being applied
        
        self.params_update = []
        _, bank_0 = next(iter(self.parties.items()))

        # use learnable only
        for name, param in bank_0.model.gnn.named_parameters():
            self.params_update.append(name)

    def send_global_w(self, condition = None):
        # reconsider if this should be done more similar to update_w, like in terms of indexing etc.
        for bank_id, party in self.parties.items():
            if condition and not condition(bank_id): continue
            for param, value in party.model.gnn.named_parameters():
                value.data = self.global_w[param].data.clone()

    def send_global_w_params(self):
        bank_0_id, bank_0 = next(iter(self.parties.items()))
        condition = lambda bank_id: bank_0_id != bank_id
        self.send_global_w(condition)

    def get_global_w(self):
        self._bank_0, bank_0 = next(iter(self.parties.items()))
        self.global_w = {param: value.data.clone() for param, value in bank_0.model.gnn.named_parameters()}

    def update_global_w(self):

        for bank, weights in self.parties_w.items():
            for param, value in weights.items():
                if bank == self._bank_0:
                    self.global_w[param].data = value.data.clone() / self._num_parties
                else:
                    self.global_w[param].data += value.data.clone() / self._num_parties


    def tuning(self, laundering_values_vali):

        # when tuning / training gnn I need to remember to test 4 different seeds


        # --------------------------------
        
        # for boost it would want it inside the hyperparameter loop.
        # however, if spliting and processing several datapoints at once, then
        # maybe it could be avoided, however too much memory usage?
        # So don't do that, find solution to only do once for reg and gnn?
        # will see at booster trees

        # preferable this should be a 'reuseable loop' for all the models
        for bank_id, party in self.parties.items():
            party.prep_data_tuning()

        # probably need to first init hyparameters here

        # need to do this several times for -gnn
        hyperparameters_tuning = self.init_hyperparams()


        # this part here is only needed for gnn --------------------------

        _, scores = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)
        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        #init_hp = self.init_hyperparams(sample_space)
        #hyperparameters_tuning = init_hp['hp_list']
        hyperparameters_tuning = self.init_hyperparams(sample_space)

        # ------------------

        results, _ = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)

        return results


    def _get_search_space(self, hyperparameters_tuning):

        intervals = {
            'hid_em_size_interval': [1e9, -1e9],
            'lr_interval': [1e9, -1e9],
            'gnn_layer_interval': [1e9, -1e9],
            'dropout_interval': [1e9, -1e9],
            'w_ce2_interval': [1e9, -1e9],
        }

        for params in hyperparameters_tuning:
            p = params['params']
            intervals['hid_em_size_interval'] = tune_u.update_interval(p['hidden_embedding_size'], intervals['hid_em_size_interval'])
            intervals['lr_interval'] = tune_u.update_interval(p['learning rate'], intervals['lr_interval'])
            intervals['gnn_layer_interval'] = tune_u.update_interval(p['gnn_layers'], intervals['gnn_layer_interval'])
            intervals['dropout_interval'] = tune_u.update_interval(p['dropout'], intervals['dropout_interval'])
            intervals['w_ce2_interval'] = tune_u.update_interval(p['w_ce2'], intervals['w_ce2_interval'])

        return intervals



# ------------------------------------------------------------------------------------------


# -------------------------------------------
# Combiners ---------------------------------
# -------------------------------------------

# Regression ---------------------------------------------------------------

@regi_algo_party('FedGD_regression')
class FedGD_Regression_Party(RegressionMixinParty, FedGD_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FedGD_Regression_Party(**kwargs)


@regi_algo_manager("FedGD_regression")
class FedGD_Regression_Manager(RegressionMixinManager, FedGD_manager):

    @staticmethod
    def return_class(args):
        return FedGD_Regression_Manager(args)
    

# GNN ---------------------------------------------------------------------

@regi_algo_party('FedGD_graph')
class FedGD_GNN_Party(GNNMixingParty, FedGD_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FedGD_GNN_Party(**kwargs)


@regi_algo_manager("FedGD_graph")
class FedGD_GNN_Manager(GNNMixingManager, FedGD_manager):

    @staticmethod
    def return_class(args):
        return FedGD_GNN_Manager(args)




