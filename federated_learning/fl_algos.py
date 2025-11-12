
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
        train_data = feature_engi_regular_data(train_data, self.scaler_encoders)
        vali_data = feature_engi_regular_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, vali_data

    def prep_data(self):
        # if regular or graph, then it should only process once, and this should be skipped
        # need to figure out how
        if self.mode == 'tuning':
            splits = ('train_data', 'vali_data')
            data_1, data_2 = self.data['train_data'], self.data['vali_data']
        elif self.mode == 'training':
            splits = ('train_data', 'test_data')
            data_1, data_2 = self._prep_helper(), self.data['test_data']

        processed_1, processed_2 = self.feature_engineering(data_1, data_2)
        self.procs_data = dict(zip(splits, (processed_1, processed_2)))

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
        self._num_features = feature_engi_regular_data(self.parties[self._smallest_bank].data['train_data'], self.parties[self._smallest_bank].scaler_encoders)['x'].shape[1]
        
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




# ----------------------------------------------------------------------------------------------------

from training.gnn_utils import gnn_m
import training.hyperparams as tune_u
from federated_learning.registry import GNN_REGISTRY
import configs.configs as configs
from federated_learning.inference import metrics
import copy
import federated_learning.fl_training as fl_tr
import federated_learning.inference as flin
from federated_learning.gnn_model import GNN


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


    def feature_engineering(self, train_data, eval_data):
        # I need to double check that it process the data right, that it makes a copy etc.
        train_data = feature_engi_graph_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        eval_data = feature_engi_graph_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, eval_data

    def prep_data(self):
        if self.mode == 'tuning':
            train_proc, eval_proc = self.feature_engineering(self.data['train_data'], 
                                                             self.data['vali_data'])
        elif self.mode == 'training':
            train_proc, eval_proc = self.feature_engineering(self.data['vali_data'], 
                                                             self.data['test_data'])
            
        self.procs_data = {'train_data': train_proc, 'eval_data': eval_proc}
    

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
        return {'gd_data': self.procs_data['eval_data']['df'], 
                'pred_indices': self.procs_data['eval_data']['pred_indices']}
    



class GNNMixingManager:

    def init_hyperparams(self, sample_intervals = None):
        x_0 = fl_utils.get_tuning_configs(self.args)['individual_banks']['small']['x_0']
        hp_list = [fl_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    
    def init_models(self, hyperparams, bank_id = None):

        sample_party = next(iter(self.parties.values()))
        node_features = sample_party.procs_data['train_data']['df'].x.shape[1]
        edge_dim = sample_party.procs_data['train_data']['df'].edge_attr.shape[1]
   
        if bank_id:
            self.parties[bank_id].model = GNN(self, hyperparams, node_features, edge_dim)
        else:
            # when init here, one also have to assign whether the party needs batching or not
            for bank_id, party in self.parties.items():
                party.model = GNN(self, hyperparams, node_features, edge_dim)

        # Currently no register for gnn models
        # should potentially make that, just as there is for reg

        # need to 'save' parameters from model 1 as starting, or not, because the models will vary
        # Here get the relevant parameters etc. or it can be placed in get_global_w

    def _get_relevant_parameters(self):

        self.params_update = []
        _, bank_0 = next(iter(self.parties.items()))

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


    def _gnn_tuning(self, laundering_values, **kwargs):

        hyperparameters_tuning = self.init_hyperparams()

        _, scores = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)
        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        hyperparameters_tuning = self.init_hyperparams(sample_space)
        results, _ = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

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
            intervals['hid_em_size_interval'] = tune_u.update_interval(params['hidden_embedding_size'], intervals['hid_em_size_interval'])
            intervals['lr_interval'] = tune_u.update_interval(params['learning rate'], intervals['lr_interval'])
            intervals['gnn_layer_interval'] = tune_u.update_interval(params['gnn_layers'], intervals['gnn_layer_interval'])
            intervals['dropout_interval'] = tune_u.update_interval(params['dropout'], intervals['dropout_interval'])
            intervals['w_ce2_interval'] = tune_u.update_interval(params['w_ce2'], intervals['w_ce2_interval'])

        return intervals



# still need to add these two to the registry

class FLGNNManager(GNNMixingManager):

    def tuning(self, laundering_values):

        # when tuning / training gnn I need to remember to test 4 different seeds

        # --------------------------------

        for bank_id, party in self.parties.items():
            party.prep_data()

        tuned_hp = self._gnn_tuning(laundering_values)

        return tuned_hp

    def tuning_loop(self, hyperparameters_tuning, laundering_values):

        best_f1 = -1
        best_hyperparameters = None
        best_w = None
        best_metrics = None
        best_preditcions = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams)
            self.get_global_w()
            self.send_global_w_params()

            # if reg or graph epochs is used. Or also is for decision trees, yes?
            # just update in one, and then another for sending to manager?
            results = fl_tr.fl_training(self, laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_w = results['w']
                best_metrics = results['metrics']
                best_preditcions = copy.deepcopy(results['laundering_values'])
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])

        return [{'best_hyperparameters': best_hyperparameters, 
                'best_w': best_w, 'best_metrics': best_metrics,
                'best_preditcions': best_preditcions}, scores]
    
    
    def train(self, tuned_values, laundering_values):

        self.set_mode('training')

        # init -------------
        hyper_parameteres_training = tuned_values['best_hyperparameters']

        # initiate models: this would be here the, hyperparameters would be used
        self.init_models(hyper_parameteres_training)
        self.get_global_w()

        # here one should do such that the parties find the data they will be using
        # training set initial parameters and get relevant data
        self.send_global_w_params()

        # One could keep the processed data outside the parties in a dict
        # But for fl settings, they probably need to hold it themselves anyway.
        # need this to be training, but tuning data
        for bank_id, party in self.parties.items():
            party.prep_data()

        results = fl_tr.fl_training(self, laundering_values)

        return results
    
    def fl_training(self, laundering_values):

        # maybe include hyperparameters here, like as input to the function here

        best_w = None
        best_metrics = None
        best_preditcions = None
        best_f1 = -1

        epochs = 20 if self.args['data_parser'].testing else configs.epochs_fl

        for i in range(0, epochs):

            for bank_id, party in self.parties.items():    
                # little unsure if I should update parameters like this, or if 
                # it should be kept inside the party

                party.update_local_w()
                party.send_local_w(self)
            
            self.update_global_w()
            self.send_global_w()

            # I need to reset laundering_values or something every time new parameters are tested
            # inference / status
            if (i+1) % 20 == 0:

                # reset preditcions
                laundering_values['predictions_fl'] = 0
                
                for bank_id, party in self.parties.items():
                    flin.update_laundering_values(party, laundering_values)

                tmp_metrics = metrics(laundering_values['true_y'], laundering_values['predictions_fl'])

                if tmp_metrics['f1'] > best_f1:
                    best_w = self.global_w
                    best_metrics = tmp_metrics
                    best_preditcions = copy.copy(laundering_values['predictions_fl'])
                    best_f1 = tmp_metrics['f1']
                    
        laundering_values['predictions_fl'] = best_preditcions

        return {'w': best_w, 'metrics': best_metrics, 'laundering_values': laundering_values}
    












class IndividualGNNManager(GNNMixingManager):

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_eval_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def tuning(self, laundering_values):

        results = {}
        for bank_id, party in self.parties.items():
            party.prep_data()
            party_laundering_values = self._helper_party_tuning(party, laundering_values)
            results[bank_id] = self._gnn_tuning(party_laundering_values, bank_id = bank_id)
            
        return results

    def tuning_loop(self, hyperparameters_tuning, party_laundering_values, bank_id):

        best_f1 = -1
        best_hyperparameters = None
        best_metrics = None
        best_preditcions = None
        scores = []

        # preferable this should be a 'reuseable loop' for all the models
        for hyperparams in hyperparameters_tuning:
            
            # initiate models:
            self.init_models(hyperparams, bank_id)

            # if reg or graph epochs is used. Or also is for decision trees, yes?
            # just update in one, and then another for sending to manager?
            results = self.party_train(self.parties[bank_id], party_laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_metrics = results['metrics']
                best_preditcions = copy.deepcopy(results['laundering_values'])
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])

        return [{'best_hyperparameters': best_hyperparameters, 
                'best_metrics': best_metrics, 'best_preditcions': best_preditcions}, scores]
    

    def party_train(self, party, party_laundering_values):

        best_metrics = None
        best_preditcions = None
        best_f1 = -1
        best_model = None

        epochs = 20 if self.args['data_parser'].testing else configs.epochs_fl

        for i in range(0, epochs):
            party.update_local_w()

            if (i+1) % 20 == 0:

                party_laundering_values['predictions_fl'] = 0

                predictions = party.model.predict_binary(party.get_eval_data())
                tmp_metrics = metrics(party_laundering_values['true_y'], predictions)
                
                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_preditcions = predictions
                    best_f1 = tmp_metrics['f1']
                    best_model = copy.deepcopy(party.model.gnn.state_dict())

        party_laundering_values['predictions_fl'] = best_preditcions

        if best_model is not None:
            party.model.gnn.load_state_dict(best_model)
        
        return {'metrics': best_metrics, 'laundering_values': party_laundering_values}
    

    def _train(self, tuned_values, laundering_values):

        self.set_mode('training')
        results = {}

        for bank_id, party in self.parties.items():
            party.prep_data()
            party_laundering_values = self._helper_party_tuning(party, laundering_values)
            self.init_models(tuned_values[bank_id]['best_hyperparameters'], bank_id)
            results[bank_id] = self.party_train(self.parties[bank_id], party_laundering_values)

        for bank_id, party in self.parties.items():
            flin.update_laundering_values(party, laundering_values)

        return results
    
    def train(self, tuned_values, laundering_values):

        self._train(tuned_values, laundering_values)
        self.inference()
    
    def inference():

        for bank_id, party in self.parties.items():
            flin.update_laundering_values(party, laundering_values)
        

        return 0
        
        



    
# ------------------------------------------------------------------------------------------


# Booster -------------------------------------------------------------------------------------------


class BoosterMixingParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class BoosterMixingManager:


    def init_hyperparams(self):
        pass

    def _get_relevant_parameters(self):
        pass



class FLBoosterManager(BoosterMixingManager):

    def tuning_loop(self):
        pass


    def tuning(self):
        # for boost it would want it inside the hyperparameter loop.
        # however, if spliting and processing several datapoints at once, then
        # maybe it could be avoided, however too much memory usage?
        # So don't do that, find solution to only do once for reg and gnn?
        # will see at booster trees
        pass


class IndividualBoosterManager(BoosterMixingManager):

    def tuning(self):
        pass

    def tuning_loop(self):
        pass




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

# Party --------------------------

@regi_algo_party("FedGD_gnn")
class FedGD_GNN_Party(GNNMixingParty, FedGD_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FedGD_GNN_Party(**kwargs)


@regi_algo_party("individual_gnn")
class Individual_GNN_Party(GNNMixingParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return Individual_GNN_Party(**kwargs)



# Manager --------------------------

@regi_algo_manager("FedGD_gnn")
class FedGD_GNN_Manager(FLGNNManager, FedGD_manager): #FLGNNManager #GNNMixingManager

    @staticmethod
    def return_class(args):
        return FedGD_GNN_Manager(args)


@regi_algo_manager("individual_gnn")
class Individual_GNN_Manager(IndividualGNNManager, Manager): #FLGNNManager #GNNMixingManager

    @staticmethod
    def return_class(args):
        return Individual_GNN_Manager(args)



