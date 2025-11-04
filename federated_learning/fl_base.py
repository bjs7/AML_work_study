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

#from federated_learning.fl_algos import FedGB_Regression_Manager, FedGB_Regression_Party

import federated_learning.fl_training as fl_tr


class BaseFL:
    REGISTRY = None

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = None


# add current_w also to the actual party?
class Party(BaseFL):

    REGISTRY = FL_ALGO_REGISTRY_PARTY

    """Individual computation node that can send/receive messages"""
    
    def __init__(self, args, bank_id, data, indices, manager, scalar_encoders) -> None:
        super().__init__(args)
        self.bank_id = bank_id
        self.inbox = queue.Queue()
        self.peers: Dict[str, 'Party'] = {}
        self.running = False
        self.thread = None
        self.indices = indices
        self.data = data
        self.manager = manager
        # for model, scaler etc. and also global parameters, I might
        # wanna keep it all in a dict, so they are collected together,
        # also it would make dir of manager smaller
        self.scalar_encoders = scalar_encoders
        self.model = None
        self.tr_configs = {}
        
        # for now it is fine that I use train data right away,
        # but I should do such that it is easily adjusted for tuning/training etc.        

        if self.manager:
            self.manager.add_party(self)
        
    @classmethod
    def get_algo_class(cls, parsers: argparse.Namespace, **kwargs) -> classmethod:

        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(args = parsers, **kwargs)

    def get_train_data(self):
        return 0

    def get_eval_indices(self):
        return self.indices['vali_indices'] if self.mode == 'tuning' else self.indices['test_indices']
    
    def get_eval_data(self):
        pass
    


import copy

    
# some of the attributes should be changed to indicate or only be for internal usage
class Manager(BaseFL):
    REGISTRY = FL_ALGO_REGISTRY_MANAGER

    def __init__(self, args):
        super().__init__(args)
        self.parties: Dict[int, Party] = {}
        self.parties_w: Dict[int, Any] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.global_w = None
        # for model, scaler etc. and also global parameters, I might
        # wanna keep it all in a dict, so they are collected together,
        # also it would make dir of manager smaller

    @classmethod
    def get_algo_class(cls, parsers):

        #algo_class = args.fl_algo
        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(parsers)

    def add_party(self, party: Party):
        self.parties[party.bank_id] = party
        print(f"[Manager] added: party {party.bank_id}")

    def set_mode(self, mode):
        self.mode = mode
        for _, party in self.parties.items():
            party.mode = mode

    def update_global_w(self):
        pass
        

    def send_global_w(self):

        if self.args['fl_parser'].model_type == 'regression':
            for bank_id, party in self.parties.items():
                party.model.current_w = self.global_w
        
        if self.args['fl_parser'].model_type == 'graph':
            return 0

    # similar to the one above, but here I also need it to give the relevant hyperparameters
    # to the parties. At least I think, not sure if necessary. Doesn't seem to be for reg, but
    # might be for GNN and decision trees
    def send_global_w_params(self):
        
        if self.args['fl_parser'].model_type == 'regression':
            self.send_global_w()
        
        if self.args['fl_parser'].model_type == 'graph':
            return 0

    def send_parameters(self, parameters):
        for bank_id, party in self.parties.items():
            party.inbox.put(parameters)

    #@abstractmethod
    def init_models(self):
        pass

    def get_global_w(self):
        pass

    def tuning_loop(self, hyperparameters_tuning, laundering_values_vali):

        best_f1 = -1
        best_hyperparameters = None
        best_w = None
        best_metrics = None
        best_preditcions = None
        scores = []

        #hyperparams = hyperparameters_tuning[0]
        # preferable this should be a 'reuseable loop' for all the models
        for hyperparams in hyperparameters_tuning:
            
            # initiate models:
            self.init_models(hyperparams)

            # set initi parameters
            self.get_global_w()
            #manager.global_w = init_w

            # training set initial parameters and get relevant data
            self.send_global_w_params()

            # if reg or graph epochs is used. Or also is for decision trees, yes?
            # just update in one, and then another for sending to manager?
            #laundering_values = laundering_values_vali
            results = fl_tr.fl_training(self, laundering_values_vali)
            
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

        # set initi parameters
        self.get_global_w()

        # here one should do such that the parties find the data they will be using
        # training set initial parameters and get relevant data
        self.send_global_w_params()

        # One could keep the processed data outside the parties in a dict
        # But for fl settings, they probably need to hold it themselves anyway.
        # need this to be training, but tuning data
        for bank_id, party in self.parties.items():
            party.prep_data_training()

        results = fl_tr.fl_training(self, laundering_values)

        return results
        

    
    
        


