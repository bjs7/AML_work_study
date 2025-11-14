# packages
import superseded.utils as utils
import logging
import argparse
import pandas as pd
from models.base import Model
from data.raw_data_processing import get_data
from configs.configs import split_perc
from mix.relevant_banks import get_relevant_banks
from mix.relevant_banks import load_relevant_banks
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
import utils as fl_utils

from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY
from federated_learning.registry import regi_algo_manager, regi_algo_party
from data.get_indices_type_data import get_indices_bdt
from data.get_indices_type_data import get_booster_data

# remove files once no longer needed?

# setup -----------------------------------------------

parsers = fl_utils.parser_all()
fl_args, data_args, gnn_args = parsers['fl_parser'], parsers['data_parser'], parsers['gnn_parser']

utils.logger_setup()
utils.set_seed(data_args.seed, True)

df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{data_args.size}' + f'_{data_args.ir}' + '.csv')
raw_data = get_data(df, fl_args, split_perc = split_perc)
scalar_encoders = fl_utils.extract_enc_cats(df)

fr_banks, sr_banks = get_relevant_banks(data_args)
relevant_banks = fr_banks + sr_banks
relevant_banks = relevant_banks[0:10]


# fl setup --------------------------

@dataclass
class Message:
    """Simple message structure for node communication"""
    sender_id: str
    recipient_id: str
    content: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseFL:
    REGISTRY = None

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abstractmethod
    def send_message(self):
        pass


# add current_w also to the actual party?
class Party(BaseFL):

    REGISTRY = FL_ALGO_REGISTRY_PARTY

    """Individual computation node that can send/receive messages"""
    
    def __init__(self, args, bank_id, data, manager, scalar_encoders) -> None:
        super().__init__(args)
        self.bank_id = bank_id
        self.inbox = queue.Queue()
        self.peers: Dict[str, 'Party'] = {}
        self.running = False
        self.thread = None
        self.data = data
        self.manager = manager
        self.scalar_encoders = scalar_encoders
        #self.model = self.manager.create_model()
        
        # for now it is fine that I use train data right away,
        # but I should do such that it is easily adjusted for tuning/training etc.

        self.tmp_train, tmp_vali = self.feature_engineering(self.data['train_data'], self.data['vali_data'])
        
        self.model = fl_utils.LogReg(self.tmp_train['x'], self.tmp_train['y'])
        #self.data['train_data']['x'].iloc[:, 0:11], 
        #self.data['train_data']['y']

        if self.manager:
            self.manager.add_party(self)
        
    @classmethod
    def get_algo_class(cls, args: argparse.Namespace, bank: int, data: int, manager, scalar_encoders) -> classmethod:
        algo_class = args.fl_algo
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(args, bank, data, manager, scalar_encoders)

    def process_messages(self):
        while self.running:
            try:
                self.model.current_w = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
    
    
    def feature_engineering(self, train_data, vali_data):

        train_data = feature_engi_regular_data(train_data, self.scalar_encoders)
        vali_data = feature_engi_regular_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, vali_data
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.process_messages)
        #self.thread = threading.Thread()
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

        #return super().send_messages()

from functools import singledispatch

class Manager(BaseFL):
    REGISTRY = FL_ALGO_REGISTRY_MANAGER

    def __init__(self, args):
        super().__init__(args)
        self.parties: Dict[int, Party] = {}
        self.parties_weights: Dict[int, Any] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None

        # I need to change this so it picks the 'right' model.
        # For now I just use reg to make things 'work'
        #self.create_model = fl_utils.reg_model
        #self.model = self.create_model()
        #self.create_model = fl_utils.LinearReg(0,1)

    @classmethod
    def get_algo_class(cls, args):
        algo_class = args.fl_algo
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(args)

    def start(self):
        self.running = True
        self.thread = threading.Thread()
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def add_party(self, party: Party):
        self.parties[party.bank_id] = party
        print(f"[Manager] added: party {party.bank_id}")

    def send_parameters(self, parameters):
        for bank_id, party in self.parties.items():
            party.inbox.put(parameters)

    def send_message(self, sender_id: str, recipient_id: int, content: Any):
        msg = Message(sender_id, recipient_id, content)
        self.parties[recipient_id].inbox.put(msg)

    def broad_cast(self, content: Any, sender_id: str = "Manager"):
        for bank_id in self.parties:
            self.send_message(sender_id, bank_id, content)


@regi_algo_party('FedGD')
class FedGD_party(Party):

    def __init__(self, args, bank_id, data, manager=None, scalar_encoders = None):
        super().__init__(args, bank_id, data, manager, scalar_encoders)
        print('FedGD_party was selected')
    
    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)
    
    @staticmethod
    def return_class(args, bank_id, data, manager = None, scalar_encoders = None):
        return FedGD_party(args, bank_id, data, manager, scalar_encoders)


@regi_algo_manager('FedGD')
class FedGD_manager(Manager):

    def __init__(self, args):
        super().__init__(args)
        print('FedGD_manager was selected')

    def get_adjacency_matrix(self):
        return 0

    @staticmethod
    def return_class(args):
        return FedGD_manager(args)
    


# Setup for manager and parties ------------------------------------------------------

# Get the manager
manager = Manager.get_algo_class(fl_args)

# Get the parties, and their data
for bank in relevant_banks:
    
    bank_indices = get_indices_bdt(raw_data, bank=bank)
    train_data, vali_data, test_data = get_booster_data(data_args, raw_data[data_args.data_type], bank_indices)
    tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
    Party.get_algo_class(fl_args, bank, tmp_data, manager, scalar_encoders)
    #manager.parties[bank].start()


parties = manager.parties
manager.global_model = fl_utils.LogReg(0,1)

tmp_train, tmp_vali = parties[0].feature_engineering(parties[0].data['train_data'], parties[0].data['vali_data'])
num_start_params = tmp_train['x'].shape[1]

import numpy as np
init_parameters = np.random.normal(size = num_start_params)
manager.parties_weights = {}

# --------------------------------------------


for bank_id, party in manager.parties.items():
        party.model.current_w = init_parameters
        manager.parties_weights[bank_id] = party.model.current_w




for i in range(0, 10):

    for bank_id, party in manager.parties.items():
        party.model.update_weights()
        manager.parties_weights[bank_id] = party.model.current_w

    manager.global_weights = sum(manager.parties_weights.values()) / len(manager.parties_weights)
    for bank_id, party in manager.parties.items():
        party.model.current_w = manager.global_weights

    










# --------------------------------------------

# Start the manager and parties
manager.start()
for bank in relevant_banks:
    manager.parties[bank].start()

manager.send_parameters(init_parameters)
manager.parties[0].model.current_w

parties[0].model.update_weights()

parties[0].model.new_w
parties[0].model.current_w

parties[0].model.log_regi()




manager.model.initial_parameters()
manager.model.current_parameters
manager.send_parameters(manager.model.current_parameters)

manager.parties[0].model.current_w

manager.parties[0].data['train_data']


# Send initial model etc.


# simple training

manager.parties[0].model

X = manager.parties[0].data['train_data']['x'].iloc[:, 0:11]
y = manager.parties[0].data['train_data']['y']

tmpmodel = LinearRegression()

tmpmodel.fit(X,y)

#tmpmodel.coef_


m = X.shape[0]
X_transposed = X.T
Q_value = 1/m * X_transposed @ X
q_value = -2/m * X_transposed @ y




# Tuning ------------------------------------------------------

manager.args.model















# Training ------------------------------------------------------


# Save model(s) ------------------------------------------------------


# Inference ------------------------------------------------------



# need feature engineering functions etc.

feature_engi_regular_data()








parties = manager.parties

parties[bank]
dir(parties[bank])

parties[bank].running = True
manager.parties[bank].running










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