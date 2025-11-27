
# packages
import pandas as pd
from data.feature_engi import feature_engi_regular_data, feature_engi_graph_data
# packages for FL
from typing import Dict, List, Optional, Any
import utils
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
from data.get_indices_type_data import get_indices_bdt
from data.get_indices_type_data import get_booster_data
import torch
import numpy as np
from federated_learning.fl_base import Manager, Party
import training.utils as tr_utils
from federated_learning.registry import GNN_REGISTRY
import configs.configs as configs
from inference import metrics
import copy
import inference as flin
from models.gnn import GNN


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
            party.prep_data()
        
        results, _ = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)

        return results    
