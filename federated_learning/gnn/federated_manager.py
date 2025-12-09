"""Federated GNN Manager - coordinates training across all parties with weight aggregation."""

import copy
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from relbanks_saving_analysis.relevant_banks import get_relevant_banks
from .communication import GNNCommunicationMixin
from .manager_mixin import GNNMixinManager
from training.utils import ibm_gnn
import logging
from sklearn.metrics import f1_score


from models.gnn import add_arange_ids, batching_masker, get_loaders
import pandas as pd
import numpy as np

import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv, LayerNorm
import torch.nn.functional as F
import torch


# one could also have it such that manager holds all labels,
# individual calculates their own network, or fedavg or something like that,
# but rather then "splitting" or "combining" networks, banks just do caculations for themself
# though there could be some weight sharing or similar as there is in fedavg

logger = logging.getLogger(__name__)


class FLGNNManagerVertical(GNNCommunicationMixin, GNNMixinManager):
    """Vertical Federated Learning Manager."""

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):

        self.label_data = laundering_values

        fr_banks, sr_banks = get_relevant_banks(parsers)

        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]


        # Add and tune fr_banks
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        tuned_hp, _ = self.tuning(laundering_values)

        # Add sr_banks
        if self.args['data_parser'].train_for_final:
            sr_banks = []
        utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders)

        return tuned_hp
        
    def tuning(self, laundering_values):

        # --------------------------------

        if self.args['data_parser'].ibm_hp:
            return ibm_gnn, None

        self.set_mode('tuning')

        for bank_id, party in self.parties.items():
            party.prep_data()

        return self._gnn_tuning(laundering_values)
    

    def train(self):

        self.set_mode('training')
        results_by_seed = {}

        return 0
    
    

    def fl_training(self):

        
        # get intersects of banks

        for bank_id, party in self.parties.items():
            party.intersects = {}

        # might be possible to simple just use this 
        # np.where(np.isin(self.parties[0].indices['train_indices'], self.parties[2].indices['train_indices']))

        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):

            for i in list(self.parties)[idx:]:
                tmp_intersects = np.intersect1d(party.indices['train_indices'], self.parties[i].indices['train_indices'])
                party.intersects[i] = tmp_intersects
                self.parties[i].intersects[bank_id] = tmp_intersects


        party1 = self.parties[0].data['train_data']['df']
        party2 = self.parties[2].data['train_data']['df']

        #embeddings

        class Embeddings(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)



        self.parties[0].data['train_data']
        
        self.parties[0].indices['train_indices']
        
        self.parties[2].indices['train_indices']

        

        # can one extract the nodes from this somehow?
        self.parties[0].indices['train_indices'][0:10]
        self.parties[2].indices['train_indices'][0:10]

        np.where(13  in self.parties[0].indices['train_indices'])

        part1 = [i for i, value in enumerate(self.parties[0].indices['train_indices']) if value == 13]
        part2 = [i for i, value in enumerate(self.parties[2].indices['train_indices']) if value == 13]

        self.parties[0].data['train_data']['df'].edge_attr[part1,:]
        self.parties[2].data['train_data']['df'].edge_attr[part2,:]


        part1 = [i for i, value in enumerate(self.parties[0].indices['train_indices']) if value == 21]
        part2 = [i for i, value in enumerate(self.parties[2].indices['train_indices']) if value == 21]

        self.parties[0].data['train_data']['df'].edge_attr[part1,:]
        self.parties[2].data['train_data']['df'].edge_attr[part2,:]







# FedAvg
class FLGNNManager(GNNCommunicationMixin, GNNMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):

        fr_banks, sr_banks = get_relevant_banks(parsers)

        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]

        # Add and tune fr_banks
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        tuned_hp, _ = self.tuning(laundering_values)

        # Add sr_banks
        if self.args['data_parser'].train_for_final:
            sr_banks = []
        utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders)

        return tuned_hp

    def tuning(self, laundering_values):

        # --------------------------------

        if self.args['data_parser'].ibm_hp:
            return ibm_gnn, None

        self.set_mode('tuning')

        for bank_id, party in self.parties.items():
            party.prep_data()

        return self._gnn_tuning(laundering_values)

    def tuning_loop(self, hyperparameters_tuning, laundering_values):

        best_f1 = -1
        best_hyperparameters = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams)
            self.get_global_weights()
            self.send_global_weights_params()

            # if reg or graph epochs is used. Or also is for decision trees, yes?
            # just update in one, and then another for sending to manager?
            results = self.fl_training(laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])

        return best_hyperparameters, scores, best_f1
    
    def train(self, hyperparameters, laundering_values, seeds = 4):

        self.set_mode('training')
        results_by_seed = {}
        #best_f1 = -1; best_model = None

        logger.info("="*80)
        logger.info("Starting training with %d seeds for federated learning")
        logger.info("="*80)

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(hyperparameters, copy.deepcopy(laundering_values))

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
            seed_value,
            results_by_seed[seed_value]['metrics']['f1'],
            results_by_seed[seed_value]['metrics']['roc_auc'],
            results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)
        
        return results_by_seed
    
    def _train(self, hyperparameters, laundering_values):

        self.init_models(hyperparameters)
        
        self.get_global_weights()
        self.send_global_weights_params()

        return self.fl_training(laundering_values)
    
    def fl_training(self, laundering_values):

        best_weights = None
        best_metrics = None
        best_f1 = -1

        epochs = 20 if self.args['data_parser'].testing else configs.epochs

        for epoch in range(epochs):

            for bank_id, party in self.parties.items():
                # little unsure if I should update parameters like this, or if 
                # it should be kept inside the party
                party.update_local_weights()
                party.send_local_weights(self)

            self.update_global_weights()
            self.send_global_weights()

            # I need to reset laundering_values or something every time new parameters are tested
            # inference / status
            #if (epoch+1) % 20 == 0:

            # reset preditcions
            for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
                laundering_values[col] = 0
            
            for bank_id, party in self.parties.items():
                flin.update_laundering_values(party, laundering_values)

            f1_eval = f1_score(laundering_values['true_y'], laundering_values['pred_label'])

            logger.info("Epoch %d/%d - F1: %.4f", epoch + 1, epochs, f1_eval)

            if f1_eval > best_f1:
                best_laundering_values = copy.deepcopy(laundering_values)
                best_weights = copy.deepcopy(self.global_weights)
                best_f1 = f1_eval

        best_metrics = metrics(y_true = best_laundering_values['true_y'], 
                                    y_pred_probabilities = best_laundering_values['avg_prob'], 
                                    y_pred_binary = best_laundering_values['pred_label'])
        
        if best_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", best_metrics['f1'])
        if (best_metrics['precision'] == 0 or best_metrics['recall'] == 0):
            logger.warning("Zero precision or recall - Model may not be learning properly")
        

        return {'weights': best_weights, 'metrics': best_metrics, 'laundering_values': best_laundering_values}

