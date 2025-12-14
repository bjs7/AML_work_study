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
from data.get_indices_type_data import get_indices_bdt
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


# in the case of this, one still need banks, that doesn't have any laundring values, like regardless of whether
# they are in fr or sr subset, they are still needed, because in case they hold
# info or work as a link between other banks, they are needed
# so potentially every single bank is need, and needs to send data, else something might get lost
# because if a bank is missing, then other banks wont have the ability to send and receive data from it

logger = logging.getLogger(__name__)

# approach sharing embeddings between banks

class FLGNNManagerVertical(GNNCommunicationMixin, GNNMixinManager):
    """Vertical Federated Learning Manager."""

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):

        self.label_data = laundering_values
        
        fr_banks = set(pd.concat([
            df['regular_data']['train_data']['x'][['From Bank', 'To Bank']].stack(),
            df['regular_data']['vali_data']['x'][['From Bank', 'To Bank']].stack(),
            df['regular_data']['test_data']['x'][['From Bank', 'To Bank']].stack()]))
        
        fr_banks = list(fr_banks)
        sr_banks = []

        self.fl_fr_banks = []
        self.fl_sr_banks = []

        for bank in fr_banks:
            bank_indices = get_indices_bdt(df, bank=bank)
            if len(bank_indices['train_indices']) > 1:
                self.fl_fr_banks.append(bank)
            else:
                self.fl_sr_banks.append(bank)

        fr_banks = self.fl_fr_banks
        #len(fr_banks)
        #len(self.fl_fr_banks)
        #5705

        #self.full_fl_fr_sr_banks = self.fl_fr_banks + self.fl_sr_banks

        #banks_to_remove = []
        #for bank in fr_banks:
            #if len(np.where(df['regular_data']['train_data']['x'][['From Bank', 'To Bank']].stack() == bank)[0]) < 2:
                #banks_to_remove.append(bank)

        #fr_banks, sr_banks = get_relevant_banks(parsers)
        #if parsers['data_parser'].testing:
            #fr_banks = fr_banks[0:5]
            #sr_banks = sr_banks[0:2]

        #len(np.where(df['regular_data']['train_data']['x'][['From Bank', 'To Bank']].stack() == 1168)[0])

        #np.where(df['regular_data']['train_data']['x']['From Bank'] == 1168)
        #np.where(df['regular_data']['train_data']['x']['To Bank'] == 1168)
        #fr_banks = [71]
        #[idx for idx, value in enumerate(fr_banks) if value == 1168]
        #fr_banks[825:835]
        #bank = bank_id = 1170

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
    
    def train(self, hyperparameters, laundering_values, df, seeds = 1):

        logger.info("Staring training on FLvert")

        self.data = df

        # ------------------
        self.set_mode('training')
        results_by_seed = {}
        bank_str = f'{len(self.parties)} banks' if self.args['fl_parser'].fl_algo != 'full_info' else 'full info'

        logger.info("="*80)
        logger.info("Starting training with %d seeds for %s", seeds, bank_str)
        logger.info("="*80)

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(hyperparameters)

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                       seed_value,
                       results_by_seed[seed_value]['metrics']['f1'],
                       results_by_seed[seed_value]['metrics']['roc_auc'],
                       results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)
        
        return results_by_seed
    
    
    def _train(self, hyperparameters):

        best_f1 = -1
    
        self.init_models(hyperparams=hyperparameters, bank_id=0)
        for bank_id, party in self.parties.items():
            if bank_id == 0:
                continue
            party.model = self.parties[0].model

        self.model = self.parties[0].model

        # training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.model.gnn.parameters(), lr=hyperparameters.get('learning_rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparameters.get('w_ce1'), hyperparameters.get('w_ce2')]).to(device)) 
        epochs = 5 if self.args['data_parser'].testing else configs.epochs

        for epoch in range(epochs):

            self.model.gnn.train()
            self.optimizer.zero_grad()
            self.embeddings_indices = {}

            for bank_id, party in self.parties.items():
                
                party.current_embeddings = party.model.gnn.emed_features(party.procs_data['train_data']['df'].x,
                                                                    party.procs_data['train_data']['df'].edge_attr)
                
                party.current_embeddings['nodes'] = torch.where(torch.isnan(party.current_embeddings['nodes']), 0, party.current_embeddings['nodes'])
                party.current_embeddings['edges'] = torch.where(torch.isnan(party.current_embeddings['edges']), 0, party.current_embeddings['edges'])

                for layer_idx in range(party.model.gnn.num_gnn_layers):
                    party.current_embeddings = party.model.gnn.apply_gnn_layer(party.current_embeddings['nodes'],
                                                                    party.current_embeddings['edges'],
                                                                    party.procs_data['train_data']['df'].edge_index,
                                                                    layer_idx)
                
                
                party.current_embeddings = party.model.gnn.prep_nodes_edges(party.current_embeddings['nodes'],
                                                                party.current_embeddings['edges'],
                                                                party.data['train_data']['df'].edge_index)
                
                            
                self.embeddings_indices[bank_id] = dict(zip(party.indices['train_indices'], party.current_embeddings))
                
            
            #df_preds = pd.DataFrame(data = {'indices': [], 'true_y': [], 'preds': []})
            preds, true_y = [], []
            for index, row in self.data['regular_data']['train_data']['x'].iterrows():
                if index == 100000:
                    break
                #if np.isin([0,2,4], [int(row['From Bank'])]).any() and np.isin([0,2,4], [int(row['To Bank'])]).any():
                if int(row['From Bank']) == int(row['To Bank']):
                    embed = torch.concat([self.embeddings_indices[int(row['From Bank'])][index], 
                                    torch.zeros(self.embeddings_indices[int(row['From Bank'])][index].shape)])
                else:
                    embed = torch.concat(
                        [self.embeddings_indices[int(row['From Bank'])][index],
                        self.embeddings_indices[int(row['To Bank'])][index]]
                    )
                logits = self.model.gnn.mlp_vert(embed)
                preds.append(logits)
                true_y.append(int(row['Is Laundering']))

            preds_tensor = torch.stack(preds)
            true_y_tensor = torch.tensor(true_y, dtype=torch.long)
            loss = self.loss_fn(preds_tensor, true_y_tensor)
            loss.backward()
            self.optimizer.step()


            # eval
            self.embeddings_indices = {}
            for bank_id, party in self.parties.items():

                party.current_embeddings = party.model.gnn.emed_features(party.procs_data['eval_data']['df'].x,
                                                                    party.procs_data['eval_data']['df'].edge_attr)
                
                party.current_embeddings['nodes'] = torch.where(torch.isnan(party.current_embeddings['nodes']), 0, party.current_embeddings['nodes'])
                party.current_embeddings['edges'] = torch.where(torch.isnan(party.current_embeddings['edges']), 0, party.current_embeddings['edges'])
                
                for layer_idx in range(party.model.gnn.num_gnn_layers):
                    party.current_embeddings = party.model.gnn.apply_gnn_layer(party.current_embeddings['nodes'],
                                                                    party.current_embeddings['edges'],
                                                                    party.procs_data['eval_data']['df'].edge_index,
                                                                    layer_idx)
                
                party.current_embeddings = party.model.gnn.prep_nodes_edges(party.current_embeddings['nodes'],
                                                                party.current_embeddings['edges'],
                                                                party.procs_data['eval_data']['df'].edge_index)
                
                        
                self.embeddings_indices[bank_id] = dict(zip(party.indices['test_indices'], party.current_embeddings[party.procs_data['eval_data']['pred_indices']]))

            df_preds = []
            for index, row in self.data['regular_data']['test_data']['x'].iterrows():
                if index == 100000:
                    break
                #if np.isin([0,2,4], [int(row['From Bank'])]).any() and np.isin([0,2,4], [int(row['To Bank'])]).any():
                if int(row['From Bank']) == int(row['To Bank']):
                    embed = torch.concat([self.embeddings_indices[int(row['From Bank'])][index], 
                                    torch.zeros(self.embeddings_indices[int(row['From Bank'])][index].shape)])
                else:
                    embed = torch.concat(
                        [self.embeddings_indices[int(row['From Bank'])][index],
                        self.embeddings_indices[int(row['To Bank'])][index]]
                    )

                    logits = self.model.gnn.mlp_vert(embed)

                    df_preds.append(
                                    {'indices': index,
                                    'true_y': int(row['Is Laundering']),
                                    'pred_probabilities': logits.softmax(dim=0)[1].item(),
                                    'pred_label': logits.argmax(dim=-1).item()}
                                    )
                    
            df_preds = pd.DataFrame(df_preds)
            eval_f1 = f1_score(df_preds['true_y'], df_preds['pred_label'])

            if eval_f1 > best_f1:
                best_df_preds = copy.deepcopy(df_preds)
                best_model = copy.deepcopy(self.model.gnn.state_dict())
                best_f1 = eval_f1
            
        perform_metrics = metrics(y_true = best_df_preds['true_y'], y_pred_probabilities = best_df_preds['pred_probabilities'])

        return {'metrics': perform_metrics, 
                'laundering_values': best_df_preds, 
                'weights': best_model}


        




        
            



            
            
            

        








    
    
    def fl_training(self):


        
        












        
        # get intersects of banks

        self = manager

        for bank_id, party in self.parties.items():
            party.intersects = {}

        # might be possible to simple just use this 
        # np.where(np.isin(self.parties[0].indices['train_indices'], self.parties[2].indices['train_indices']))

        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
            for i in list(self.parties)[idx:]:
                tmp_intersects = np.intersect1d(party.indices['train_indices'], self.parties[i].indices['train_indices'])
                party.intersects[i] = tmp_intersects
                self.parties[i].intersects[bank_id] = tmp_intersects

        #test1 = np.where(np.isin(self.parties[0].indices['train_indices'], self.parties[0].intersects[2]))[0]
        #test2 = np.isin(self.parties[0].indices['train_indices'], self.parties[0].intersects[2])















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

