"""GNN-specific Party mixin providing data preparation and weight updates."""

from models.gnn import add_arange_ids, batching_masker, get_loaders
from data.feature_engi import feature_engi_graph_data
from data.data_utils import z_norm
from inference import metrics, probs_to_binary
from sklearn.metrics import f1_score
import configs.configs as configs
import pandas as pd
import numpy as np
import logging
import torch
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)


class GNNMixinParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ibm_fe = self.manager.args['data_parser'].ibm_fe
        self._train_for_final = self.manager.args['data_parser'].train_for_final
        self._batching = self.manager.args['data_parser'].batching

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

        if self._ibm_fe:

            if train_data['df'] is not None:

                train_data = copy.deepcopy(train_data)
                if train_data['df'].x.shape[0] > 1:
                    z_norm(train_data['df'].x)
                else:
                    train_data['df'].x = torch.tensor([[0.]])

                if train_data['df'].edge_attr.shape[0] > 1:
                    train_data['df'].edge_attr[:,1:] = z_norm(train_data['df'].edge_attr[:,1:]) #TODO: Adjust to given scenario
                else:
                    df_means_tr = self.manager.data['means_tr']
                    df_std_tr = self.manager.data['std_tr']
                    for idx, col in enumerate(['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format'], 1):
                        train_data['df'].edge_attr[:,idx] = (train_data['df'].edge_attr[:,idx] - df_means_tr[col]) / df_std_tr[col]

            if eval_data['df'] is not None:
                eval_data = copy.deepcopy(eval_data)
                if eval_data['df'].x.shape[0] > 1:
                    z_norm(eval_data['df'].x)
                else:
                    eval_data['df'].x = torch.tensor([[0.]])

                if eval_data['df'].edge_attr.shape[0] > 1:
                    eval_data['df'].edge_attr[:,1:] = z_norm(eval_data['df'].edge_attr[:,1:])
                else:
                    df_means_eval = self.manager.data['means_eval']
                    df_std_eval = self.manager.data['std_eval']
                    for idx, col in enumerate(['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format'], 1):
                        eval_data['df'].edge_attr[:,idx] = (eval_data['df'].edge_attr[:,idx] - df_means_eval[col]) / df_std_eval[col]

            return train_data, eval_data

        train_data = feature_engi_graph_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        eval_data = feature_engi_graph_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, eval_data

    def prep_data(self):

        if self.mode == 'tuning':
            train_proc, eval_proc = self.feature_engineering(self.data['train_data'], 
                                                             self.data['vali_data'])
        elif self.mode == 'training':
            tr_data = self.data['vali_data'] if not self._train_for_final else self.data['train_data']
            train_proc, eval_proc = self.feature_engineering(tr_data, 
                                                             self.data['test_data'])
            
        self.procs_data = {'train_data': train_proc, 'eval_data': eval_proc}
    

    def update_local_weights(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']['df']
        loss = None
        for epoch in range(num_local_epochs):
            loss = self.model.update_weights_no_batching(tr_data)
        
        return loss

    def send_local_weights(self, manager):
        manager.parties_weights[self.bank_id] = {param: value.data.clone() for param, value in self.model.gnn.named_parameters()}
    
    def get_eval_data(self):
        return {'df': self.procs_data['eval_data']['df'], 
                'pred_indices': self.procs_data['eval_data']['pred_indices']}
    
    def _get_loaders(self):

        train_data = copy.deepcopy(self.procs_data['train_data']['df'])

        if (self.mode == 'tuning') or self.args['data_parser'].train_for_final:
            train_indices = self.data['train_data']['pred_indices']
        else:
            train_indices = torch.cat([self.data['train_data']['pred_indices'], 
                                       self.data['vali_data']['pred_indices']])

        eval_data = copy.deepcopy(self.procs_data['eval_data']['df'])
        eval_indices = self.procs_data['eval_data']['pred_indices']

        if self._batching:
            add_arange_ids([train_data, eval_data])
            num_neighbors = [100]*self.model.gnn.num_gnn_layers
            train_loader, eval_loader = get_loaders(train_data, eval_data, eval_indices, num_neighbors)
        else:
             train_loader, eval_loader = [train_data], [eval_data]

        return train_loader, eval_loader, train_data, eval_data, train_indices, eval_indices
    

    def train(self, laundering_values):
 
        best_pred_probabilities, best_model, best_f1 = None, None, -1
        laundering_values['pred_probabilities'], laundering_values['pred_label'] = 0, 0
        bank_str = f" {self.bank_id}" if self.bank_id is not None else " full_info"
        
        train_loader, eval_loader, train_data, eval_data, train_indices, eval_indices = self._get_loaders()
        laundering_values = pd.concat([pd.DataFrame(data = {'party_indices': eval_indices}), laundering_values], axis=1)
        epochs = 20 if self.args['data_parser'].testing else configs.epochs
        
        for epoch in range(epochs):

            preds, ground_truths, preds_eval = [], [], []
            ground_truths_eval, eval_pred_ids, total_loss = [], [], 0

            for batch in train_loader:
                mask, _ = batching_masker(batch, train_data, train_loader, train_indices) if self._batching else (torch.ones(train_data.y.shape[0], dtype=torch.bool), None)
                pred, true_y, loss = self.model.update_weights(batch, mask)
                total_loss += loss
                preds.append(pred.argmax(dim=-1))
                ground_truths.append(true_y)


            if (self.args['fl_parser'].fl_algo == 'full_info' or self.args['data_parser'].testing) and self.mode == 'training':
                preds = torch.cat(preds, dim=0).detach().cpu().numpy()
                ground_truths = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
                f1 = f1_score(ground_truths, preds)
                logger.info("Epoch %d/%d - Loss: %.4f, F1: %.4f", epoch + 1, epochs, total_loss, f1)
                if len(ground_truths) != len(train_indices):
                    logger.warning("Difference in the size of ground_truths and train_data indices, %d and %d", 
                                    len(ground_truths), len(train_indices))


            for batch in eval_loader:
                mask, pred_ids = batching_masker(batch, eval_data, eval_loader, eval_indices) if self._batching else (torch.zeros(eval_data.y.shape[0], dtype=torch.bool).index_fill_(0, eval_indices, True), eval_indices) 
                pred = self.model.predict(batch, mask)
                preds_eval.append(pred)
                ground_truths_eval.append(batch.y[mask])
                eval_pred_ids.append(pred_ids)

            preds_eval = torch.cat(preds_eval, dim=0).detach().cpu().numpy()
            preb_binary_eval = probs_to_binary(preds_eval)
            ground_truths_eval = torch.cat(ground_truths_eval, dim=0).detach().cpu().numpy()
            pred_ids = torch.cat(eval_pred_ids).detach().cpu().numpy()


            if len(ground_truths_eval) != len(laundering_values['true_y']):
                    logger.warning("Difference in the size of ground_truths and laundering_values['true_y'], %d and %d in bank %s",
                                    len(ground_truths_eval), len(laundering_values['true_y']), bank_str)


            f1_eval = f1_score(ground_truths_eval, preb_binary_eval)

            if self.args['fl_parser'].fl_algo == 'full_info' or self.args['data_parser'].testing:
                logger.info(f'Test F1: {f1_eval}')

            if torch.isnan(torch.tensor(preds_eval)).any() and self.mode == 'training':
                logger.warning("Model predictions contain NaN values! In bank %s", bank_str)
            
            if f1_eval > best_f1:
                best_pred_probabilities = copy.deepcopy(preds_eval)
                best_pred_binary = probs_to_binary(preds_eval)
                best_ground_truths = copy.deepcopy(ground_truths_eval)
                best_pred_ids = copy.deepcopy(pred_ids)

                best_model = copy.deepcopy(self.model.gnn.state_dict())
                best_f1 = f1_eval
                logger.debug("New best F1: %.4f at epoch %d", best_f1, epoch + 1)


        if self.mode == 'tuning':
            return {'f1': best_f1}

        
        if best_model is None:
            logger.error("No evaluation occurred during training (epochs=%d). Check evaluation frequency in bank %s", epochs, bank_str)
            raise ValueError(f"No evaluation occurred during training (epochs={epochs}). Check evaluation frequency.")

        perform_metrics = metrics(y_true = best_ground_truths, y_pred_probabilities = best_pred_probabilities)

        df_launderings = pd.DataFrame(data = {'party_indices': best_pred_ids,'true_y': best_ground_truths, 
                                             'pred_probabilities': best_pred_probabilities, 
                                             'pred_label': best_pred_binary})  
        df_launderings = df_launderings.sort_values(by=['party_indices'], ignore_index=True)

        if not np.all(laundering_values['true_y'] == df_launderings['true_y']):
            logger.warning("Difference in the true y from laudering_values and true y from df_launderings in bank %s", bank_str)

        df_launderings = df_launderings.set_index('party_indices')
        laundering_values = laundering_values.set_index('party_indices')
        laundering_values.update(df_launderings[['pred_probabilities', 'pred_label']])
        laundering_values = laundering_values.reset_index()

        if self.args['fl_parser'].fl_algo == 'full_info' or self.args['data_parser'].testing:
            logger.info("Training complete - Best F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                    perform_metrics['f1'], perform_metrics['precision'], perform_metrics['recall'],
                    perform_metrics['roc_auc'], perform_metrics['pr_auc'])

        if perform_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration in bank %s", perform_metrics['f1'], bank_str)
        if (perform_metrics['precision'] == 0 or perform_metrics['recall'] == 0):
            logger.warning("Zero precision or recall - Model may not be learning properly in bank %s", bank_str)
        
        return {'metrics': perform_metrics, 
                'laundering_values': laundering_values, 
                'model': best_model}

