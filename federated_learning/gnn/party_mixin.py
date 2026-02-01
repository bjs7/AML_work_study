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
        self._batching = self.manager.args['data_parser'].batching
        self._use_global_stats = self.manager.args['data_parser'].use_global_stats

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


    def feature_engineering(self, *data_configs):
        """Process multiple data splits with feature engineering.

        Args:
            *data_configs: Tuples of (data_dict, means_key, std_key) for each split.
        Returns:
            List of processed data dicts, one per input config.
        """
        if self._ibm_fe:

            results = []
            for data_dict, means_key, std_key in data_configs:
                if data_dict['df'] is None:
                    results.append(data_dict)
                    continue

                data_dict = copy.deepcopy(data_dict)
                df = data_dict['df']
                if df.x.shape[0] > 1:
                    z_norm(df.x)
                else:
                    df.x = torch.tensor([[0.]])

                # Edge features: use global or local statistics for standardization
                start = self.edge_feat_start
                if self._use_global_stats or df.edge_attr.shape[0] <= 1:
                    # Use global statistics from manager
                    df_means = self.manager.data[means_key]
                    df_std = self.manager.data[std_key]
                    for idx, col in enumerate(['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format'], start):
                        df.edge_attr[:,idx] = (df.edge_attr[:,idx] - df_means[col]) / df_std[col]
                else:
                    # Use local statistics (z_norm)
                    df.edge_attr[:,start:] = z_norm(df.edge_attr[:,start:])
                results.append(data_dict)

            return results

        # Non-ibm_fe path
        train_data = data_configs[0][0]
        processed_train = feature_engi_graph_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        results = [processed_train]
        for data_dict, _, _ in data_configs[1:]:
            results.append(feature_engi_graph_data(data_dict, self.args['gnn_parser'],
                                                   scaler_encoders=processed_train.get('scaler_encoders')))

        return results

    def prep_data(self):

        if self.mode == 'tuning':
            train_proc, vali_proc = self.feature_engineering(
                (self.data['train_data'], 'means_tr', 'std_tr'),
                (self.data['vali_data'], 'means_vali', 'std_vali'))
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc}
        elif self.mode == 'training':
            train_proc, vali_proc, test_proc = self.feature_engineering(
                (self.data['train_data'], 'means_tr', 'std_tr'),
                (self.data['vali_data'], 'means_vali', 'std_vali'),
                (self.data['test_data'], 'means_test', 'std_test'))
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc, 'test_data': test_proc}
    

    def update_local_weights(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']['df']
        loss = None
        for epoch in range(num_local_epochs):
            loss = self.model.update_weights_no_batching(tr_data)
        
        return loss

    def send_local_weights(self, manager):
        manager.parties_weights[self.bank_id] = {param: value.data.clone() for param, value in self.model.gnn.named_parameters()}
    
    def get_eval_data(self):
        return {'df': self.procs_data['vali_data']['df'],
                'pred_indices': self.procs_data['vali_data']['pred_indices']}

    def get_test_data(self):
        return {'df': self.procs_data['test_data']['df'],
                'pred_indices': self.procs_data['test_data']['pred_indices']}
    
    def _get_loaders(self):

        train_data = copy.deepcopy(self.procs_data['train_data']['df'])
        train_indices = self.data['train_data']['pred_indices']

        vali_data = copy.deepcopy(self.procs_data['vali_data']['df'])
        vali_indices = self.procs_data['vali_data']['pred_indices']

        has_test = 'test_data' in self.procs_data
        test_data = copy.deepcopy(self.procs_data['test_data']['df']) if has_test else None
        test_indices = self.procs_data['test_data']['pred_indices'] if has_test else None

        if self._batching:
            datasets = [train_data, vali_data] + ([test_data] if has_test else [])
            add_arange_ids(datasets)
            num_neighbors = [100]*self.model.gnn.num_gnn_layers
            train_loader, vali_loader = get_loaders(train_data, vali_data, vali_indices, num_neighbors)
            test_loader = get_loaders(train_data, test_data, test_indices, num_neighbors)[1] if has_test else None
        else:
            train_loader, vali_loader = [train_data], [vali_data]
            test_loader = [test_data] if has_test else None

        return (train_loader, vali_loader, test_loader,
                train_data, vali_data, test_data,
                train_indices, vali_indices, test_indices)
    

    def _eval_on_loader(self, loader, data, indices, bank_str):
        """Run evaluation on a given loader, return predictions and ground truths."""
        preds_list, gt_list, id_list = [], [], []
        for batch in loader:
            mask, pred_ids = batching_masker(batch, data, loader, indices) if self._batching else (torch.zeros(data.y.shape[0], dtype=torch.bool).index_fill_(0, indices, True), indices)
            pred = self.model.predict(batch, mask)
            preds_list.append(pred)
            gt_list.append(batch.y[mask])
            id_list.append(pred_ids)

        preds = torch.cat(preds_list, dim=0).detach().cpu().numpy()
        ground_truths = torch.cat(gt_list, dim=0).detach().cpu().numpy()
        pred_ids = torch.cat(id_list).detach().cpu().numpy()
        return preds, ground_truths, pred_ids

    def train(self, laundering_values, laundering_values_test=None):

        best_model, best_f1 = None, -1
        laundering_values['pred_probabilities'], laundering_values['pred_label'] = 0, 0
        bank_str = f" {self.bank_id}" if self.bank_id is not None else " full_info"

        (train_loader, vali_loader, test_loader,
         train_data, vali_data, test_data,
         train_indices, vali_indices, test_indices) = self._get_loaders()
        laundering_values = pd.concat([pd.DataFrame(data = {'party_indices': vali_indices}), laundering_values], axis=1)
        epochs = 10 if self.args['data_parser'].testing else configs.epochs

        # --- Epoch loop: train on train, select best model via vali ---
        for epoch in range(epochs):

            preds, ground_truths, total_loss = [], [], 0

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
                logger.info("Epoch %d/%d - Loss: %.4f, Train F1: %.4f", epoch + 1, epochs, total_loss, f1)
                if len(ground_truths) != len(train_indices):
                    logger.warning("Difference in the size of ground_truths and train_data indices, %d and %d",
                                    len(ground_truths), len(train_indices))

            # Validation evaluation for model selection
            preds_vali, ground_truths_vali, pred_ids_vali = self._eval_on_loader(
                vali_loader, vali_data, vali_indices, bank_str)

            if len(ground_truths_vali) != len(laundering_values['true_y']):
                logger.warning("Difference in the size of ground_truths and laundering_values['true_y'], %d and %d in bank %s",
                                len(ground_truths_vali), len(laundering_values['true_y']), bank_str)

            f1_vali = f1_score(ground_truths_vali, probs_to_binary(preds_vali))

            if self.args['fl_parser'].fl_algo == 'full_info' or self.args['data_parser'].testing:
                logger.info(f'Vali F1: {f1_vali}')

            if torch.isnan(torch.tensor(preds_vali)).any() and self.mode == 'training':
                logger.warning("Model predictions contain NaN values! In bank %s", bank_str)

            if f1_vali > best_f1:
                best_model = copy.deepcopy(self.model.gnn.state_dict())
                best_f1 = f1_vali
                logger.debug("New best Vali F1: %.4f at epoch %d", best_f1, epoch + 1)

        if self.mode == 'tuning':
            return {'f1': best_f1}

        if best_model is None:
            logger.error("No evaluation occurred during training (epochs=%d). Check evaluation frequency in bank %s", epochs, bank_str)
            raise ValueError(f"No evaluation occurred during training (epochs={epochs}). Check evaluation frequency.")

        # --- Final test evaluation with best model ---
        self.model.gnn.load_state_dict(best_model)
        preds_test, ground_truths_test, pred_ids_test = self._eval_on_loader(
            test_loader, test_data, test_indices, bank_str)
        pred_binary_test = probs_to_binary(preds_test)

        perform_metrics = metrics(y_true=ground_truths_test, y_pred_probabilities=preds_test)

        df_launderings = pd.DataFrame(data={'party_indices': pred_ids_test, 'true_y': ground_truths_test,
                                            'pred_probabilities': preds_test,
                                            'pred_label': pred_binary_test})
        df_launderings = df_launderings.sort_values(by=['party_indices'], ignore_index=True)

        # Update laundering_values_test with test predictions
        if laundering_values_test is not None:
            laundering_values_test['pred_probabilities'], laundering_values_test['pred_label'] = 0, 0
            lv_test = pd.concat([pd.DataFrame(data={'party_indices': test_indices}), laundering_values_test], axis=1)

            if not np.all(lv_test['true_y'].values == df_launderings['true_y'].values):
                logger.warning("Difference in true_y from laundering_values_test and df_launderings in bank %s", bank_str)

            df_launderings = df_launderings.set_index('party_indices')
            lv_test = lv_test.set_index('party_indices')
            lv_test.update(df_launderings[['pred_probabilities', 'pred_label']])
            laundering_values_test = lv_test.reset_index()

        if self.args['fl_parser'].fl_algo == 'full_info' or self.args['data_parser'].testing:
            logger.info("Training complete - Best Vali F1: %.4f | Test F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                    best_f1, perform_metrics['f1'], perform_metrics['precision'], perform_metrics['recall'],
                    perform_metrics['roc_auc'], perform_metrics['pr_auc'])

        if perform_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration in bank %s", perform_metrics['f1'], bank_str)
        if (perform_metrics['precision'] == 0 or perform_metrics['recall'] == 0):
            logger.warning("Zero precision or recall - Model may not be learning properly in bank %s", bank_str)

        return {'metrics': perform_metrics,
                'laundering_values': laundering_values_test if laundering_values_test is not None else laundering_values,
                'model': best_model}


class GNNMixinPartyVert(GNNMixinParty):

    def set_up_parties(self):
        pass

