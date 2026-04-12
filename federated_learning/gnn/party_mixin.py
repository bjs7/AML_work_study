"""GNN-specific Party mixin providing data preparation and weight updates."""

from models.gnn import add_arange_ids, batching_masker, get_loaders
from torch_geometric.loader import LinkNeighborLoader
from data.feature_engineering import feature_engi_graph_data
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
    """Base GNN party mixin — shared by all algorithms (FL and individual)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ibm_fe = self.manager.args['data_parser'].ibm_fe
        self._batching = self.manager.args['data_parser'].batching
        self._use_global_stats = self.manager.args['data_parser'].use_global_stats
        self._add_missing_edges = not self.manager.args['data_parser'].replicate_ibm

    def _get_batch_configs(self):
        """Determine per-party batch configuration based on training data size.

        Respects self._batching as global override (if False, no batching).
        Otherwise tiered by edge count:
          >= 100K edges: batch_size=8192, num_neighbors=[100]*layers
          >= 8192 edges: batch_size=8192, num_neighbors=[50]*layers
          < 8192 edges:  no batching
        """
        if not self._batching:
            self.train_configs['use_batching'] = False
            self.train_configs['batch_size'] = 0
            self.train_configs['num_neighbors'] = None
            return

        num_edges = self.procs_data['train_data']['df'].num_edges
        num_gnn_layers = self.model.gnn.num_gnn_layers
        batch_size = self.manager.args['data_parser'].batch_size

        if num_edges >= 100_000:
            self.train_configs['use_batching'] = True
            self.train_configs['batch_size'] = batch_size
            self.train_configs['num_neighbors'] = [100] * num_gnn_layers
        elif num_edges >= batch_size:
            self.train_configs['use_batching'] = True
            self.train_configs['batch_size'] = batch_size
            self.train_configs['num_neighbors'] = [25] * num_gnn_layers
        else:
            self.train_configs['use_batching'] = False
            self.train_configs['batch_size'] = 0
            self.train_configs['num_neighbors'] = None

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
                    df.x = z_norm(df.x)
                else:
                    df.x = torch.tensor([[0.]])

                # Edge features: use global or local statistics for standardization
                start = self.edge_feat_start
                is_fedgraph = self.manager.args['fl_parser'].fl_algo == 'FedGraph'
                if self._use_global_stats or (is_fedgraph and df.edge_attr.shape[0] <= 1):
                    # Use global statistics from manager (FedGraph)
                    df_means = self.manager.data[means_key]
                    df_std = self.manager.data[std_key]
                    for idx, col in enumerate(['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format'], start):
                        df.edge_attr[:,idx] = (df.edge_attr[:,idx] - df_means[col]) / df_std[col]
                elif df.edge_attr.shape[0] > 1:
                    # Use local statistics (z_norm) — only if enough edges
                    df.edge_attr[:,start:] = z_norm(df.edge_attr[:,start:])
                # else: skip standardization — too few edges for meaningful local stats
                results.append(data_dict)

            return results

        # Non-ibm_fe path
        train_data = data_configs[0][0]

        # If train data is insufficient to fit scalers, use global encoders for OHE
        # and fit amount scaler on the first available eval split (eval-only parties)
        if train_data['df'] is None or train_data['df'].edge_attr.shape[0] < 2:
            fitted_encoders = self.scaler_encoders
            results = []
            for data_dict, _, _ in data_configs:
                if data_dict['df'] is None:
                    results.append(data_dict)
                    continue
                processed = feature_engi_graph_data(data_dict, self.args, fitted_encoders, edge_feat_start=self.edge_feat_start)
                fitted_encoders = processed.get('scaler_encoders', fitted_encoders)
                results.append(processed)
            return results

        # Pre-fitted encoders (from extract_enc_cats) ensure categories use global encoders,
        # while amount scalers are fitted per-party on the train split
        processed_train = feature_engi_graph_data(train_data, self.args, self.scaler_encoders, edge_feat_start=self.edge_feat_start)
        results = [processed_train]
        for data_dict, _, _ in data_configs[1:]:
            if data_dict['df'] is None:
                results.append(data_dict)
                continue
            results.append(feature_engi_graph_data(data_dict, self.args,
                                                   scaler_encoders=processed_train.get('scaler_encoders'),
                                                   edge_feat_start=self.edge_feat_start))

        return results

    def prep_data(self):

        if self.mode == 'tuning':
            train_proc, vali_proc = self.feature_engineering(
                (self.data['train_data'], 'means_tr', 'std_tr'),
                (self.data['vali_data'], 'means_vali', 'std_vali'))
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc}
            assert 'test_data' not in self.procs_data, "Test data must not be processed during tuning!"
        elif self.mode == 'training':
            train_proc, vali_proc, test_proc = self.feature_engineering(
                (self.data['train_data'], 'means_tr', 'std_tr'),
                (self.data['vali_data'], 'means_vali', 'std_vali'),
                (self.data['test_data'], 'means_test', 'std_test'))
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc, 'test_data': test_proc}

    def get_vali_data(self):
        return {'df': self.procs_data['vali_data']['df'],
                'pred_indices': self.procs_data['vali_data']['pred_indices']}

    def _eval_on_loader(self, loader, data, indices, bank_str):
        """Run evaluation on a given loader, return predictions and ground truths."""
        preds_list, gt_list, id_list = [], [], []
        for batch in loader:
            mask, pred_ids = batching_masker(batch, data, loader, indices) if self.train_configs['use_batching'] else (torch.zeros(data.y.shape[0], dtype=torch.bool).index_fill_(0, indices, True), indices)
            pred = self.model.predict(batch, mask)
            preds_list.append(pred)
            gt_list.append(batch.y[mask])
            id_list.append(pred_ids)

        preds = torch.cat(preds_list, dim=0).detach().cpu().numpy()
        ground_truths = torch.cat(gt_list, dim=0).detach().cpu().numpy()
        pred_ids = torch.cat(id_list).detach().cpu().numpy()
        return preds, ground_truths, pred_ids


class GNNMixinPartyIndi(GNNMixinParty):
    """Individual/FullInfo party mixin — used by individual and full_info algorithms."""

    def _get_loaders(self):

        self._get_batch_configs()

        train_data = copy.deepcopy(self.procs_data['train_data']['df'])
        train_indices = self.data['train_data']['pred_indices']

        vali_data = copy.deepcopy(self.procs_data['vali_data']['df'])
        vali_indices = self.procs_data['vali_data']['pred_indices']

        has_test = 'test_data' in self.procs_data
        test_data = copy.deepcopy(self.procs_data['test_data']['df']) if has_test else None
        test_indices = self.procs_data['test_data']['pred_indices'] if has_test else None

        # Invariant: train/vali/test indices must never overlap
        tr_set = set(train_indices.tolist() if hasattr(train_indices, 'tolist') else train_indices)
        va_set = set(vali_indices.tolist() if hasattr(vali_indices, 'tolist') else vali_indices)
        assert tr_set.isdisjoint(va_set), "Train and vali indices overlap!"
        if test_indices is not None:
            te_set = set(test_indices.tolist() if hasattr(test_indices, 'tolist') else test_indices)
            assert tr_set.isdisjoint(te_set), "Train and test indices overlap!"
            assert va_set.isdisjoint(te_set), "Vali and test indices overlap!"

        if self.train_configs['use_batching']:
            datasets = [train_data, vali_data] + ([test_data] if has_test else [])
            add_arange_ids(datasets)
            num_neighbors = self.train_configs['num_neighbors']
            batch_size = self.train_configs['batch_size']
            train_loader, vali_loader = get_loaders(train_data, vali_data, vali_indices, num_neighbors, batch_size)
            test_loader = get_loaders(train_data, test_data, test_indices, num_neighbors, batch_size)[1] if has_test else None
        else:
            train_loader, vali_loader = [train_data], [vali_data]
            test_loader = [test_data] if has_test else None

        return (train_loader, vali_loader, test_loader,
                train_data, vali_data, test_data,
                train_indices, vali_indices, test_indices)

    def train(self, laundering_values, laundering_values_test=None):

        best_model, best_f1 = None, -1
        laundering_values['pred_probabilities'], laundering_values['pred_label'] = 0, 0
        if laundering_values_test is not None:
            laundering_values_test['pred_probabilities'], laundering_values_test['pred_label'] = 0, 0
        bank_str = f" {self.bank_id}" if self.bank_id is not None else " full_info"

        (train_loader, vali_loader, test_loader,
         train_data, vali_data, test_data,
         train_indices, vali_indices, test_indices) = self._get_loaders()
        laundering_values = pd.concat([pd.DataFrame(data = {'party_indices': vali_indices}), laundering_values], axis=1)
        epochs = 10 if self.args['data_parser'].testing else self.args['fl_parser'].num_rounds

        # --- Epoch loop: train on train, select best model via vali ---
        for epoch in range(epochs):

            preds, ground_truths, total_loss = [], [], 0

            for batch in train_loader:
                mask, _ = batching_masker(batch, train_data, train_loader, train_indices, add_missing_edges=self._add_missing_edges) if self.train_configs['use_batching'] else (torch.ones(train_data.y.shape[0], dtype=torch.bool), None)
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
                'model': best_model,
                'best_vali_f1': best_f1}


class GNNMixinPartyFL(GNNMixinParty):
    """FL-specific party mixin — used by FedAvg, FedProx, FedGraph."""

    def _setup_train_loader(self):
        """Set up batch configs and create train loader if batching is needed.

        Must be called after model init (needs num_gnn_layers).
        """
        self._get_batch_configs()

        if not self.train_configs['use_batching']:
            self._train_loader = None
            return

        tr_data = copy.deepcopy(self.procs_data['train_data']['df'])
        add_arange_ids([tr_data])

        self._train_loader_data = tr_data
        self._train_indices = self.procs_data['train_data']['pred_indices']
        self._train_loader = LinkNeighborLoader(
            tr_data,
            num_neighbors=self.train_configs['num_neighbors'],
            edge_label_index=tr_data.edge_index,
            edge_label=tr_data.y,
            batch_size=self.train_configs['batch_size'],
            shuffle=True
        )

    def _setup_eval_loader(self, mode='vali'):
        """Set up and cache a vali/test loader for batched prediction.

        Must be called after _setup_train_loader (needs train_configs).

        Args:
            mode: 'vali' for validation loader, 'test' for test loader.
        """
        data_key = 'test_data' if mode == 'test' else 'vali_data'
        loader_attr = f'_{mode}_loader'
        data_attr = f'_{mode}_loader_data'
        indices_attr = f'_{mode}_pred_indices'

        if not self.train_configs.get('use_batching'):
            setattr(self, loader_attr, None)
            setattr(self, data_attr, None)
            setattr(self, indices_attr, None)
            return

        data_copy = copy.deepcopy(self.procs_data[data_key]['df'])
        pred_indices = self.procs_data[data_key]['pred_indices']
        add_arange_ids([data_copy])

        setattr(self, data_attr, data_copy)
        setattr(self, indices_attr, pred_indices)
        setattr(self, loader_attr, LinkNeighborLoader(
            data_copy,
            num_neighbors=self.train_configs['num_neighbors'],
            edge_label_index=data_copy.edge_index[:, pred_indices],
            edge_label=data_copy.y[pred_indices],
            batch_size=self.train_configs['batch_size'],
            shuffle=False
        ))

    def _set_global_weight_reference(self):
        """Snapshot current (global) weights as reference for FedProx proximal term.

        Called by the manager before update_local_weights() each round.
        At this point, the party's model holds the freshly-distributed global weights.
        """
        self.model.set_global_weight_reference()

    def update_local_weights(self, num_local_epochs=1):

        tr_data = self.procs_data['train_data']['df']
        loss = None

        for epoch in range(num_local_epochs):
            if self.train_configs.get('use_batching'):
                for batch in self._train_loader:
                    mask, _ = batching_masker(batch, self._train_loader_data,
                                              self._train_loader, self._train_indices,
                                              add_missing_edges=self._add_missing_edges)
                    _, _, loss = self.model.update_weights(batch, mask)
            else:
                loss = self.model.update_weights_no_batching(tr_data)

        return loss

    def send_local_weights(self, manager):
        manager.parties_weights[self.bank_id] = {param: value.data.clone().cpu() for param, value in self.model.gnn.named_parameters()}

    def get_predictions(self, mode='vali'):
        """Get predictions for vali or test data, handling batching internally.

        Uses cached loaders from _setup_eval_loader() when batching is enabled.
        Falls back to predict_no_batching() otherwise.

        Args:
            mode: 'vali' for validation data, 'test' for test data.
        Returns:
            pred_probabilities as numpy array, aligned with get_vali_indices()/get_test_indices().
        """
        loader = getattr(self, f'_{mode}_loader', None)

        if loader is not None:
            data_copy = getattr(self, f'_{mode}_loader_data')
            pred_indices = getattr(self, f'_{mode}_pred_indices')

            preds, _, pred_ids = self._eval_on_loader(loader, data_copy, pred_indices, "")

            # Reorder predictions to match pred_indices order so they
            # align with get_vali_indices()/get_test_indices()
            pred_indices_np = pred_indices.numpy() if isinstance(pred_indices, torch.Tensor) else np.array(pred_indices)
            pred_map = dict(zip(pred_ids.astype(int).tolist(), preds.tolist()))
            ordered_preds = np.array([pred_map[int(pi)] for pi in pred_indices_np])

            return ordered_preds
        else:
            data_key = 'test_data' if mode == 'test' else 'vali_data'
            return self.model.predict_no_batching(self.procs_data[data_key])


class GNNMixinPartyVert(GNNMixinPartyFL):
    """Vertical FL party mixin for FedGraph."""

    def set_up_parties(self):
        pass
