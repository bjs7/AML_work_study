"""Federated GNN Manager - coordinates training across all parties with weight aggregation."""

import copy
import random
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from data.relevant_banks import load_relevant_banks, apply_bank_filter
from .communication import GNNCommunicationMixin
from .manager_mixin import GNNMixinManager
from training.utils import ibm_gnn
import logging
from sklearn.metrics import f1_score
from training.parallel import parallel_party_execute

from models.gnn import add_arange_ids, batching_masker, get_loaders
from data.get_indices_type_data import get_indices_bdt
import pandas as pd
import numpy as np

import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv, LayerNorm
import torch.nn.functional as F
import torch

from .vertical import setup, forward, training_utils


logger = logging.getLogger(__name__)


class FLGNNManagerVertical(GNNCommunicationMixin, GNNMixinManager):
    """Vertical Federated Learning Manager.

    Coordinates training across parties using embedding exchange rather than
    weight aggregation. All parties share a single model and exchange
    embeddings for cross-bank transactions.
    """

    def cal_global_sats(self):

        # Calculate statistics for normalization
        cols = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        self.data['means_tr'] = self.data['train_data'][cols].apply(np.mean)
        self.data['std_tr'] = self.data['train_data'][cols].apply(np.std)
        self.data['means_vali'] = self.data['vali_data'][cols].apply(np.mean)
        self.data['std_vali'] = self.data['vali_data'][cols].apply(np.std)
        if 'test_data' in self.data:
            self.data['means_test'] = self.data['test_data'][cols].apply(np.mean)
            self.data['std_test'] = self.data['test_data'][cols].apply(np.std)


    def get_relevant_banks_vert(self):

        # Get banks for first round (train) and second round (vali/test only)
        train_banks = set(self.data['train_data'][['From Bank', 'To Bank']].stack())
        vali_banks = set(self.data['vali_data'][['From Bank', 'To Bank']].stack())
        test_banks = None
        if 'test_data' in self.data:
            test_banks = set(self.data['test_data'][['From Bank', 'To Bank']].stack())
            test_banks = list(test_banks - vali_banks - train_banks)
        vali_banks = list(vali_banks - train_banks)
        train_banks = list(train_banks)

        #full_key = 'test_data' if 'test_data' in self.data else 'vali_data'
        #sr_banks = set(self.data[full_key][['From Bank', 'To Bank']].stack())
        #sr_banks = list(sr_banks - fr_banks)
        #fr_banks = list(fr_banks)

        return train_banks, vali_banks, test_banks

    def set_manager_data(self, reg_df, mode):

        train = reg_df['train_data']['x']
        vali = reg_df['vali_data']['x']
        test = reg_df['test_data']['x']
        train_vali = pd.concat([train, vali])

        if mode == 'tuning':
            self.data = {'train_data': train, 'vali_data': train_vali}
            self.indices = {
                'train': train.index,
                'vali': train_vali.index[train.shape[0]:]
            }
        else:
            self.data = {'train_data': train, 'vali_data': train_vali, 'test_data': pd.concat([train_vali, test])}
            self.indices = {
                'train': train.index,
                'vali': train_vali.index[train.shape[0]:],
                'test': pd.concat([train_vali, test]).index[train_vali.shape[0]:]
            }

    def add_parties_prep_data(self, mode, df, scaler_encoders):

        self.set_manager_data(df['regular_data'], mode)
        self.cal_global_sats()

        #fr_banks, sr_banks = self.get_relevant_banks_vert()
        train_banks, vali_banks, test_banks = self.get_relevant_banks_vert()
        for banks, bank_type in zip([train_banks, vali_banks, test_banks], ['train', 'vali', 'test']):
            if banks:
                utils.add_banks_to_manager(self.args, banks, self, df, scaler_encoders, bank_type=bank_type)

        #utils.add_banks_to_manager(self.args, fr_banks, self, df, scaler_encoders)
        #utils.add_banks_to_manager(self.args, sr_banks, self, df, scaler_encoders, is_sr=True)
        #utils.add_banks_to_manager(self.args, sr_banks, self, df, scaler_encoders, is_sr=True)

        self.set_mode(mode)
        parties = self.test_parties if mode == 'training' else self.vali_parties
        for bank_id, party in parties.items(): #TODO Needs to only be self.parties when tuning?
            party.prep_data()

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup parties for vertical FL.

        Args:
            df: DataFrame with graph and regular data
            parsers: Parser dict with data_parser, gnn_parser, fl_parser
            scaler_encoders: Scaler/encoder objects for data preprocessing
            laundering_values: Laundering values DataFrame for evaluation
        """

        add_arange_ids([df['graph_data']['train_data']['df'], 
                        df['graph_data']['vali_data']['df'],
                        df['graph_data']['test_data']['df']])

        if not self.args['data_parser'].ibm_hp:
            self.add_parties_prep_data('tuning', df, scaler_encoders)
            #self.label_data = laundering_values
            
        tuned_hp, _ = self.tuning(laundering_values)

        self.add_parties_prep_data('training', df, scaler_encoders)

        return tuned_hp
    
    def tuning(self, laundering_values):

        # --------------------------------

        return ibm_gnn, None

        

        #if self.args['data_parser'].ibm_hp:
        #    return ibm_gnn, None
        #self.set_mode('tuning')
        #for bank_id, party in self.parties.items():
        #    party.prep_data()
        #return 0 #self._gnn_tuning(laundering_values)

    def setup_vertical(self, batching=True):
        """Set up vertical FL structures (intersects, mappings, batches).

        Args:
            batching: Whether to use batching (True) or process all data at once (False)
        """
        setup.setup_vertical(self, batching=batching)

    def setup_model(self, hyperparameters, laundering_values):
        """Initialize the shared model and optimizer.

        Args:
            hyperparameters: Model hyperparameters dict
            laundering_values: Laundering values for evaluation
        """
        #self.args['fl_parser'].model = 'GINe_vert'
        #if hyperparameters is None:
        #    hyperparameters = self.tuning(laundering_values)[0]

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model on first party and share with all
        self.init_models(hyperparams=hyperparameters, bank_id=0)
        all_parties = self.test_parties if self.mode == 'training' else self.vali_parties
        for bank_id, party in all_parties.items():
            if bank_id == 0:
                continue
            party.model = self.parties[0].model
        self.model = self.parties[0].model

        # Move model to device (all parties share this same model reference)
        self.model.gnn.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.gnn.parameters(),
            lr=hyperparameters.get('learning_rate')
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([
                hyperparameters.get('w_ce1'),
                hyperparameters.get('w_ce2')
            ]).to(self.device)
        )

        return hyperparameters

    def get_batch_data(self, mode, batch_num=None, batch_banks=None):
        """Get batch data for a given mode and batch."""
        return forward.get_batch_data(self, mode, batch_num, batch_banks)

    def forward_pass(self, mode, batch_num, batch_banks, batch_data):
        """Execute forward pass with embedding exchange."""
        return forward.forward_pass(self, mode, batch_num, batch_banks, batch_data)
    
    def train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds
        results_by_seed = {}

        logger.info("="*80)
        logger.info("Starting training with %d seeds for federated learning", seeds)
        logger.info("="*80)

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(hyperparameters, copy.deepcopy(laundering_values_test))

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
        self.setup_vertical(batching=self.args['data_parser'].batching)
        self.setup_model(hyperparameters, laundering_values)
        return self.train_vertical(laundering_values, batching=self.args['data_parser'].batching)

    def _forward_eval(self, mode, batching, precomputed_batch_data=None):
        """Run a forward pass on vali or test mode and return preds/labels."""
        all_preds, all_labels = [], []
        batch_iter = range(self.ctx[mode]['num_batches']) if batching else [None]

        with torch.no_grad():
            for batch_num in batch_iter:
                if batching:
                    batch_banks = self.ctx[mode][batch_num]['batch_parties']
                    batch_data = self.get_batch_data(mode, batch_num, batch_banks)
                else:
                    batch_banks = self.ctx[mode][None]['batch_parties']
                    batch_data = precomputed_batch_data

                preds, labels = self.forward_pass(mode, batch_num, batch_banks, batch_data)
                all_preds.append(preds)
                all_labels.append(labels)

        return training_utils.prep_eval_preds_labels(all_labels, all_preds)

    def train_vertical(self, laundering_values, epochs=None, batching=True):
        """Train the vertical FL model.

        Args:
            laundering_values: Test laundering values for final evaluation
            epochs: Number of training epochs (uses config default if None)
            batching: Whether to use batching

        Returns:
            Dict with best model state and metrics
        """
        if epochs is None:
            epochs = 2 if self.args['data_parser'].testing else configs.epochs

        best_f1 = -1
        best_model_state = None

        # Pre-generate batch data for non-batching mode
        if not batching:
            batch_data_train = self.get_batch_data('train')
            batch_data_vali = self.get_batch_data('vali')

        for epoch in range(epochs):

            # Training phase
            self.model.gnn.train()
            train_loss = 0
            batch_iter = range(self.ctx['train']['num_batches']) if batching else [None]
            all_preds, all_labels = [], []

            for batch_num in batch_iter:
                if batching:
                    batch_banks = self.ctx['train'][batch_num]['batch_parties']
                    batch_data = self.get_batch_data('train', batch_num, batch_banks)
                else:
                    batch_banks = self.ctx['train'][None]['batch_parties']
                    batch_data = batch_data_train

                self.optimizer.zero_grad()
                preds, labels = self.forward_pass('train', batch_num, batch_banks, batch_data)
                all_preds.append(preds)
                all_labels.append(labels)

                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            training_utils.log_train_performance(all_labels, all_preds, train_loss, epoch, epochs)

            # Validation phase (model selection)
            self.model.gnn.eval()
            labels_np, preds_probs, preds_binary = self._forward_eval(
                'vali', batching, batch_data_vali if not batching else None)

            f1_vali = f1_score(labels_np, preds_binary)
            logger.info(f'Vali F1: {f1_vali}')

            best_f1, best_model_state = training_utils.update_best_model(
                f1_vali, best_f1, best_model_state,
                labels_np, preds_probs, self.model, epoch
            )

        # --- Final test evaluation with best model ---
        assert best_model_state is not None, "No best model found — model selection on vali never succeeded!"
        self.model.gnn.load_state_dict(best_model_state['best_model'])
        self.model.gnn.eval()

        if not batching:
            batch_data_test = self.get_batch_data('test')

        labels_np, preds_probs, preds_binary = self._forward_eval(
            'test', batching, batch_data_test if not batching else None)

        final_metrics = training_utils.compute_final_metrics_from_preds(labels_np, preds_probs)
        logger.info(f'Test F1: {final_metrics["f1"]:.4f}')

        laundering_values['true_y'] = labels_np
        laundering_values['pred_probabilities'] = preds_probs
        laundering_values['pred_label'] = preds_binary
        best_model = best_model_state['best_model']

        return {
            'weights': best_model,
            'metrics': final_metrics,
            'laundering_values': laundering_values
        }



# FedAvg
class FLGNNManager(GNNCommunicationMixin, GNNMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values, analysis = False):

        fedavg_banks = load_relevant_banks(parsers).get('FedAvg')
        train_banks = fedavg_banks['train_banks']
        vali_banks = fedavg_banks['vali_banks']
        test_banks = fedavg_banks['test_banks']

        if parsers['data_parser'].bank_filter:
            train_banks = apply_bank_filter(
                train_banks, df, parsers['data_parser'].bank_filter)

        if parsers['data_parser'].testing:
            train_banks = train_banks[0:5]
            vali_banks = vali_banks[0:5]
            test_banks = test_banks[0:5]

        if analysis:
            for banks, bank_type in zip([train_banks], ['train']):
                if banks:
                    utils.add_banks_to_manager(parsers, banks, self, df, scaler_encoders, bank_type=bank_type, superset_merge=False)

            tuned_hp, _ = self.tuning(laundering_values)

            return tuned_hp          

        for banks, bank_type in zip([train_banks, vali_banks, test_banks], ['train', 'vali', 'test']):
            if banks:
                utils.add_banks_to_manager(parsers, banks, self, df, scaler_encoders, bank_type=bank_type, superset_merge=False)

        tuned_hp, _ = self.tuning(laundering_values)

        return tuned_hp

    def tuning(self, laundering_values):

        # --------------------------------

        if self.args['data_parser'].ibm_hp:
            return ibm_gnn, None

        self.set_mode('tuning')

        for bank_id, party in self.iter_parties(include_test=False):
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
    
    def train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds
        results_by_seed = {}

        logger.info("="*80)
        logger.info("Starting training with %d seeds for federated learning", seeds)
        logger.info("="*80)

        for bank_id, party in self.iter_parties(include_test=True):
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(
                hyperparameters, copy.deepcopy(laundering_values_vali), copy.deepcopy(laundering_values_test))

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
            seed_value,
            results_by_seed[seed_value]['metrics']['f1'],
            results_by_seed[seed_value]['metrics']['roc_auc'],
            results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)

        return results_by_seed

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        self.init_models(hyperparameters)

        self.get_global_weights()
        self.send_global_weights_params()

        return self.fl_training(laundering_values_vali, laundering_values_test)

    def _compute_party_weights(self, selected_parties, weighting):
        """Compute aggregation weights for selected parties.

        Args:
            selected_parties: Dict of {bank_id: party} for this round's participants.
            weighting: 'proportional' or 'uniform'.

        Returns:
            Dict mapping bank_id -> float weight (sums to 1.0).
        """
        if weighting == 'uniform':
            n = len(selected_parties)
            return {bank_id: 1.0 / n for bank_id in selected_parties}

        # Proportional: w_k = n_k / sum(n_j for j in selected)
        dataset_sizes = {}
        for bank_id, party in selected_parties.items():
            dataset_sizes[bank_id] = party.procs_data['train_data']['df'].num_edges

        total = sum(dataset_sizes.values())
        return {bank_id: n_k / total for bank_id, n_k in dataset_sizes.items()}

    def fl_training(self, laundering_values_vali, laundering_values_test, max_workers=None):

        best_weights = None
        best_f1 = -1

        epochs = 20 if self.args['data_parser'].testing else self.args['fl_parser'].num_rounds

        # Read FL config
        fl_parser = self.args['fl_parser']
        if max_workers is None:
            max_workers = getattr(fl_parser, 'max_workers', 1)
        client_fraction = getattr(fl_parser, 'client_fraction', 1.0)
        num_local_epochs = getattr(fl_parser, 'num_local_epochs', 1)
        weighting = getattr(fl_parser, 'weighting', 'proportional')
        mu = getattr(fl_parser, 'mu', 0.0)

        # Party sampling setup
        all_bank_ids = list(self.parties.keys())
        num_total = len(all_bank_ids)
        num_sampled = max(1, int(client_fraction * num_total))

        # Setup loaders for train and vali parties (after model init)
        for bank_id, party in self.parties.items():
            party._setup_train_loader()
        for bank_id, party in self.iter_parties(include_test=False):
            party._setup_eval_loader(mode='vali')

        logger.info("FL training: %d total parties, sampling %d per round, "
                    "%d local epochs, weighting=%s, mu=%.4f",
                    num_total, num_sampled, num_local_epochs, weighting, mu)

        # --- Epoch loop: train, validate on vali for model selection ---
        for epoch in range(epochs):

            # Party sampling
            if num_sampled < num_total:
                sampled_bank_ids = random.sample(all_bank_ids, num_sampled)
            else:
                sampled_bank_ids = all_bank_ids

            selected_parties = {bid: self.parties[bid] for bid in sampled_bank_ids}

            # Clear parties_weights — only sampled parties contribute this round
            self.parties_weights = {}

            def _train_party(bank_id, party):
                if mu > 0:
                    party._set_global_weight_reference()
                party.update_local_weights(num_local_epochs=num_local_epochs)
                party.send_local_weights(self)

            parallel_party_execute(selected_parties, _train_party, max_workers=max_workers)

            # Weighted aggregation and broadcast
            party_weights_map = self._compute_party_weights(selected_parties, weighting)
            self.update_global_weights(party_weights_map)
            self.send_global_weights()

            # Evaluate on vali for model selection
            for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
                laundering_values_vali[col] = 0

            # Parallel predictions, sequential DataFrame updates
            vali_parties = dict(self.iter_parties(include_test=False))
            vali_preds = parallel_party_execute(
                vali_parties, lambda bid, p: p.get_predictions(mode='vali'), max_workers=max_workers)
            for bank_id, party in vali_parties.items():
                flin.update_laundering_values(party, laundering_values_vali,
                                              pred_probabilities=vali_preds[bank_id], mode='vali')

            f1_vali = f1_score(laundering_values_vali['true_y'], laundering_values_vali['pred_label'])

            logger.info("Epoch %d/%d - Vali F1: %.4f", epoch + 1, epochs, f1_vali)

            if f1_vali > best_f1:
                best_weights = copy.deepcopy(self.global_weights)
                best_f1 = f1_vali

        # --- Final test evaluation with best weights ---
        assert best_weights is not None, "No best weights found — model selection on vali never succeeded!"
        self.global_weights = best_weights
        self.send_global_weights()

        for bank_id, party in self.iter_parties(include_test=True):
            party._setup_eval_loader(mode='test')

        for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
            laundering_values_test[col] = 0

        for bank_id, party in self.iter_parties(include_test=True):
            flin.update_laundering_values(party, laundering_values_test, mode='test')

        best_metrics = metrics(y_true=laundering_values_test['true_y'],
                               y_pred_probabilities=laundering_values_test['avg_prob'],
                               y_pred_binary=laundering_values_test['pred_label'])

        # Per-party test metrics (global model evaluated on each bank's local test data)
        party_performance = {}
        for bank_id, party in self.iter_parties(include_test=True):
            pred_probs = party.get_predictions(mode='test')
            party_indices = party.get_test_indices()
            mask = laundering_values_test['indices'].isin(party_indices)
            true_y = laundering_values_test.loc[mask, 'true_y']
            if true_y.sum() > 0:
                party_performance[bank_id] = metrics(
                    y_true=true_y, y_pred_probabilities=pred_probs)

        logger.info("Test F1: %.4f", best_metrics['f1'])

        if best_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", best_metrics['f1'])
        if (best_metrics['precision'] == 0 or best_metrics['recall'] == 0):
            logger.warning("Zero precision or recall - Model may not be learning properly")

        return {'weights': best_weights, 'metrics': best_metrics, 'laundering_values': laundering_values_test,
                'best_vali_f1': best_f1, 'party_performance': party_performance}

