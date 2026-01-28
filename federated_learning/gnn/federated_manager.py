"""Federated GNN Manager - coordinates training across all parties with weight aggregation."""

import copy
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from data.relevant_banks import get_relevant_banks
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
        # might need to create a "global" of these, like where one take the average of the averages
        cols = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        self.data['means_tr'] = self.data['train_data'][cols].apply(np.mean)
        self.data['std_tr'] = self.data['train_data'][cols].apply(np.std)
        self.data['means_eval'] = self.data['eval_data'][cols].apply(np.mean)
        self.data['std_eval'] = self.data['eval_data'][cols].apply(np.std)


    def get_relevant_banks_vert(self):

        # Get banks for first round (train) and second round (eval only)
        # not sure if this part here is actually "necessary"
        # the part about only looping over banks in the first round, but in the second
        # round other banks can be considered to. I think it is needed and that I need to
        # remember to account for this
        fr_banks = set(self.data['train_data'][['From Bank', 'To Bank']].stack())
        sr_banks = set(self.data['eval_data'][['From Bank', 'To Bank']].stack())
        sr_banks = list(sr_banks - fr_banks)
        fr_banks = list(fr_banks)

        return fr_banks, sr_banks

    def set_manager_data(self, reg_df, mode):

        train = reg_df['train_data']['x']
        vali = reg_df['vali_data']['x']
        test = reg_df['test_data']['x']
        train_vali = pd.concat([train, vali])

        if mode == 'tuning':
            self.data = {'train_data': train, 'eval_data': train_vali}
        #elif mode == 'training':
        else:
            self.data = {'train_data': train_vali, 'eval_data': pd.concat([train_vali, test])}

        self.indices = {'train': self.data['train_data'].index, 'eval': self.data['eval_data'].index[self.data['train_data'].shape[0]:]}

    def add_parties_prep_data(self, mode, df, scaler_encoders):

        self.set_manager_data(df['regular_data'], mode)
        self.cal_global_sats()

        fr_banks, sr_banks = self.get_relevant_banks_vert()
        utils.add_banks_to_manager(self.args, fr_banks, self, df, scaler_encoders)
        utils.add_banks_to_manager(self.args, sr_banks, self, df, scaler_encoders, is_sr=True)

        # might still be able to optimize on the prep of banks with limited data, like share
        # the mean/std of values etc.
        self.set_mode(mode)
        for bank_id, party in self.sr_parties.items():
            party.prep_data()

        self.setup_vertical(batching=self.args['data_parser'].batching)


    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup parties for vertical FL.

        Args:
            df: DataFrame with graph and regular data
            parsers: Parser dict with data_parser, gnn_parser, fl_parser
            scaler_encoders: Scaler/encoder objects for data preprocessing
            laundering_values: Laundering values DataFrame for evaluation
        """
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
        self.args['fl_parser'].model = 'GINe_vert'
        #if hyperparameters is None:
        #    hyperparameters = self.tuning(laundering_values)[0]

        # Initialize model on first party and share with all
        self.init_models(hyperparams=hyperparameters, bank_id=0)
        for bank_id, party in self.sr_parties.items():
            if bank_id == 0:
                continue
            party.model = self.parties[0].model
        self.model = self.parties[0].model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(
            self.model.gnn.parameters(),
            lr=hyperparameters.get('learning_rate')
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([
                hyperparameters.get('w_ce1'),
                hyperparameters.get('w_ce2')
            ]).to(device)
        )

        return hyperparameters

    def get_batch_data(self, mode, batch_num=None, batch_banks=None):
        """Get batch data for a given mode and batch."""
        return forward.get_batch_data(self, mode, batch_num, batch_banks)

    def forward_pass(self, mode, batch_num, batch_banks, batch_data):
        """Execute forward pass with embedding exchange."""
        return forward.forward_pass(self, mode, batch_num, batch_banks, batch_data)
    
    def train(self, hyperparameters, laundering_values, seeds = 4):

        self.set_mode('training')
        results_by_seed = {}
        #best_f1 = -1; best_model = None

        logger.info("="*80)
        logger.info("Starting training with %d seeds for federated learning")
        logger.info("="*80)

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
        self.setup_model(hyperparameters, laundering_values)
        return self.train_vertical(laundering_values, batching = self.args['data_parser'].batching)

    def train_vertical(self, laundering_values, epochs=None, batching=True):
        """Train the vertical FL model.

        Args:
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
            batch_data_eval = self.get_batch_data('eval')

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

            # Evaluation phase
            self.model.gnn.eval()
            all_preds_eval, all_labels_eval = [], []
            batch_iter = range(self.ctx['eval']['num_batches']) if batching else [None]

            with torch.no_grad():
                for batch_num in batch_iter:
                    if batching:
                        batch_banks = self.ctx['eval'][batch_num]['batch_parties']
                        batch_data = self.get_batch_data('eval', batch_num, batch_banks)
                    else:
                        batch_banks = self.ctx['eval'][None]['batch_parties']
                        batch_data = batch_data_eval

                    preds, labels = self.forward_pass('eval', batch_num, batch_banks, batch_data)
                    all_preds_eval.append(preds)
                    all_labels_eval.append(labels)

            labels_np, preds_probs, preds_binary = training_utils.prep_eval_preds_labels(
                all_labels_eval, all_preds_eval
            )
            f1_eval = f1_score(labels_np, preds_binary)
            logger.info(f'Test F1: {f1_eval}')

            best_f1, best_model_state = training_utils.update_best_model(
                f1_eval, best_f1, best_model_state,
                labels_np, preds_probs, self.model, epoch
            )

        final_metrics = training_utils.compute_final_metrics(best_model_state)

        laundering_values['true_y'] = best_model_state['best_ground_truths']
        laundering_values['pred_probabilities'] = best_model_state['best_pred_probabilities']
        laundering_values['pred_label'] = best_model_state['best_pred_binary']
        best_model = best_model_state['best_model']

        return {
            'weights': best_model,
            'metrics': final_metrics,
            'laundering_values': laundering_values
        }



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

