"""Federated Booster Manager - coordinates XGBoost training across parties with ensemble averaging."""

import copy
import random
import utils
from inference import metrics
import inference as flin
from .manager_mixin import BoosterMixinManager
from data.relevant_banks import load_relevant_banks, apply_bank_filter
from training.parallel import parallel_party_execute
from sklearn.metrics import f1_score
from inference import probs_to_binary
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FLBoosterManager(BoosterMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):

        if parsers['data_parser'].eval_mode == 'comparable':
            train_banks = load_relevant_banks(parsers['data_parser']).get('individual').get('banks')
            vali_banks, test_banks = [], []
        else:
            fedavg_banks = load_relevant_banks(parsers['data_parser']).get('FedAvg')
            train_banks = fedavg_banks['train_banks']
            vali_banks = fedavg_banks['vali_banks']
            test_banks = fedavg_banks['test_banks']

        if parsers['data_parser'].bank_filter:
            train_banks = apply_bank_filter(train_banks, df, parsers['data_parser'].bank_filter)

        if parsers['data_parser'].testing:
            train_banks = train_banks[0:5]
            vali_banks = vali_banks[0:5]
            test_banks = test_banks[0:5]

        for banks, bank_type in zip([train_banks, vali_banks, test_banks], ['train', 'vali', 'test']):
            if banks:
                utils.add_banks_to_manager(parsers, banks, self, df, scaler_encoders,
                                           bank_type=bank_type, superset_merge=False)

        tuned_hp = self.tuning(laundering_values)
        return tuned_hp

    def tuning(self, laundering_values):

        if self.args['data_parser'].ibm_hp:
            return self.init_hyperparams()[0], None

        self.set_mode('tuning')
        for bank_id, party in self.iter_parties(include_test=False):
            party.prep_data()

        return self._booster_tuning(laundering_values)

    def _booster_tuning(self, laundering_values):
        hyperparameters_tuning = self.init_hyperparams()
        _, scores, _ = self.tuning_loop(hyperparameters_tuning, laundering_values)

        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        hyperparameters_tuning = self.init_hyperparams(sample_space)
        tuned_hyperparameters, _, f1_score_for_hp = self.tuning_loop(hyperparameters_tuning, laundering_values)

        return tuned_hyperparameters, f1_score_for_hp

    def _get_search_space(self, hyperparameters_tuning):
        intervals = {}
        keys_to_track = ['num_rounds', 'max_depth', 'learning_rate', 'lambda', 'scale_pos_weight',
                         'colsample_bytree', 'subsample']
        for key in keys_to_track:
            vals = []
            for hp in hyperparameters_tuning:
                val = hp.get(key) or hp.get('params', {}).get(key)
                if val is not None:
                    vals.append(val)
            if vals:
                intervals[f'{key}_interval'] = [min(vals), max(vals)]
        return intervals

    def tuning_loop(self, hyperparameters_tuning, laundering_values):

        best_f1 = -1
        best_hyperparameters = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            self.init_models(hyperparams)
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

        logger.info("=" * 80)
        logger.info("Starting training with %d seeds for federated boosting", seeds)
        logger.info("=" * 80)

        for bank_id, party in self.iter_parties(include_test=True):
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("-" * 80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-" * 80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(
                hyperparameters, copy.deepcopy(laundering_values_vali), copy.deepcopy(laundering_values_test))

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                        seed_value,
                        results_by_seed[seed_value]['metrics']['f1'],
                        results_by_seed[seed_value]['metrics']['roc_auc'],
                        results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("=" * 80)
        logger.info("All seeds completed")
        logger.info("=" * 80)

        return results_by_seed

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        self.init_models(hyperparameters)
        return self.fl_training(laundering_values_vali, laundering_values_test)

    def fl_training(self, laundering_values_vali, laundering_values_test=None, max_workers=None):
        """Federated booster training with ensemble averaging.

        Each party trains its own XGBoost model. At inference time,
        predictions from all parties are averaged (ensemble).
        """
        best_models = None
        best_f1 = -1

        if max_workers is None:
            max_workers = getattr(self.args['fl_parser'], 'max_workers', 1)

        epochs = 20 if self.args['data_parser'].testing else self.args['fl_parser'].num_rounds

        fl_parser = self.args['fl_parser']
        client_fraction = getattr(fl_parser, 'client_fraction', 1.0)
        num_local_epochs = getattr(fl_parser, 'num_local_epochs', 1)

        all_bank_ids = list(self.parties.keys())
        num_total = len(all_bank_ids)
        num_sampled = max(1, int(client_fraction * num_total))

        logger.info("FL booster training: %d total parties, sampling %d per round, %d local epochs",
                     num_total, num_sampled, num_local_epochs)

        for epoch in range(epochs):

            # Party sampling
            if num_sampled < num_total:
                sampled_bank_ids = random.sample(all_bank_ids, num_sampled)
            else:
                sampled_bank_ids = all_bank_ids

            selected_parties = {bid: self.parties[bid] for bid in sampled_bank_ids}

            # Parallel local training
            def _train_party(bank_id, party):
                party.update_local_weights(num_local_epochs=num_local_epochs)

            parallel_party_execute(selected_parties, _train_party, max_workers=max_workers)

            # Ensemble evaluation on vali
            if laundering_values_vali is not None:
                for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
                    laundering_values_vali[col] = 0

                for bank_id, party in self.iter_parties(include_test=False):
                    flin.update_laundering_values(party, laundering_values_vali, mode='vali')

                f1_vali = f1_score(laundering_values_vali['true_y'], laundering_values_vali['pred_label'])
                logger.info("Epoch %d/%d - Vali F1: %.4f", epoch + 1, epochs, f1_vali)

                if f1_vali > best_f1:
                    best_models = {bid: party.model.get_model_raw()
                                   for bid, party in self.parties.items()}
                    best_f1 = f1_vali

        # If tuning (no test data), return early
        if laundering_values_test is None:
            return {'metrics': {'f1': best_f1}}

        # Restore best models
        assert best_models is not None, "No best models found"
        for bank_id, party in self.parties.items():
            if bank_id in best_models:
                party.model.load_model_raw(best_models[bank_id])

        # Final test evaluation
        for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
            laundering_values_test[col] = 0

        for bank_id, party in self.iter_parties(include_test=True):
            flin.update_laundering_values(party, laundering_values_test, mode='test')

        best_metrics = metrics(y_true=laundering_values_test['true_y'],
                               y_pred_probabilities=laundering_values_test['avg_prob'],
                               y_pred_binary=laundering_values_test['pred_label'])

        logger.info("Final metrics - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                     best_metrics['f1'], best_metrics['roc_auc'], best_metrics['pr_auc'])

        return {
            'metrics': best_metrics,
            'laundering_values': copy.deepcopy(laundering_values_test),
            'models': best_models
        }
