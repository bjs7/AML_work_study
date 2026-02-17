"""Individual Booster Manager - trains each party independently with separate models."""

import copy
import numpy as np
import utils
from inference import metrics
import inference as flin
from .manager_mixin import BoosterMixinManager
from data.relevant_banks import get_relevant_banks
from training.parallel import parallel_party_execute
from federated_learning.gnn.manager_mixin import GNNMixinManager_Fullinfo_Indi
import logging

logger = logging.getLogger(__name__)


class IndividualBoosterManager(BoosterMixinManager, GNNMixinManager_Fullinfo_Indi):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup banks, tune them."""
        banks = get_relevant_banks(parsers)

        if parsers['data_parser'].testing:
            banks = banks[0:5]
            logger.info("Testing mode: Limited to %d banks", len(banks))
        else:
            logger.info("Production mode: Using %d banks", len(banks))

        logger.info("Adding %d banks to manager", len(banks))
        utils.add_banks_to_manager(parsers, banks, self, df, scaler_encoders)
        logger.info("Starting hyperparameter tuning for banks")
        tuned_hp = self.tuning(laundering_values)
        logger.info("Hyperparameter tuning completed")

        logger.info("Setup complete: Total %d banks", len(self.parties))
        return tuned_hp

    def _tuning_helper(self, laundering_values, party, bank_id):
        party_laundering_values = self._helper_party_tuning(party, laundering_values)
        return self._booster_tuning(party_laundering_values, bank_id=bank_id)

    def _booster_tuning(self, laundering_values, **kwargs):
        hyperparameters_tuning = self.init_hyperparams()
        _, scores, _ = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        # Get top 5 and refine
        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        hyperparameters_tuning = self.init_hyperparams(sample_space)
        tuned_hyperparameters, _, f1_score_for_hp = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        return tuned_hyperparameters, f1_score_for_hp

    def _get_search_space(self, hyperparameters_tuning):
        """Extract search intervals from top hyperparameters for refinement."""
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

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_vali_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def _helper_party_test(self, party, laundering_values_test):
        mask = np.isin(laundering_values_test['indices'], party.get_test_indices())
        return laundering_values_test.iloc[mask,].reset_index(drop=True)

    def _train_party(self, laundering_values, **kwargs):
        bank_id = kwargs.get('bank_id')
        return self.parties[bank_id].train(laundering_values)

    def _train_helper(self, hyperparameters, laundering_values_vali, laundering_values_test):
        return self._train(hyperparameters, laundering_values_vali, laundering_values_test)

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test, max_workers=None):

        models_hyperparameters = {}
        party_predictions = {}
        party_individual_performans = {}

        if max_workers is None:
            max_workers = getattr(self.args['fl_parser'], 'max_workers', 1)

        # Setup: init models and prepare per-party data (sequential)
        party_data = {}
        logger.info("Training %d individual banks", len(self.parties))
        for bank_id, party in self.parties.items():
            self.init_models(hyperparameters[bank_id]['hyperparameters'], bank_id)
            party_data[bank_id] = {
                'lv_vali': self._helper_party_tuning(party, laundering_values_vali),
                'lv_test': self._helper_party_test(party, laundering_values_test),
                'hyperparameters': hyperparameters[bank_id]['hyperparameters']
            }

        # Training: run party.train() in parallel
        def _train_party(bank_id, party):
            return party.train(party_data[bank_id]['lv_vali'], party_data[bank_id]['lv_test'])

        train_results = parallel_party_execute(self.parties, _train_party, max_workers=max_workers)

        # Collect results (sequential)
        for bank_id, tmp_model in train_results.items():
            models_hyperparameters[bank_id] = {'model': tmp_model['model'],
                                               'hyperparameters': party_data[bank_id]['hyperparameters']}

            party_predictions[bank_id] = tmp_model['laundering_values']['pred_probabilities']
            party_individual_performans[bank_id] = {**tmp_model['metrics'], 'best_vali_f1': tmp_model['best_vali_f1']}

            if np.all(party_predictions[bank_id] == 0):
                logger.warning("Bank %s: All predictions are zero!", bank_id)
            elif np.isnan(party_predictions[bank_id]).any():
                logger.warning("Bank %s: Predictions contain NaN values", bank_id)

        logger.info("Loading best models and aggregating predictions")
        for bank_id, party in self.parties.items():
            party.model.load_model_raw(models_hyperparameters[bank_id]['model'])
            flin.update_laundering_values(party, laundering_values_test,
                                          pred_probabilities=party_predictions[bank_id], mode='test')

        collective_metrics = metrics(y_true=laundering_values_test['true_y'],
                                     y_pred_probabilities=laundering_values_test['avg_prob'],
                                     y_pred_binary=laundering_values_test['pred_label'])

        logger.info("Final metrics - F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                   collective_metrics['f1'], collective_metrics['precision'], collective_metrics['recall'],
                   collective_metrics['roc_auc'], collective_metrics['pr_auc'])

        return {
            'metrics': collective_metrics,
            'laundering_values': copy.deepcopy(laundering_values_test),
            'models': models_hyperparameters,
            'party_performance': party_individual_performans
        }
