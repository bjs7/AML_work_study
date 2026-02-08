"""Individual GNN Manager - trains each party independently with separate models."""

from .manager_mixin import GNNMixinManager_Fullinfo_Indi
from data.relevant_banks import get_relevant_banks
from inference import metrics
import inference as flin
import numpy as np
import logging
import utils
import copy

logger = logging.getLogger(__name__)


class IndividualGNNManager(GNNMixinManager_Fullinfo_Indi):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup fr_banks, tune them, then add sr_banks with best hyperparameters."""
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
        return self._gnn_tuning(party_laundering_values, bank_id = bank_id)

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_eval_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def _train_party(self, laundering_values, **kwargs):
        bank_id = kwargs.get('bank_id')
        return self.parties[bank_id].train(laundering_values)

    def _helper_party_test(self, party, laundering_values_test):
        mask = np.isin(laundering_values_test['indices'], party.get_test_indices())
        return laundering_values_test.iloc[mask,].reset_index(drop=True)

    def _train_helper(self, hyperparameters, laundering_values_vali, laundering_values_test):
        return self._train(hyperparameters, laundering_values_vali, laundering_values_test)

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        models_hyperparameters = {}
        party_predictions = {}
        party_individual_performans = {}

        logger.info("Training %d individual banks", len(self.parties))
        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
            logger.debug("Training bank %s (%d/%d)", bank_id, idx, len(self.parties))
            
            self.init_models(hyperparameters[bank_id]['hyperparameters'], bank_id)
            party_lv_vali = self._helper_party_tuning(party, laundering_values_vali)
            party_lv_test = self._helper_party_test(party, laundering_values_test)

            tmp_model = party.train(party_lv_vali, party_lv_test)
            models_hyperparameters[bank_id] = {'model': tmp_model['model'],
                                               'hyperparameters': hyperparameters[bank_id]['hyperparameters']}

            party_predictions[bank_id] = tmp_model['laundering_values']['pred_probabilities']
            party_individual_performans[bank_id] = {**tmp_model['metrics'], 'best_vali_f1': tmp_model['best_vali_f1']}

            if np.all(party_predictions[bank_id] == 0):
                logger.warning("Bank %s: All predictions are zero!", bank_id)
            elif np.isnan(party_predictions[bank_id]).any():
                logger.warning("Bank %s: Predictions contain NaN values", bank_id)

        logger.info("Loading best models and aggregating predictions")
        for bank_id, party in self.parties.items():
            party.model.gnn.load_state_dict(models_hyperparameters[bank_id]['model'])
            flin.update_laundering_values(party, laundering_values_test,
                                          pred_probabilities=party_predictions[bank_id], mode='test')

        collective_metrics = metrics(y_true=laundering_values_test['true_y'],
                                     y_pred_probabilities=laundering_values_test['avg_prob'],
                                     y_pred_binary=laundering_values_test['pred_label'])

        logger.info("Final metrics - F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                   collective_metrics['f1'], collective_metrics['precision'], collective_metrics['recall'],
                   collective_metrics['roc_auc'], collective_metrics['pr_auc'])

        # Warnings for unusual results
        if collective_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", collective_metrics['f1'])
        if collective_metrics['precision'] == 0 or collective_metrics['recall'] == 0:
            logger.warning("Zero precision or recall - Model may not be learning properly")

        return {
            'metrics': collective_metrics,
            'laundering_values': copy.deepcopy(laundering_values_test),
            'models': models_hyperparameters,
            'party_performance': party_individual_performans
        }

