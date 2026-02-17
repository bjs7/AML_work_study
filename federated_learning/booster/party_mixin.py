"""Booster-specific Party mixin providing data preparation and weight updates."""

from data.feature_engineering import feature_engi_regular_data
from inference import metrics, probs_to_binary
from sklearn.metrics import f1_score
import configs.configs as configs
import pandas as pd
import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)


class BoosterMixinParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_feature_engineering = self.manager.args['data_parser'].ibm_fe
        self._train_for_final = self.manager.args['data_parser'].train_for_final

    def feature_engineering(self, train_data, eval_data):

        if self._skip_feature_engineering:
            return train_data, eval_data

        train_data = feature_engi_regular_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        eval_data = feature_engi_regular_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

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


    def update_local_weights(self, num_local_epochs=1):
        tr_data = self.procs_data['train_data']
        self.model.update_weights(tr_data['x'], tr_data['y'], num_rounds=num_local_epochs)

    def send_local_weights(self, manager):
        manager.parties_weights[self.bank_id] = self.model.get_model_raw()

    def get_eval_data(self):
        return self.procs_data['eval_data']

    def get_predictions(self, mode='vali'):
        """Get predictions for eval data."""
        data_key = 'eval_data' if mode in ('vali', 'test') else 'train_data'
        data = self.procs_data[data_key]
        return self.model.predict(data['x'])

    def train(self, laundering_values, laundering_values_test=None):

        best_model_raw = None
        best_f1 = -1
        bank_str = f" {self.bank_id}" if self.bank_id is not None else " full_info"

        tr_data = self.procs_data['train_data']
        eval_data = self.procs_data['eval_data']
        epochs = 10 if self.args['data_parser'].testing else configs.epochs

        laundering_values['pred_probabilities'], laundering_values['pred_label'] = 0, 0
        if laundering_values_test is not None:
            laundering_values_test['pred_probabilities'], laundering_values_test['pred_label'] = 0, 0

        # Train with early stopping on eval data
        self.model.fit(tr_data['x'], tr_data['y'], eval_data['x'], eval_data['y'],
                       early_stopping_rounds=50)

        # Get predictions on eval (vali) data
        preds_vali = self.model.predict(eval_data['x'])
        pred_binary_vali = probs_to_binary(preds_vali)
        f1_vali = f1_score(eval_data['y'], pred_binary_vali)
        best_f1 = f1_vali
        best_model_raw = self.model.get_model_raw()

        if self.mode == 'tuning':
            return {'f1': best_f1}

        # Test evaluation
        preds_test = self.model.predict(eval_data['x'])
        pred_binary_test = probs_to_binary(preds_test)

        perform_metrics = metrics(y_true=eval_data['y'], y_pred_probabilities=preds_test)

        logger.info("Bank%s - F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                    bank_str, perform_metrics['f1'], perform_metrics['precision'], perform_metrics['recall'],
                    perform_metrics['roc_auc'], perform_metrics['pr_auc'])

        df_launderings = laundering_values_test if laundering_values_test is not None else laundering_values
        df_launderings['pred_probabilities'] = preds_test
        df_launderings['pred_label'] = pred_binary_test

        return {'metrics': perform_metrics,
                'laundering_values': df_launderings,
                'model': best_model_raw,
                'best_vali_f1': best_f1}
