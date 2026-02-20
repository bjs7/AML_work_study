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


import xgboost as xgb


class FedHolder:

    def update_local_weights(self, num_local_epochs=1):
        tr_data = self.procs_data['train_data']
        self.model.update_weights(tr_data['x'], tr_data['y'], num_rounds=num_local_epochs)

    def send_local_weights(self, manager):
        manager.parties_weights[self.bank_id] = self.model.get_model_raw()


class BoosterMixinParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ibm_fe = self.manager.args['data_parser'].ibm_fe

    def feature_engineering(self, *data_configs):

        train_data = data_configs[0]
        processed_train = feature_engi_regular_data(train_data, self.args['data_parser'], self.scaler_encoders)
        results = [processed_train]
        for data_dict in data_configs[1:]:
            results.append(feature_engi_regular_data(data_dict, self.args['data_parser'], scaler_encoders=processed_train.get('scaler_encoders')))
    
        #train_data = feature_engi_regular_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        #eval_data = feature_engi_regular_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

        return results #train_data, eval_data

    def prep_data(self):

        if self.mode == 'tuning':
            train_proc, vali_proc = self.feature_engineering(self.data['train_data'],
                                                             self.data['vali_data'])
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc}
            assert 'test_data' not in self.procs_data, "Test data must not be processed during tuning!"
        elif self.mode == 'training':
            train_proc, vali_proc, test_proc = self.feature_engineering(
                                                            self.data['train_data'],
                                                             self.data['vali_data'],
                                                             self.data['test_data'])
            self.procs_data = {'train_data': train_proc, 'vali_data': vali_proc, 'test_data': test_proc}

    def get_eval_data(self):
        return self.procs_data['eval_data']

    def get_predictions(self, mode='vali'):
        """Get predictions for eval data."""
        data_key = 'eval_data' if mode in ('vali', 'test') else 'train_data'
        data = self.procs_data[data_key]
        return self.model.predict(data['x'])

    def train(self, hp, laundering_values, laundering_values_test=None):

        # is the model is the right format?
        self.model.fit(X_train = self.procs_data['train_data']['x'],
                       y_train = self.procs_data['train_data']['y'],
                       X_eval = self.procs_data['vali_data']['x'],
                       y_eval = laundering_values['true_y'])
        

        # get best f1 on vali data
        preds_vali = self.model.predict(self.procs_data['vali_data']['x'])
        best_vali_f1 = f1_score(laundering_values['true_y'], probs_to_binary(preds_vali))
        best_model_raw = self.model.get_model_raw()

        # test evaluation
        preds_test = self.model.predict(self.procs_data['test_data']['x'])
        perform_metrics = metrics(y_true=laundering_values_test['true_y'], y_pred_probabilities=preds_test)

        if laundering_values_test is not None:
            laundering_values_test['pred_probabilities'], laundering_values_test['pred_label'] = 0, 0

        laundering_values_test['pred_probabilities'] = preds_test
        laundering_values_test['pred_labels'] = probs_to_binary(preds_test)

        return {'metrics': perform_metrics,
                'laundering_values': laundering_values_test,
                'model': best_model_raw,
                'best_vali_f1': best_vali_f1}








#self = self.parties[0]
#hp = hyperparameters[bank_id]['hyperparameters']
# laundering_values = party_data[bank_id]['lv_vali']


# run with early stopping here
#booster_data_train = xgb.DMatrix(self.procs_data['train_data']['x'], self.procs_data['train_data']['y'])
#booster_data_vali = xgb.DMatrix(self.procs_data['vali_data']['x'], laundering_values['true_y'])

#model = xgb.train(hp['params'], booster_data_train, hp['num_rounds'],
#            evals = [(booster_data_vali, 'eval')], early_stopping_rounds = 50, verbose_eval=False)

#preds_vali = model.predict(xgb.DMatrix(self.procs_data['vali_data']['x']), iteration_range=(0, model.best_iteration + 1))
#best_vali_f1 = f1_score(laundering_values['true_y'], probs_to_binary(preds_vali))

# final predictions
#preds_test = model.predict(xgb.DMatrix(self.procs_data['test_data']['x']), iteration_range=(0, model.best_iteration + 1))
#perform_metrics = metrics(y_true=laundering_values_test['true_y'], y_pred_probabilities=preds_test)