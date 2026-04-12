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


import numpy as np
import pandas as pd
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
        # If train split was empty, fall back to global scaler_encoders for eval splits
        fitted_encoders = processed_train.get('scaler_encoders') or self.scaler_encoders
        results = [processed_train]
        for data_dict in data_configs[1:]:
            results.append(feature_engi_regular_data(data_dict, self.args['data_parser'], scaler_encoders=fitted_encoders))

        return results

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








class SecureBoostPartyMixin(BoosterMixinParty):
    """Party mixin for SecureBoost vertical FL.

    Extends BoosterMixinParty with the three operations the manager needs
    during tree building and inference:
      - find_best_split: histogram-based split search over local features
      - route_samples:   left/right assignment for a given split
      - get_feature_value: single feature lookup by global index (inference)

    _global_to_local is populated by setup_secureboost_post_prep (vertical/setup.py)
    after prep_data() has been called.
    """

    def get_feature_value(self, global_idx: int, feature: str):
        """Return the value of a feature for a transaction by global index.

        Returns None if this party has no data for that transaction (triggers
        the node's default routing direction in the tree).

        Uses pre-computed numpy arrays (_feature_arrays) when available for
        fast O(1) lookup; falls back to pandas iloc otherwise.

        Args:
            global_idx: global DataFrame index of the transaction.
            feature: column name in this party's procs_data.

        Returns:
            float value, or None if not held by this party.
        """
        if not hasattr(self, '_global_to_local') or global_idx not in self._global_to_local:
            return None
        mode, local_row = self._global_to_local[global_idx]
        farr = getattr(self, '_feature_arrays', {}).get(mode, {}).get(feature)
        if farr is not None:
            return float(farr[local_row])
        # Fallback: pandas iloc (used if _feature_arrays not yet built)
        data_key = f'{mode}_data'
        if data_key not in self.procs_data:
            return None
        x = self.procs_data[data_key]['x']
        if feature not in x.columns:
            return None
        return float(x.iloc[local_row][feature])

    def find_best_split(
        self,
        global_indices: list,
        g_arr: np.ndarray,
        h_arr: np.ndarray,
        idx_to_pos: dict,
        lambda_reg: float,
        n_bins: int = 32,
    ):
        """Find the best (feature, threshold) split for this party's transactions.

        Only considers transactions in global_indices that this party holds.
        Returns only summary statistics (gain, feature name, threshold) —
        never raw feature values. This is what would be encrypted in a
        production SecureBoost deployment.

        Args:
            global_indices: global transaction indices in this tree node
                            that this party holds data for.
            g_arr: first-order gradient array indexed by idx_to_pos.
            h_arr: second-order gradient (Hessian) array indexed by idx_to_pos.
            idx_to_pos: {global_idx: position_in_g_arr} mapping.
            lambda_reg: L2 regularisation term.
            n_bins: number of candidate threshold bins per feature.

        Returns:
            (best_gain, best_feature, best_threshold)
            best_gain is 0.0 if no improving split is found.
        """
        if not global_indices:
            return 0.0, None, None

        # Collect local rows grouped by mode, preserving order for gradient alignment
        rows_by_mode = {}
        for gidx in global_indices:
            mode, local_row = self._global_to_local[gidx]
            rows_by_mode.setdefault(mode, []).append((gidx, local_row))

        # Build feature matrix and aligned gradient slices using numpy arrays
        x_parts  = []
        gidx_order = []
        farrs_cache = getattr(self, '_feature_arrays', {})

        for mode, pairs in rows_by_mode.items():
            local_rows_np = np.array([lr for _, lr in pairs])
            gidxs_part    = [gi for gi, _ in pairs]
            farr_dict     = farrs_cache.get(mode)
            if farr_dict:
                X_mode = pd.DataFrame(
                    {col: arr[local_rows_np] for col, arr in farr_dict.items()}
                )
            else:
                # Fallback to pandas iloc
                x = self.procs_data[f'{mode}_data']['x']
                X_mode = x.iloc[list(local_rows_np)].reset_index(drop=True)
            x_parts.append(X_mode)
            gidx_order.extend(gidxs_part)

        X         = pd.concat(x_parts, ignore_index=True)
        positions = np.array([idx_to_pos[gi] for gi in gidx_order])
        g_node    = g_arr[positions]
        h_node    = h_arr[positions]
        G_tot     = g_node.sum()
        H_tot     = h_node.sum()

        best_gain      = 0.0
        best_feature   = None
        best_threshold = None

        for feature in X.columns:
            vals = X[feature].values.astype(float)
            if np.unique(vals).size <= 1:
                continue

            pcts       = np.linspace(0, 100, n_bins + 2)[1:-1]
            thresholds = np.unique(np.percentile(vals, pcts))

            for thr in thresholds:
                left  = vals <= thr
                right = ~left
                if not left.any() or not right.any():
                    continue

                GL = g_node[left].sum()
                GR = g_node[right].sum()
                HL = h_node[left].sum()
                HR = h_node[right].sum()

                gain = 0.5 * (
                    GL ** 2 / (HL + lambda_reg)
                    + GR ** 2 / (HR + lambda_reg)
                    - G_tot ** 2 / (H_tot + lambda_reg)
                )

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = float(thr)

        return best_gain, best_feature, best_threshold

    def route_samples(self, global_indices: list, feature: str, threshold: float):
        """Assign global indices left or right based on feature <= threshold.

        Args:
            global_indices: indices that this party holds (all must be in _global_to_local).
            feature: feature name.
            threshold: split threshold.

        Returns:
            (left_indices, right_indices): two lists of global indices.
        """
        left, right = [], []
        for gidx in global_indices:
            val = self.get_feature_value(gidx, feature)
            if val is None:
                continue
            if val <= threshold:
                left.append(gidx)
            else:
                right.append(gidx)
        return left, right


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