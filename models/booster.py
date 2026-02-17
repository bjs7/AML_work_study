"""XGBoost/LightGBM model wrapper for federated learning."""

import xgboost as xgb
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)


class Booster:

    def __init__(self, hyperparams, nthread=None):
        """Initialize booster from hyper_sampler() output.

        Args:
            hyperparams: dict with 'num_rounds' and 'params' keys
                         (as returned by training.utils.hyper_sampler).
            nthread: override number of threads. Useful to limit CPU usage
                     when multiple parties train in parallel.
        """
        self.num_rounds = hyperparams['num_rounds']
        self.params = dict(hyperparams['params'])
        if nthread is not None:
            self.params['nthread'] = nthread
        self.booster = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, early_stopping_rounds=50):
        """Train from scratch with optional early stopping.

        Args:
            X_train, y_train: training data (numpy arrays or DataFrames).
            X_eval, y_eval: optional evaluation data for early stopping.
            early_stopping_rounds: stop if no improvement for this many rounds.

        Returns:
            self
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]

        if X_eval is not None and y_eval is not None:
            deval = xgb.DMatrix(X_eval, label=y_eval)
            evals.append((deval, 'eval'))

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_eval is not None else None,
            verbose_eval=False
        )
        return self

    def update_weights(self, X_train, y_train, num_rounds=1):
        """Incremental training for FL local epochs.

        If no model exists yet, trains from scratch for num_rounds.
        Otherwise continues training the existing model.

        Args:
            X_train, y_train: training data.
            num_rounds: number of boosting rounds to add.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)

        if self.booster is None:
            self.booster = xgb.train(
                self.params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
        else:
            for _ in range(num_rounds):
                self.booster.update(dtrain, self.booster.num_boosted_rounds())

    def predict(self, X):
        """Return predicted probabilities.

        Args:
            X: features (numpy array or DataFrame).

        Returns:
            numpy array of probabilities.
        """
        dmat = xgb.DMatrix(X)
        return self.booster.predict(dmat)

    def get_model_raw(self):
        """Serialize model to bytes for federated transfer."""
        buf = io.BytesIO()
        self.booster.save_raw(raw_format='json')
        buf.write(self.booster.save_raw(raw_format='json'))
        return buf.getvalue()

    def load_model_raw(self, raw_bytes):
        """Load model from serialized bytes."""
        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(raw_bytes))
