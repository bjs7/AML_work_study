"""Full Info Booster Manager - single party with complete dataset."""

import logging
from .manager_mixin import BoosterMixinManager

logger = logging.getLogger(__name__)


class FullInfoBoosterManager(BoosterMixinManager):
    """Single party with complete dataset — no federation."""

    def __init__(self, args):
        super().__init__(args)
        self._party = None

    def add_party(self, party, bank_type='train'):
        if self._party is not None:
            raise ValueError("FullInfoBoosterManager only supports a single party")
        super().add_party(party, bank_type=bank_type)
        self._party = party

    def _tuning_helper(self, laundering_values, party, bank_id):
        return self.tune(party, laundering_values)

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        logger.info("Setting up Full Info Booster model (single party with complete dataset)")
        self._add_party(None, df, parsers, scaler_encoders)

        logger.info("Starting hyperparameter tuning")
        tuned_hp = self.tuning(laundering_values)
        logger.info("Setup complete")

        return tuned_hp[None]['hyperparameters']

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        self.init_models(hyperparameters)
        return self._party.train(hyperparameters, laundering_values_vali, laundering_values_test)
