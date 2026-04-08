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

        if parsers['fl_parser'].tune:
            logger.info("Starting hyperparameter tuning (--tune flag set)")
            tuned_hp = self.tuning(laundering_values)
            best_hp = tuned_hp[None]['hyperparameters']
            self._save_tuned_hp(best_hp)
        else:
            best_hp = self._load_tuned_hp()
            if best_hp is None:
                raise FileNotFoundError(
                    f"No saved hyperparameters found. "
                    f"Run full_info with --tune first to generate them."
                )
            logger.info("Loaded saved hyperparameters (skipping tuning)")

        logger.info("Setup complete")
        return best_hp

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        self.init_models(hyperparameters)
        return self._party.train(hyperparameters, laundering_values_vali, laundering_values_test)
