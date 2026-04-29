from .manager_mixin import GNNMixinManager, GNNMixinManagerBaseline
import logging

logger = logging.getLogger(__name__)


class FullInfoGNNManager(GNNMixinManagerBaseline):
    """Full information GNN Manager - single party with complete dataset."""

    def __init__(self, args):
        super().__init__(args)
        self._party = None  # Single party reference
        
    def add_party(self, party, bank_type='train'):
        if self._party is not None:
            raise ValueError("FullInfoGNNManager only supports single party")
        super().add_party(party, bank_type=bank_type)
        self._party = party

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        logger.info("Setting up Full Info model (single party with complete dataset)")
        self._add_party(None, df, parsers, scaler_encoders)

        if parsers['data_parser'].ibm_hp:
            logger.info("Using IBM hyperparameters")
            return self.tuning(laundering_values)[None]['hyperparameters']

        if parsers['fl_parser'].tune:
            logger.info("Starting hyperparameter tuning")
            tuned_hp = self.tuning(laundering_values)
            best_hp = tuned_hp[None]['hyperparameters']
            self._save_tuned_hp(best_hp)
            return best_hp

        hp = self._load_tuned_hp()
        if hp is None:
            raise RuntimeError(
                "No saved GNN hyperparameters found. Run with --tune first, or pass --ibm_hp to use IBM defaults."
            )
        logger.info("Loaded saved hyperparameters — skipping tuning")
        return hp

    def _tuning_helper(self, laundering_values, party, bank_id):
        return self._gnn_tuning(laundering_values)
    
    def _train_party(self, laundering_values, **kwargs):
        return self._party.train(laundering_values)
        
    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        self.init_models(hyperparameters)
        return self._party.train(laundering_values_vali, laundering_values_test)
    

