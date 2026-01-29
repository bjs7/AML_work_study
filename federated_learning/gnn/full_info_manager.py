from .manager_mixin import GNNMixinManager, GNNMixinManager_Fullinfo_Indi
import logging

logger = logging.getLogger(__name__)


class FullInfoGNNManager(GNNMixinManager_Fullinfo_Indi):
    """Full information GNN Manager - single party with complete dataset."""

    def __init__(self, args):
        super().__init__(args)
        self._party = None  # Single party reference
        self._train_for_final = self.args['data_parser'].train_for_final
        
    def add_party(self, party, is_sr=False):
        if self._party is not None:
            raise ValueError("FullInfoGNNManager only supports single party")
        super().add_party(party, is_sr=is_sr)
        self._party = party

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        logger.info("Setting up Full Info model (single party with complete dataset)")
        self._add_party(None, df, parsers, scaler_encoders)
        
        logger.info("Starting hyperparameter tuning")
        tuned_hp = self.tuning(laundering_values)
        logger.info("Setup complete")

        return tuned_hp[None]['hyperparameters']

    def _tuning_helper(self, laundering_values, party, bank_id):
        return self._gnn_tuning(laundering_values)
    
    def _train_party(self, laundering_values, **kwargs):
        return self._party.train(laundering_values)
        
    def _train_helper(self, hyperparameters, laundering_values):
        self.init_models(hyperparameters)
        return self._party.train(laundering_values)
    

