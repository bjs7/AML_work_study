
import copy
import numpy as np
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from .manager_mixin import BoosterMixinManager
from data.relevant_banks import get_relevant_banks


class IndividualBoosterManager(BoosterMixinManager):

    def tuning(self):
        pass

    def tuning_loop(self):
        pass
    
    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup fr_banks, tune them, then add sr_banks with best hyperparameters."""
        fr_banks, sr_banks = get_relevant_banks(parsers)
        
        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]
        
        # Add and tune fr_banks
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        #tuned_hp = self.tuning(laundering_values)
        
        # Add sr_banks with best hyperparameters
        #tuned_hp = utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders, tuned_hp)
        
        #return tuned_hp