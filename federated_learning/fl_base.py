# packages
import argparse
from typing import Dict, Any
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from abc import ABC, abstractmethod


class BaseFL:
    REGISTRY = None

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = None



class Party(BaseFL):

    REGISTRY = FL_ALGO_REGISTRY_PARTY

    """Individual computation node that can send/receive messages"""
    
    def __init__(self, args, bank_id, data, indices, manager, scaler_encoders) -> None:
        super().__init__(args)
        self.bank_id = bank_id
        self.indices = indices
        self.data = data
        self.manager = manager
        self.scaler_encoders = scaler_encoders
        self.model = None
        self.tr_configs = {}  

        if self.manager:
            self.manager.add_party(self)
        
    @classmethod
    def get_algo_class(cls, parsers: argparse.Namespace, **kwargs) -> classmethod:

        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(args = parsers, **kwargs)
    
    def get_eval_indices(self):
        return self.indices['vali_indices'] if self.mode == 'tuning' else self.indices['test_indices']

    @abstractmethod
    def prep_data(self):
        """Prepare and process data for training/evaluation."""
        pass

    @abstractmethod
    def get_eval_data(self):
        """Get evaluation data based on current mode (tuning/training)."""
        pass

    @abstractmethod
    def update_local_w(self):
        """Update local model weights."""
        pass

    @abstractmethod
    def feature_engineering(self, train_data, eval_data):
        """Apply feature engineering to training and evaluation data."""
        pass

from data.get_indices_type_data import get_indices_bdt
import data.data_functions as dfn

class Manager(BaseFL, ABC):
    REGISTRY = FL_ALGO_REGISTRY_MANAGER

    def __init__(self, args):
        super().__init__(args)
        self.parties: Dict[int, Party] = {}
        self.parties_weights: Dict[int, Any] = {}
        self.global_weights = None

    @classmethod
    def get_algo_class(cls, parsers):
        """Get the appropriate manager class from registry based on algorithm and model type."""
        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(parsers)

    def add_party(self, party: Party):
        self.parties[party.bank_id] = party
        print(f"[Manager] added: party {party.bank_id}")

    def _add_party(self, bank_id, df, parsers, scaler_encoders):
        """Create and add a single party to this manager."""
        bank_indices = get_indices_bdt(df, bank=bank_id)
        train_data, vali_data, test_data = dfn.fl_get_data(parsers, df, bank_indices)
        tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
        Party.get_algo_class(parsers = parsers, 
                             bank_id=bank_id, 
                             data=tmp_data, 
                             indices=bank_indices, 
                             manager=self, 
                             scaler_encoders=scaler_encoders
                             )

    def set_mode(self, mode):
        self.mode = mode
        for _, party in self.parties.items():
            party.mode = mode

    @abstractmethod
    def init_models(self):
        pass
    
    @abstractmethod
    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup parties for this scenario. Override in subclasses."""
        pass
    


