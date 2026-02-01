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

    def __init__(self, args, bank_id, data, indices, manager, scaler_encoders, bank_type='train') -> None:
        super().__init__(args)
        self.bank_id = bank_id
        self.indices = indices
        self.data = data
        self.manager = manager
        self.scaler_encoders = scaler_encoders
        self.model = None
        self.tr_configs = {}
        self.bank_type = bank_type  # Flag to indicate if this is an sr_party
        self.edge_feat_start = self.manager.edge_feat_start

        if self.manager:
            self.manager.add_party(self, bank_type=bank_type)
        
    @classmethod
    def get_algo_class(cls, parsers: argparse.Namespace, **kwargs) -> classmethod:

        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(args = parsers, **kwargs)
    
    def get_eval_indices(self):
        return self.indices['vali_indices']

    def get_test_indices(self):
        return self.indices['test_indices']

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
        self.vali_parties: Dict[int, Party] = {}  # Parties with data only in validation/test (e.g., sr_banks)
        self.test_parties: Dict[int, Party] = {}
        self.parties_weights: Dict[int, Any] = {}
        self.global_weights = None
        self.edge_feat_start = 1 if self.args['fl_parser'].fl_algo == 'FedGraph' else 0

    @classmethod
    def get_algo_class(cls, parsers):
        """Get the appropriate manager class from registry based on algorithm and model type."""
        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(parsers)

    def add_party(self, party: Party, bank_type=False):
        
        if bank_type == 'train':
            self.parties[party.bank_id] = party
        elif bank_type == 'vali':
            self.vali_parties[party.bank_id] = party
        else:
            self.test_parties[party.bank_id] = party

        #print(f"[Manager] added: party {party.bank_id}" + (" (sr_party)" if is_sr else ""))

    def _add_party(self, bank_id, df, parsers, scaler_encoders, bank_type='train'):
        """Create and add a single party to this manager.

        Args:
            bank_id: ID of the bank/party to add
            df: Data dictionary
            parsers: Parser configurations
            scaler_encoders: Encoder objects for feature engineering
            bank_type: 'train', 'vali', or 'test' — which party dict to add to
        """
        bank_indices = get_indices_bdt(df, bank=bank_id)
        train_data, vali_data, test_data = dfn.fl_get_data(parsers, df, bank_indices)
        tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
        Party.get_algo_class(parsers = parsers,
                            bank_id=bank_id,
                            data=tmp_data,
                            indices=bank_indices,
                            manager=self,
                            scaler_encoders=scaler_encoders,
                            bank_type=bank_type
                            )

    def get_parties_for_mode(self, mode):
        if mode == 'train':
            return self.parties
        elif mode == 'vali':
            return self.vali_parties
        else:
            return self.test_parties

    def set_mode(self, mode):
        self.mode = mode
        if self.test_parties:
            for _, party in self.test_parties.items():
                party.mode = mode
        elif self.vali_parties:
            for _, party in self.vali_parties.items():
                party.mode = mode
        else:
            for _, party in self.parties.items():
                party.mode = mode

    @abstractmethod
    def init_models(self):
        pass
    
    @abstractmethod
    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup parties for this scenario. Override in subclasses."""
        pass
    


