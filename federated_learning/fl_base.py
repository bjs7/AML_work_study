# packages
import argparse
from typing import Dict, Any
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from abc import ABC, abstractmethod
import torch


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
        self.train_configs = {}
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
    
    def get_vali_indices(self):
        return self.indices['vali_indices']

    def get_test_indices(self):
        return self.indices['test_indices']

    @abstractmethod
    def prep_data(self):
        """Prepare and process data for training/evaluation."""
        pass

    @abstractmethod
    def get_vali_data(self):
        """Get validation data based on current mode (tuning/training)."""
        pass

    @abstractmethod
    def update_local_weights(self):
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
        self.bank_device = {}

    @classmethod
    def get_algo_class(cls, parsers):
        """Get the appropriate manager class from registry based on algorithm and model type."""
        algo_class = f"{parsers['fl_parser'].fl_algo}_{parsers['fl_parser'].model_type}"
        if algo_class not in cls.REGISTRY:
            raise ValueError(f"Unknown algo type: {algo_class}")
        base_cls = cls.REGISTRY[algo_class]
        return base_cls.return_class(parsers)

    def add_party(self, party: Party, bank_type = 'train'):
        
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

    def iter_parties(self, include_test=False):
        """Iterate over all relevant party groups dynamically.

        Always yields from self.parties. Yields from self.vali_parties
        if it has entries. Yields from self.test_parties only when
        include_test=True and it has entries.

        Deduplicates by bank_id - each bank is yielded only once,
        prioritizing train > vali > test.
        """
        seen = set()
        for bank_id, party in self.parties.items():
            seen.add(bank_id)
            yield bank_id, party
        if self.vali_parties:
            for bank_id, party in self.vali_parties.items():
                if bank_id not in seen:
                    seen.add(bank_id)
                    yield bank_id, party
        if include_test and self.test_parties:
            for bank_id, party in self.test_parties.items():
                if bank_id not in seen:
                    yield bank_id, party

    def set_mode(self, mode):
        self.mode = mode
        for _, party in self.iter_parties(include_test=True):
            party.mode = mode

    def assign_device_to_party(self):

        data_type = ['train_data', 'vali_data']
        parties_holder = [self.parties, self.vali_parties]
        num_gpus = torch.cuda.device_count()
        sums = [0] * num_gpus

        if self.test_parties:
            data_type += ['test_data']
            parties_holder += [self.test_parties]

        for parties, type_data in zip(parties_holder, data_type):
            sums = self._assign_device_to_party(sums, parties, type_data)

    def _assign_device_to_party(self, sums, parties, type_data):
        
        party_lengths = {}        
        for bank_id, party in parties.items():
            if bank_id in self.bank_device:
                continue
            party_lengths[bank_id] = party.data[type_data]['df'].edge_attr.shape[0]
        sorted_party_lengths = sorted(party_lengths.items(), key=lambda item: item[1], reverse=True)

        for bank_id, length in sorted_party_lengths:
            if bank_id in self.bank_device:
                continue
            add_index = sums.index(min(sums))
            sums[add_index] += length
            self.bank_device[bank_id] = add_index

        return sums


    @abstractmethod
    def init_models(self):
        pass
    
    @abstractmethod
    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup parties for this scenario. Override in subclasses."""
        pass
    


