"""Federated Learning algorithm implementations and registry combiners."""

from federated_learning.fl_base import Manager, Party
from federated_learning.registry import regi_algo_manager, regi_algo_party
from federated_learning.gnn import (
    GNNMixinPartyHorizontal, GNNMixinPartyBaseline, GNNMixinPartyVertical,
    FLGNNManagerHorizontal, FLGNNManagerVertical, FLGNNManagerVerticalSimple,
    IndividualGNNManager as IndividualGNNManagerImpl,
    FullInfoGNNManager as FullInfoGNNManagerImpl,
)
from federated_learning.booster.individual_manager import IndividualBoosterManager as IndividualBoosterManagerImpl
from federated_learning.booster.federated_manager import FLBoosterManager
from federated_learning.booster.full_info_manager import FullInfoBoosterManager as FullInfoBoosterManagerImpl
from federated_learning.booster.secureboost_manager import SecureBoostManager as SecureBoostManagerImpl
from federated_learning.booster.party_mixin import BoosterMixinParty, SecureBoostPartyMixin

# Regression classes - to be implemented later
# Placeholder stubs for now
class RegressionMixinParty:
    """Regression-specific Party mixin - TODO: move to regression.py"""
    pass

class RegressionMixinManager:
    """Regression-specific Manager mixin - TODO: move to regression.py"""
    pass


# -------------------------------------------
# FL Base Protocol Classes ------------------
# -------------------------------------------

class FedGraphPartyBase(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)


class FedAvgPartyBase(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)


class FedAvgManagerBase(Manager):

    def get_adjacency_matrix(self):
        return 0


class FedGraphManagerBase(Manager):

    def get_adjacency_matrix(self):
        return 0


class FedProxPartyBase(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FedProxManagerBase(Manager):

    def get_adjacency_matrix(self):
        return 0


# -------------------------------------------
# Combiners ---------------------------------
# -------------------------------------------

# Regression ---------------------------------------------------------------

@regi_algo_party('FedAvg_regression')
class FedAvgRegressionParty(RegressionMixinParty, FedAvgPartyBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedAvgRegressionParty(**kwargs)


@regi_algo_manager("FedAvg_regression")
class FedAvgRegressionManager(RegressionMixinManager, FedAvgManagerBase):

    @staticmethod
    def return_class(args):
        return FedAvgRegressionManager(args)


# GNN ---------------------------------------------------------------------

# Party --------------------------

@regi_algo_party("full_info_gnn")
class FullInfoGNNParty(GNNMixinPartyBaseline, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FullInfoGNNParty(**kwargs)


@regi_algo_party("individual_gnn")
class IndividualGNNParty(GNNMixinPartyBaseline, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return IndividualGNNParty(**kwargs)


@regi_algo_party("FedAvg_gnn")
class FedAvgGNNParty(GNNMixinPartyHorizontal, FedAvgPartyBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedAvgGNNParty(**kwargs)


@regi_algo_party("FedProx_gnn")
class FedProxGNNParty(GNNMixinPartyHorizontal, FedProxPartyBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedProxGNNParty(**kwargs)


@regi_algo_party("FedGraph_gnn")
class FedGraphGNNParty(GNNMixinPartyVertical, FedGraphPartyBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedGraphGNNParty(**kwargs)


@regi_algo_party("FedGraphSimple_gnn")
class FedGraphSimpleGNNParty(GNNMixinPartyVertical, FedGraphPartyBase):
    """Party for simplified vertical FL — same data prep as FedGraph."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedGraphSimpleGNNParty(**kwargs)


# Manager --------------------------

@regi_algo_manager('full_info_gnn')
class FullInfoGNNManager(FullInfoGNNManagerImpl, Manager):

    @staticmethod
    def return_class(args):
        return FullInfoGNNManager(args)


@regi_algo_manager("individual_gnn")
class IndividualGNNManager(IndividualGNNManagerImpl, Manager):

    @staticmethod
    def return_class(args):
        return IndividualGNNManager(args)


@regi_algo_manager("FedAvg_gnn")
class FedAvgGNNManager(FLGNNManagerHorizontal, FedAvgManagerBase):

    @staticmethod
    def return_class(args):
        return FedAvgGNNManager(args)


@regi_algo_manager("FedProx_gnn")
class FedProxGNNManager(FLGNNManagerHorizontal, FedProxManagerBase):

    @staticmethod
    def return_class(args):
        return FedProxGNNManager(args)


@regi_algo_manager("FedGraph_gnn")
class FedGraphGNNManager(FLGNNManagerVertical, FedGraphManagerBase):

    @staticmethod
    def return_class(args):
        return FedGraphGNNManager(args)


@regi_algo_manager("FedGraphSimple_gnn")
class FedGraphSimpleGNNManager(FLGNNManagerVerticalSimple, FedGraphManagerBase):
    """Manager for simplified vertical FL (no per-layer embedding exchange)."""

    @staticmethod
    def return_class(args):
        return FedGraphSimpleGNNManager(args)


# Booster -------------------------------------------------------------------------------------------

# Party ---------------

@regi_algo_party("individual_booster")
class IndividualBoosterParty(BoosterMixinParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return IndividualBoosterParty(**kwargs)


@regi_algo_party("FedAvg_booster")
class FedAvgBoosterParty(BoosterMixinParty, FedAvgPartyBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedAvgBoosterParty(**kwargs)


@regi_algo_party("full_info_booster")
class FullInfoBoosterParty(BoosterMixinParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FullInfoBoosterParty(**kwargs)


@regi_algo_party("SecureBoost_booster")
class SecureBoostParty(SecureBoostPartyMixin, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return SecureBoostParty(**kwargs)


# Manager ---------------

@regi_algo_manager("individual_booster")
class IndividualBoosterManager(IndividualBoosterManagerImpl, Manager):

    @staticmethod
    def return_class(args):
        return IndividualBoosterManager(args)


@regi_algo_manager("FedAvg_booster")
class FedAvgBoosterManager(FLBoosterManager, FedAvgManagerBase):

    @staticmethod
    def return_class(args):
        return FedAvgBoosterManager(args)


@regi_algo_manager("full_info_booster")
class FullInfoBoosterManager(FullInfoBoosterManagerImpl, Manager):

    @staticmethod
    def return_class(args):
        return FullInfoBoosterManager(args)


@regi_algo_manager("SecureBoost_booster")
class SecureBoostManager(SecureBoostManagerImpl, Manager):

    @staticmethod
    def return_class(args):
        return SecureBoostManager(args)
