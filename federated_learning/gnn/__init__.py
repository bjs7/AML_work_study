"""GNN-specific federated learning implementations."""

from .party_mixin import GNNMixinParty #, GNNMixinParty_Individual, GNNMixinParty_Full_info
from .manager_mixin import GNNMixinManager
from .communication import GNNCommunicationMixin
from .federated_manager import FLGNNManager, FLGNNManagerVertical
from .individual_manager import IndividualGNNManager
from .full_info_manager import FullInfoGNNManager

__all__ = [
    'GNNMixinParty',
    'GNNMixinManager',
    'GNNCommunicationMixin',
    'FLGNNManager',
    'FLGNNManagerVertical',
    'IndividualGNNManager',
    'FullInfoGNNManager',
]

#    'GNNMixinParty_Individual',
#    'GNNMixinParty_Full_info',