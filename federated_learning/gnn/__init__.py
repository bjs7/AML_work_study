"""GNN-specific federated learning implementations."""

from .party_mixin import GNNMixinParty, GNNMixinPartyFL, GNNMixinPartyIndi, GNNMixinPartyVert
from .manager_mixin import GNNMixinManager
from .communication import GNNCommunicationMixin
from .federated_manager import FLGNNManager, FLGNNManagerVertical, FLGNNManagerVerticalSimple
from .individual_manager import IndividualGNNManager
from .full_info_manager import FullInfoGNNManager

__all__ = [
    'GNNMixinParty',
    'GNNMixinPartyFL',
    'GNNMixinPartyIndi',
    'GNNMixinPartyVert',
    'GNNMixinManager',
    'GNNCommunicationMixin',
    'FLGNNManager',
    'FLGNNManagerVertical',
    'FLGNNManagerVerticalSimple',
    'IndividualGNNManager',
    'FullInfoGNNManager',
]

#    'GNNMixinParty_Individual',
#    'GNNMixinParty_Full_info',