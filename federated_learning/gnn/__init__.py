"""GNN-specific federated learning implementations."""

from .party_mixin import GNNMixinParty, GNNMixinPartyHorizontal, GNNMixinPartyBaseline, GNNMixinPartyVertical
from .manager_mixin import GNNMixinManager
from .communication import GNNCommunicationMixin
from .federated_manager import FLGNNManagerHorizontal, FLGNNManagerVertical, FLGNNManagerVerticalSimple
from .individual_manager import IndividualGNNManager
from .full_info_manager import FullInfoGNNManager

__all__ = [
    'GNNMixinParty',
    'GNNMixinPartyHorizontal',
    'GNNMixinPartyBaseline',
    'GNNMixinPartyVertical',
    'GNNMixinManager',
    'GNNCommunicationMixin',
    'FLGNNManagerHorizontal',
    'FLGNNManagerVertical',
    'FLGNNManagerVerticalSimple',
    'IndividualGNNManager',
    'FullInfoGNNManager',
]

#    'GNNMixinParty_Individual',
#    'GNNMixinParty_Full_info',