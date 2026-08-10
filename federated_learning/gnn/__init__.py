"""GNN-specific federated learning implementations."""

from .party_mixin import GNNMixinParty, GNNMixinPartyHorizontal, GNNMixinPartyBaseline, GNNMixinPartyFedAvgSplit
from .manager_mixin import GNNMixinManager
from .communication import GNNCommunicationMixin
from .federated_manager import FLGNNManagerHorizontal, FLGNNManagerFedGraph, FLGNNManagerSplitFed, FLGNNManagerFedAvgSplit
from .individual_manager import IndividualGNNManager
from .full_info_manager import FullInfoGNNManager

__all__ = [
    'GNNMixinParty',
    'GNNMixinPartyHorizontal',
    'GNNMixinPartyBaseline',
    'GNNMixinPartyFedAvgSplit',
    'GNNMixinManager',
    'GNNCommunicationMixin',
    'FLGNNManagerHorizontal',
    'FLGNNManagerFedGraph',
    'FLGNNManagerSplitFed',
    'FLGNNManagerFedAvgSplit',
    'IndividualGNNManager',
    'FullInfoGNNManager',
]

#    'GNNMixinParty_Individual',
#    'GNNMixinParty_Full_info',