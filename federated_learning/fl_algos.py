"""Federated Learning algorithm implementations and registry combiners."""

from federated_learning.fl_base import Manager, Party
from federated_learning.registry import regi_algo_manager, regi_algo_party
from federated_learning.gnn import GNNMixinParty, GNNMixinPartyVert, FLGNNManager, IndividualGNNManager, FullInfoGNNManager, FLGNNManagerVertical #GNNMixinParty_Individual, GNNMixinParty_Full_info, 
from federated_learning.booster.individual_manager import IndividualBoosterManager
from federated_learning.booster.party_mixin import BoosterMixinParty

# Regression classes - to be implemented later
# Placeholder stubs for now
class RegressionMixinParty:
    """Regression-specific Party mixin - TODO: move to regression.py"""
    pass

class RegressionMixinManager:
    """Regression-specific Manager mixin - TODO: move to regression.py"""
    pass


# -------------------------------------------
# FL Algos ----------------------------------
# -------------------------------------------

# FedGraph -------------------------------------------------------

class FedGraph_party(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)

# FedAvg -------------------------------------------------------

# FedAvg (Federated Averaging) implementation
class FedAvg_party(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)


class FedAvg_manager(Manager):

    def get_adjacency_matrix(self):
        return 0
    

class FedGraph_manager(Manager):

    def get_adjacency_matrix(self):
        return 0

# FedProx -------------------------------------------------------

class FedProx_party(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FedProx_manager(Manager):

    def get_adjacency_matrix(self):
        return 0


# ------------------------------------------------------------------------------------------

# -------------------------------------------
# Combiners ---------------------------------
# -------------------------------------------

# Regression ---------------------------------------------------------------

@regi_algo_party('FedAvg_regression')
class FedAvg_Regression_Party(RegressionMixinParty, FedAvg_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedAvg_Regression_Party(**kwargs)


@regi_algo_manager("FedAvg_regression")
class FedAvg_Regression_Manager(RegressionMixinManager, FedAvg_manager):

    @staticmethod
    def return_class(args):
        return FedAvg_Regression_Manager(args)
    

# GNN ---------------------------------------------------------------------

# Party --------------------------

@regi_algo_party("full_info_gnn")
class FullInfo_GNN_Party(GNNMixinParty, Party):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FullInfo_GNN_Party(**kwargs)

#FullInfo_GNN_Party.__mro__

@regi_algo_party("individual_gnn")
class Individual_GNN_Party(GNNMixinParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return Individual_GNN_Party(**kwargs)

#Individual_GNN_Party.__mro__

@regi_algo_party("FedAvg_gnn")
class FedAvg_GNN_Party(GNNMixinParty, FedAvg_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedAvg_GNN_Party(**kwargs)
    

@regi_algo_party("FedProx_gnn")
class FedProx_GNN_Party(GNNMixinParty, FedProx_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedProx_GNN_Party(**kwargs)


@regi_algo_party("FedGraph_gnn")
class FedGraph_GNN_Party(GNNMixinPartyVert, FedGraph_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def return_class(**kwargs):
        return FedGraph_GNN_Party(**kwargs)


# Manager --------------------------

@regi_algo_manager('full_info_gnn')
class FullInfo_GNN_Manager(FullInfoGNNManager, Manager):

    @staticmethod
    def return_class(args):
        return FullInfo_GNN_Manager(args)
    

@regi_algo_manager("individual_gnn")
class Individual_GNN_Manager(IndividualGNNManager, Manager): #FLGNNManager #GNNMixinManager

    @staticmethod
    def return_class(args):
        return Individual_GNN_Manager(args)



@regi_algo_manager("FedAvg_gnn")
class FedAvg_GNN_Manager(FLGNNManager, FedAvg_manager):

    @staticmethod
    def return_class(args):
        return FedAvg_GNN_Manager(args)


@regi_algo_manager("FedProx_gnn")
class FedProx_GNN_Manager(FLGNNManager, FedProx_manager):

    @staticmethod
    def return_class(args):
        return FedProx_GNN_Manager(args)


@regi_algo_manager("FedGraph_gnn")
class FedGraph_GNN_Manager(FLGNNManagerVertical, FedGraph_manager): #FLGNNManager #GNNMixinManager

    @staticmethod
    def return_class(args):
        return FedGraph_GNN_Manager(args)




# Booster -------------------------------------------------------------------------------------------

# Party ---------------

@regi_algo_party("individual_booster")
class Individual_Booster_Party(BoosterMixinParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return Individual_Booster_Party(**kwargs)






#class BoosterMixinParty:
    #def __init__(self, **kwargs):
        #super().__init__(**kwargs)





# Manager ---------------

class BoosterMixinManager:


    def init_hyperparams(self):
        pass

    def _get_relevant_parameters(self):
        pass


@regi_algo_manager("individual_booster")
class Individual_Booster_Manager(IndividualBoosterManager, Manager): #FLGNNManager #GNNMixinManager

    @staticmethod
    def return_class(args):
        return Individual_Booster_Manager(args)





"""
    def _get_relevant_parameters(self):

        self.params_update = []
        _, bank_0 = next(iter(self.parties.items()))

        for name, param in bank_0.model.gnn.named_parameters():
            self.params_update.append(name)
"""

