"""Federated Learning algorithm implementations and registry combiners."""

from federated_learning.fl_base import Manager, Party
from federated_learning.registry import regi_algo_manager, regi_algo_party
from federated_learning.gnn import GNNMixinParty, FLGNNManager, IndividualGNNManager, FullInfoGNNManager
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

# FedAVG ------------------------------------------------------


# FedGD -------------------------------------------------------

# this is just for FedGD, but right now I have built for fedavg?
class FedGD_party(Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def send_messages(self, recipient, content):
        return super().send_messages(recipient, content)
    

class FedGD_manager(Manager):

    def get_adjacency_matrix(self):
        return 0

    
# ------------------------------------------------------------------------------------------

# -------------------------------------------
# Combiners ---------------------------------
# -------------------------------------------

# Regression ---------------------------------------------------------------

@regi_algo_party('FedGD_regression')
class FedGD_Regression_Party(RegressionMixinParty, FedGD_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FedGD_Regression_Party(**kwargs)


@regi_algo_manager("FedGD_regression")
class FedGD_Regression_Manager(RegressionMixinManager, FedGD_manager):

    @staticmethod
    def return_class(args):
        return FedGD_Regression_Manager(args)
    

# GNN ---------------------------------------------------------------------

# Party --------------------------

@regi_algo_party("full_info_gnn")
class FullInfo_GNN_Party(GNNMixinParty, Party):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FullInfo_GNN_Party(**kwargs)


@regi_algo_party("individual_gnn")
class Individual_GNN_Party(GNNMixinParty, Party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return Individual_GNN_Party(**kwargs)


@regi_algo_party("FedGD_gnn")
class FedGD_GNN_Party(GNNMixinParty, FedGD_party):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def return_class(**kwargs):
        return FedGD_GNN_Party(**kwargs)


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


@regi_algo_manager("FedGD_gnn")
class FedGD_GNN_Manager(FLGNNManager, FedGD_manager): #FLGNNManager #GNNMixinManager

    @staticmethod
    def return_class(args):
        return FedGD_GNN_Manager(args)







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

