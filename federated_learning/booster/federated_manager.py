




class FLBoosterManager(BoosterMixinManager):

    def tuning_loop(self):
        pass


    def tuning(self):
        # for boost it would want it inside the hyperparameter loop.
        # however, if spliting and processing several datapoints at once, then
        # maybe it could be avoided, however too much memory usage?
        # So don't do that, find solution to only do once for reg and gnn?
        # will see at booster trees
        pass

