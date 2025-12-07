"""Communication mixin for federated GNN training - handles weight synchronization."""



class GNNCommunicationMixin:

    def send_global_weights(self, condition = None):
        for bank_id, party in self.parties.items():
            if condition and not condition(bank_id): continue
            for param, value in party.model.gnn.named_parameters():
                value.data = self.global_weights[param].data.clone()

    def send_global_weights_params(self):
        bank_0_id, bank_0 = next(iter(self.parties.items()))
        condition = lambda bank_id: bank_0_id != bank_id
        self.send_global_weights(condition)

    def get_global_weights(self):
        self._bank_0, bank_0 = next(iter(self.parties.items()))
        self.global_weights = {param: value.data.clone() for param, value in bank_0.model.gnn.named_parameters()}

    def update_global_weights(self):

        for bank, weights in self.parties_weights.items():
            for param, value in weights.items():
                if bank == self._bank_0:
                    self.global_weights[param].data = value.data.clone() / self._num_parties
                else:
                    self.global_weights[param].data += value.data.clone() / self._num_parties




