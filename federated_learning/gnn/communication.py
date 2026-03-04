"""Communication mixin for federated GNN training - handles weight synchronization."""

from concurrent.futures import ThreadPoolExecutor


class GNNCommunicationMixin:

    def send_global_weights(self, condition=None):
        include_test = self.mode == 'training'

        # Group parties by assigned GPU device so each thread owns one device
        parties_by_device = {}
        for bank_id, party in self.iter_parties(include_test=include_test):
            if condition and not condition(bank_id):
                continue
            device_idx = self.bank_device.get(bank_id, 0)
            if device_idx not in parties_by_device:
                parties_by_device[device_idx] = []
            parties_by_device[device_idx].append(party)

        def _send_to_parties(parties_list):
            for party in parties_list:
                for param, value in party.model.gnn.named_parameters():
                    value.data = self.global_weights[param].data.clone().to(value.device)

        if len(parties_by_device) <= 1:
            # Single device or no GPU — run sequentially
            _send_to_parties(next(iter(parties_by_device.values()), []))
        else:
            # Parallel across GPUs — PyTorch releases GIL for CUDA ops
            with ThreadPoolExecutor(max_workers=len(parties_by_device)) as executor:
                futures = [executor.submit(_send_to_parties, parties)
                           for parties in parties_by_device.values()]
                for f in futures:
                    f.result()

    def send_global_weights_params(self):
        bank_0_id, bank_0 = next(iter(self.parties.items()))
        condition = lambda bank_id: bank_0_id != bank_id
        self.send_global_weights(condition)

    def get_global_weights(self):
        self._bank_0, bank_0 = next(iter(self.parties.items()))
        self.global_weights = {param: value.data.clone().cpu() for param, value in bank_0.model.gnn.named_parameters()}

    def update_global_weights(self, party_weights_map=None):
        """Aggregate local weights into global weights.

        Args:
            party_weights_map: Dict mapping bank_id -> float weight for this round.
                If None, uses uniform 1/N over all parties in self.parties_weights.
        """
        if party_weights_map is None:
            n = len(self.parties_weights)
            party_weights_map = {bank: 1.0 / n for bank in self.parties_weights}

        first = True
        for bank, weights in self.parties_weights.items():
            w_k = party_weights_map[bank]
            for param, value in weights.items():
                if first:
                    self.global_weights[param].data = value.data * w_k
                else:
                    self.global_weights[param].data += value.data * w_k
            first = False




