import training.utils as tr_utils
from models.booster import Booster
import os


class BoosterMixinManager:

    def init_hyperparams(self, sample_intervals=None):

        x_0 = tr_utils.get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]['x_0']
        hp_list = [tr_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list

    def init_models(self, hyperparams, bank_id=None):
        """Create Booster model(s) for parties.

        Args:
            hyperparams: dict with 'num_rounds' and 'params' from hyper_sampler.
            bank_id: if provided, only init model for that party.
                     Otherwise init for all parties.
        """
        # Divide CPU threads across parties when running in parallel
        num_parties = len(self.parties)
        total_cpus = os.cpu_count() or 1
        nthread = max(1, total_cpus // num_parties)

        if bank_id is not None:
            self.parties[bank_id].model = Booster(hyperparams, nthread=nthread)
        else:
            include_test = self.mode == 'training'
            for bid, party in self.iter_parties(include_test=include_test):
                party.model = Booster(hyperparams, nthread=nthread)
