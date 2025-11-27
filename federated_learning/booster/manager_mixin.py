import training.utils as tr_utils


class BoosterMixinManager:

    def init_hyperparams(self, sample_intervals = None):
            
        x_0 = tr_utils.get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]['x_0']
        hp_list = [tr_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    

    def init_models(self):
        return 0
    
    