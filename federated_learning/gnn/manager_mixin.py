"""Shared GNN Manager functionality for hyperparameter tuning and model initialization."""

import training.utils as tr_utils
from models.gnn import GNN



class GNNMixinManager:

    def init_hyperparams(self, sample_intervals = None):
            
        x_0 = tr_utils.get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]['x_0']
        hp_list = [tr_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    
    def init_models(self, hyperparams, bank_id = None):

        sample_party = next(iter(self.parties.values()))
        node_features = sample_party.procs_data['train_data']['df'].x.shape[1]
        edge_dim = sample_party.procs_data['train_data']['df'].edge_attr.shape[1]
   
        if bank_id:
            self.parties[bank_id].model = GNN(self, hyperparams, node_features, edge_dim)
        else:
            for bank_id, party in self.parties.items():
                party.model = GNN(self, hyperparams, node_features, edge_dim)

        # need to 'save' parameters from model 1 as starting, or not, because the models will vary
        # Here get the relevant parameters etc. or it can be placed in get_global_w

    def _gnn_tuning(self, laundering_values, **kwargs):
        
        # first loop
        hyperparameters_tuning = self.init_hyperparams()
        _, scores, _ = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        # get top 5 of parameters and set sample space
        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        # second loop
        hyperparameters_tuning = self.init_hyperparams(sample_space)
        tuned_hyparameters, _, f1_score_for_hp = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        return tuned_hyparameters, f1_score_for_hp


    def _get_search_space(self, hyperparameters_tuning):

        intervals = {
            'hid_em_size_interval': [float('inf'), float('-inf')],
            'lr_interval': [float('inf'), float('-inf')],
            'gnn_layer_interval': [float('inf'), float('-inf')],
            'dropout_interval': [float('inf'), float('-inf')],
            'final_dropout_interval': [float('inf'), float('-inf')],
            'w_ce2_interval': [float('inf'), float('-inf')],
        }

        for params in hyperparameters_tuning:
            intervals['hid_em_size_interval'] = tr_utils.update_interval(params['hidden_embedding_size'], intervals['hid_em_size_interval'])
            intervals['lr_interval'] = tr_utils.update_interval(params['learning_rate'], intervals['lr_interval'])
            intervals['gnn_layer_interval'] = tr_utils.update_interval(params['num_gnn_layers'], intervals['gnn_layer_interval'])
            intervals['dropout_interval'] = tr_utils.update_interval(params['dropout'], intervals['dropout_interval'])
            intervals['final_dropout_interval'] = tr_utils.update_interval(params['final_dropout'], intervals['final_dropout_interval'])
            intervals['w_ce2_interval'] = tr_utils.update_interval(params['w_ce2'], intervals['w_ce2_interval'])

        return intervals
