"""GNN-specific Party mixin providing data preparation and weight updates."""

from data.feature_engi import feature_engi_graph_data


class GNNMixinParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_feature_engineering = self.manager.args['data_parser'].ibm_fe
        self._train_for_final = self.manager.args['data_parser'].train_for_final
        #self._get_batch_configs()

    def _get_batch_configs(self):

        # uses test data to decide whether batching or not, as should be consistent and be the same for train, validation and testing. 
        # Should check for research on batch trained networks generalizing to full sample.
        #graph density should potentially be incorporated here and adjuset/set batch size, num neighbos based on that. #num_neighbors = [20, 20, 10, 5] #num_neighbors = [20, 15, 5, 5]
        if self.data['test_data']['df'].num_nodes >= 250e3:
            self.tr_configs['num_neighbors'] = [5, 4, 3, 2]
            self.tr_configs['batch_size'] = 2048 #2048 #4096 8192

        elif self.data['test_data']['df'].num_nodes >= 100e3:
            self.tr_configs['num_neighbors'] = [5, 4, 3, 2]
            self.tr_configs['batch_size'] = 1024 #512  

        else:
            self.tr_configs['num_neighbors'] = None
            self.tr_configs['batch_size'] = 0


    def feature_engineering(self, train_data, eval_data):

        if self._skip_feature_engineering:
            return train_data, eval_data

        train_data = feature_engi_graph_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        eval_data = feature_engi_graph_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

        return train_data, eval_data

    def prep_data(self):

        if self.mode == 'tuning':
            train_proc, eval_proc = self.feature_engineering(self.data['train_data'], 
                                                             self.data['vali_data'])
        elif self.mode == 'training':
            tr_data = self.data['vali_data'] if not self._train_for_final else self.data['train_data']
            train_proc, eval_proc = self.feature_engineering(tr_data, 
                                                             self.data['test_data'])
            
        self.procs_data = {'train_data': train_proc, 'eval_data': eval_proc}
    

    def update_local_w(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']['df']
        # can add FL-specifics here
        for epoch in range(num_local_epochs):
            self.model.update_w(tr_data)

    def send_local_w(self, manager):
        # in theory this functions could be dropped and the manager could just collect the parameters itself
        # though here one would apply some encrypt?
        manager.parties_w[self.bank_id] = {param: value.data.clone() for param, value in self.model.gnn.named_parameters()}
    
    def get_eval_data(self):
        return {'df': self.procs_data['eval_data']['df'], 
                'pred_indices': self.procs_data['eval_data']['pred_indices']}


