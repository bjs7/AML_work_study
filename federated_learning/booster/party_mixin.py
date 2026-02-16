"""GNN-specific Party mixin providing data preparation and weight updates."""

from data.feature_engineering import feature_engi_regular_data


class BoosterMixinParty:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_feature_engineering = self.manager.args['data_parser'].ibm_fe
        self._train_for_final = self.manager.args['data_parser'].train_for_final

    def feature_engineering(self, train_data, eval_data):

        if self._skip_feature_engineering:
            return train_data, eval_data

        train_data = feature_engi_regular_data(train_data, self.args['gnn_parser'], self.scaler_encoders)
        eval_data = feature_engi_regular_data(eval_data, self.args['gnn_parser'], scaler_encoders = train_data.get('scaler_encoders'))

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
    

    def update_local_weights(self, num_local_epochs = 1):

        tr_data = self.procs_data['train_data']['df']
        # can add FL-specifics here
        for epoch in range(num_local_epochs):
            self.model.update_w(tr_data)

    def send_local_weights(self, manager):
        # in theory this functions could be dropped and the manager could just collect the parameters itself
        # though here one would apply some encrypt?
        manager.parties_weights[self.bank_id] = {param: value.data.clone() for param, value in self.model.gnn.named_parameters()}
    
    def get_eval_data(self):
        return {'df': self.procs_data['eval_data']['df'], 
                'pred_indices': self.procs_data['eval_data']['pred_indices']}


