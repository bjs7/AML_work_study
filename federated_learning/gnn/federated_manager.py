"""Federated GNN Manager - coordinates training across all parties with weight aggregation."""

import copy
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from relbanks_saving_analysis.relevant_banks import get_relevant_banks
from .communication import GNNCommunicationMixin
from .manager_mixin import GNNMixinManager
from training.utils import ibm_gnn


class FLGNNManager(GNNCommunicationMixin, GNNMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):

        fr_banks, sr_banks = get_relevant_banks(parsers)

        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]

        # Add and tune fr_banks
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        tuned_hp, _ = self.tuning(laundering_values)

        # Add sr_banks
        if self.args['data_parser'].train_for_final:
            sr_banks = []
        utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders)

        return tuned_hp

    def tuning(self, laundering_values):

        # --------------------------------

        if self.args['data_parser'].ibm_hp:
            return ibm_gnn, None

        self.set_mode('tuning')

        for bank_id, party in self.parties.items():
            party.prep_data()

        return self._gnn_tuning(laundering_values)

    def tuning_loop(self, hyperparameters_tuning, laundering_values):

        best_f1 = -1
        best_hyperparameters = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams)
            self.get_global_w()
            self.send_global_w_params()

            # if reg or graph epochs is used. Or also is for decision trees, yes?
            # just update in one, and then another for sending to manager?
            results = self.fl_training(laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])

        return best_hyperparameters, scores, best_f1
    
    def train(self, hyperparameters, laundering_values, seeds = 4):

        self.set_mode('training')
        results_by_seed = {}
        #best_f1 = -1; best_model = None

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            utils.set_seed(seed + 1)
            results_by_seed[(seed + 1)] = self._train(hyperparameters, copy.deepcopy(laundering_values))
        
        return results_by_seed
    
    def _train(self, hyperparameters, laundering_values):

        self.init_models(hyperparameters)
        self.get_global_w()
        self.send_global_w_params()

        return self.fl_training(laundering_values)
    
    def fl_training(self, laundering_values):

        best_w = None
        best_metrics = None
        best_f1 = -1

        epochs = 20 if self.args['data_parser'].testing else configs.epochs

        for i in range(0, epochs):

            for bank_id, party in self.parties.items():    
                # little unsure if I should update parameters like this, or if 
                # it should be kept inside the party

                party.update_local_w()
                party.send_local_w(self)
            
            self.update_global_w()
            self.send_global_w()

            # I need to reset laundering_values or something every time new parameters are tested
            # inference / status
            if (i+1) % 20 == 0:

                # reset preditcions
                for col in ['pred_label', 'pred_probabilities', 'num_prob', 'avg_prob', 'max_prob']:
                    laundering_values[col] = 0
                
                for bank_id, party in self.parties.items():
                    flin.update_laundering_values(party, laundering_values)

                tmp_metrics = metrics(y_true = laundering_values['true_y'], 
                                      y_pred_probabilities = laundering_values['avg_prob'], 
                                      y_pred_binary = laundering_values['pred_label'])

                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_laundering_values = copy.deepcopy(laundering_values)
                    best_w = copy.deepcopy(self.global_w)
                    best_f1 = tmp_metrics['f1']

        return {'w': best_w, 'metrics': best_metrics, 'laundering_values': best_laundering_values}

