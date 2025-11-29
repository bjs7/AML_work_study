"""Individual GNN Manager - trains each party independently with separate models."""

import copy
import numpy as np
import utils
import configs.configs as configs
from inference import metrics, predictions_helper
import inference as flin
from .manager_mixin import GNNMixinManager
from relbanks_saving_analysis.relevant_banks import get_relevant_banks
from training.utils import ibm_gnn


class IndividualGNNManager(GNNMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup fr_banks, tune them, then add sr_banks with best hyperparameters."""
        fr_banks, sr_banks = get_relevant_banks(parsers)
        
        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]
        
        # Add and tune fr_banks
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        tuned_hp = self.tuning(laundering_values)
        
        # Add sr_banks with best hyperparameters
        if self.args['data_parser'].train_for_final:
            sr_banks = []
        tuned_hp = utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders, tuned_hp)
        
        return tuned_hp

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_eval_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def tuning(self, laundering_values):

        results = {}
        self.set_mode('tuning')
        
        for bank_id, party in self.parties.items():

            if self.args['data_parser'].ibm_hp:
                tuned_hyparameters, f1_score_for_hp = ibm_gnn, 1
            else:
                party.prep_data()
                party_laundering_values = self._helper_party_tuning(party, laundering_values)
                tuned_hyparameters, f1_score_for_hp = self._gnn_tuning(party_laundering_values, bank_id = bank_id)

            results[bank_id] = {'hyperparameters': tuned_hyparameters, 'f1_score': f1_score_for_hp}
            
        return results

    def tuning_loop(self, hyperparameters_tuning, party_laundering_values, bank_id):

        best_f1 = -1
        best_hyperparameters = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams, bank_id)
            results = self.party_train(self.parties[bank_id], party_laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']

            scores.append(results['metrics']['f1'])

        return best_hyperparameters, scores, best_f1
    

    def party_train(self, party, party_laundering_values):

        best_metrics = None
        #best_preditcions = None

        best_pred_label = None
        best_pred_probabilities = None

        best_f1 = -1
        best_model = None

        party_laundering_values['pred_label'] = 0
        party_laundering_values['pred_probabilities'] = 0


        epochs = 20 if self.args['data_parser'].testing else configs.epochs

        for i in range(0, epochs):
            party.update_local_w()

            if (i+1) % 20 == 0:  

                pred_probabilities = party.model.predict(party.get_eval_data())
                tmp_metrics = metrics(party_laundering_values['true_y'], pred_probabilities)
                
                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_pred_label = predictions_helper(pred_probabilities)
                    best_pred_probabilities = copy.deepcopy(pred_probabilities)
                    best_model = copy.deepcopy(party.model.gnn.state_dict())
                    best_f1 = tmp_metrics['f1']

        party_laundering_values['pred_label'] = best_pred_label
        party_laundering_values['pred_probabilities'] = best_pred_probabilities
        
        return {'model': best_model, 'metrics': best_metrics, 'laundering_values': party_laundering_values}

    def train(self, hyperparameters, laundering_values, seeds = 4):

        self.set_mode('training')
        results_by_seed = {}

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            utils.set_seed(seed + 1)
            # Pass a fresh copy to each seed iteration
            results_by_seed[(seed + 1)] = self._train(hyperparameters, copy.deepcopy(laundering_values))

        return results_by_seed

    def _train(self, hyperparameters, laundering_values):

        models_hyperparameters = {}
        party_predictions = {}

        for bank_id, party in self.parties.items():
            self.init_models(hyperparameters[bank_id], bank_id)
            party_laundering_values = self._helper_party_tuning(party, laundering_values)

            # train individual party
            tmp_model = self.party_train(party, party_laundering_values)
            models_hyperparameters[bank_id] = {'model': tmp_model['model'], 
                                               'hyperparameters': hyperparameters[bank_id]}
            
            party_predictions[bank_id] = tmp_model['laundering_values']['pred_probabilities']
        

        for bank_id, party in self.parties.items():
            party.model.gnn.load_state_dict(models_hyperparameters[bank_id]['model'])
            flin.update_laundering_values(party, laundering_values, pred_probabilities=party_predictions[bank_id])

        collective_metrics = metrics(y_true = laundering_values['true_y'], 
                                     y_pred_probabilities = laundering_values['avg_prob'],
                                     y_pred_binary = laundering_values['pred_label'])

        return {
            'metrics': collective_metrics,
            'laundering_values': copy.deepcopy(laundering_values),
            'models': models_hyperparameters
        }




