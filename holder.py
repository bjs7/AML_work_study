# holder for old individual training


"""Individual GNN Manager - trains each party independently with separate models."""

import copy
import numpy as np
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from .manager_mixin import GNNMixinManager
from relbanks_saving_analysis.relevant_banks import get_relevant_banks


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
        tuned_hp = utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders, tuned_hp)
        
        return tuned_hp

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_eval_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def tuning(self, laundering_values):

        self.set_mode('tuning')
        results = {}
        for bank_id, party in self.parties.items():
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
        best_preditcions = None
        best_f1 = -1
        best_model = None
        party_laundering_values['predictions_fl'] = 0

        epochs = 20 if self.args['data_parser'].testing else configs.epochs_fl

        for i in range(0, epochs):
            party.update_local_w()

            if (i+1) % 20 == 0:  

                predictions = party.model.predict_binary(party.get_eval_data())
                tmp_metrics = metrics(party_laundering_values['true_y'], predictions)
                
                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_preditcions = copy.deepcopy(predictions)
                    best_model = copy.deepcopy(party.model.gnn.state_dict())
                    best_f1 = tmp_metrics['f1']

        party_laundering_values['predictions_fl'] = best_preditcions
        
        return {'model': best_model, 'metrics': best_metrics, 'laundering_values': party_laundering_values}


    def train(self, hyperparameters, laundering_values):

        self.set_mode('training')
        results, models_hyperparameters, party_predictions = {}, {}, {}

        for bank_id, party in self.parties.items():
            party.prep_data()
            self.init_models(hyperparameters[bank_id], bank_id)
            party_laundering_values = self._helper_party_tuning(party, laundering_values)

            tmp_model = self._train(party, party_laundering_values)
            models_hyperparameters[bank_id] = {'model': tmp_model['model'], 
                                               'hyperparameters': hyperparameters[bank_id]}
            
            party_predictions[bank_id] = tmp_model['laundering_values']['predictions_fl']

        for bank_id, party in self.parties.items():
            flin.update_laundering_values(party, laundering_values, predictions=party_predictions[bank_id])

        results['metrics'] = metrics(laundering_values['true_y'], laundering_values['predictions_fl'])
        results['laundering_values'] = laundering_values
        results['models'] = models_hyperparameters

        return results
    
    def _train(self, party, party_laundering_values, seeds = 4):

        best_f1 = -1; best_model = None

        for seed in range(seeds):
            utils.set_seed(seed + 1)
            current_model = self.party_train(party, party_laundering_values)

            if current_model['metrics']['f1'] > best_f1:
                best_model = copy.deepcopy(current_model)
                best_f1 = current_model['metrics']['f1']

        if best_model is not None:
            party.model.gnn.load_state_dict(best_model['model'])

        return best_model





# full info manager -------------------------------------------

from .manager_mixin import GNNMixinManager
import copy
import utils
import configs.configs as configs
from inference import metrics
import inference as flin
from training.utils import ibm_gnn


class FullInfoGNNManager(GNNMixinManager):
    """Full information GNN Manager - single party with complete dataset."""
    
    def __init__(self, args):
        super().__init__(args)
        self._party = None  # Single party reference
    
    
    def add_party(self, party):
        """Override to ensure only one party."""
        if self._party is not None:
            raise ValueError("FullInfoGNNManager only supports single party")
        super().add_party(party)
        self._party = party

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup single party with all data."""
        self._add_party(None, df, parsers, scaler_encoders)
        return self.tuning(laundering_values)
    
    def tuning(self, laundering_values):
        """Simplified tuning for single party."""

        if self.args['data_parser'].ibm_hp:
            return ibm_gnn

        self.set_mode('tuning')
        self._party.prep_data()

        return self._gnn_tuning(laundering_values)
    
    
    def tuning_loop(self, hyperparameters_tuning, laundering_values):
        """Tuning loop for single party."""
        best_f1 = -1
        best_hyperparameters = None
        scores = []
        
        for hyperparams in hyperparameters_tuning:
            self.init_models(hyperparams)
            results = self._train(laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])
        
        return best_hyperparameters, scores
    
    def train(self, hyperparameters, laundering_values, seeds=4):
        """Train with seed loop."""
        self.set_mode('training')
        results_by_seed = {}
        
        #best_f1 = -1
        #best_model = None
        
        self._party.prep_data()
        
        for seed in range(seeds):
            utils.set_seed(seed + 1)
            self.init_models(hyperparameters)
            results_by_seed[(seed + 1)] = self._train(copy.deepcopy(laundering_values))
            
            #if current_model['metrics']['f1'] > best_f1:
            #    best_model = copy.deepcopy(current_model)
            #    best_f1 = current_model['metrics']['f1']
        
        return results_by_seed
    
    def train(self, hyperparameters, laundering_values, seeds=4):
        """Train with seed loop."""
        self.set_mode('training')
        best_f1 = -1
        best_model = None
        
        self._party.prep_data()
        
        for seed in range(seeds):
            utils.set_seed(seed + 1)
            self.init_models(hyperparameters)
            current_model = self._train(laundering_values)
            
            if current_model['metrics']['f1'] > best_f1:
                best_model = copy.deepcopy(current_model)
                best_f1 = current_model['metrics']['f1']
        
        return best_model
    
    def _train(self, laundering_values):
        """Training loop for single party - no loops over parties."""
        best_metrics = None
        best_predictions = None
        best_f1 = -1
        best_model = None
        laundering_values['predictions_fl'] = 0
        
        epochs = 20 if self.args['data_parser'].testing else configs.epochs_fl
        
        for epoch in range(epochs):
            self._party.update_local_w()
            
            if (epoch + 1) % 20 == 0:
                predictions = self._party.model.predict_binary(self._party.get_eval_data())
                tmp_metrics = metrics(laundering_values['true_y'], predictions)
                
                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_predictions = copy.deepcopy(predictions)
                    best_model = copy.deepcopy(self._party.model.gnn.state_dict())
                    best_f1 = tmp_metrics['f1']
        
        if best_model is None:
            raise ValueError(f"No evaluation occurred during training (epochs={epochs}). Check evaluation frequency.")
    
        # Update laundering values with best predictions
        laundering_values['predictions_fl'] = best_predictions
        
        return {'metrics': best_metrics, 
                'laundering_values': copy.deepcopy(laundering_values), 
                'model': best_model}
    



