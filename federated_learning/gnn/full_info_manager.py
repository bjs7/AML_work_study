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
        
        self._party.prep_data()
        
        for seed in range(seeds):
            utils.set_seed(seed + 1)
            self.init_models(hyperparameters)
            results_by_seed[(seed + 1)] = self._train(copy.deepcopy(laundering_values))
        
        return results_by_seed
    
    def _train(self, laundering_values):
        """Training loop for single party - no loops over parties."""
        best_metrics = None
        best_predictions = None
        best_f1 = -1
        best_model = None
        laundering_values['predictions_fl'] = 0
        
        epochs = 20 if self.args['data_parser'].testing else configs.epochs
        
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
    

