from models.base import Model, InferenceModel, MethodMixing
from data.get_indices_type_data import get_graph_data #get_indices_bdt, 
from data.data_utils import prep_helper
from data.feature_engi import feature_engi_graph_data
#from training.tuning import tune_gnn
from training.trainer_gnn import train_gnn
import inference_saving.inference_functions as pu
from training.tuning import GNNTune
import training.gnn_utils as tgu
from inference_saving.inference_functions import GNNpredictions

from models.registry import regi_model_types, regi_infer_types

# GNN class ---------------------------------------------------------------------------------------------------------------------

class GNNMixing(MethodMixing):

    def get_data(self, data, bank_indices):
        df = data[self.args.data_type]
        return get_graph_data(self.args, df, bank_indices)

    def _feature_engi(self, data, **kwargs):
        return feature_engi_graph_data(data, **kwargs)
    
    def _prep_helper(self, train, vali):
        return train
        


@regi_model_types('graph')
class GNN(GNNMixing, Model):

    def __init__(self, args):
        super().__init__(args)
    
    def tuning(self, train_data, vali_data):
        train_data, vali_data = self.feature_engineering(train_data, vali_data)
        tuner = GNNTune(self.args, train_data, vali_data)
        return tuner.tune()

    def train(self, train_data, vali_data, test_data, hype_params, seeds = 4):
        vali_data, test_data = self.feature_engineering(vali_data, test_data)
        return train_gnn(self.args, vali_data, test_data, hype_params, seeds)
    
    @staticmethod
    def from_model_type(args):
        return GNN(args)
    

@regi_infer_types('graph')
class GNNInf(GNNMixing, InferenceModel):

    def __init__(self, args):
        super().__init__(args)
        self.main_folder, self.model_settings = pu.get_folder_path_gnn(self.args)

    def get_test_indices_data(self, raw_data, bank = None):
        train_data, vali_data, test_data, bank_indices = self.get_indices_data(raw_data, bank)
        train_data, test_data = self.feature_engineering(vali_data, test_data)
        return test_data, bank_indices['test_indices']

    def get_model(self, test_data, model_parameters, tmp_folder):
        return tgu.get_model(test_data['df'], model_parameters['params'], self.model_settings['model_settings'], self.args)
    
    def get_predictions(self, model, test_data, model_parameters, tmp_folder, f1_values = None):
        predictor = GNNpredictions(test_data, self.model_settings, model_parameters, tmp_folder)
        return predictor.get_predictions(model, f1_values)
    
    @staticmethod
    def from_model_type(args):
        return GNNInf(args)
    
    

