from models.base import Model, InferenceModel, MethodMixing
from data.get_indices_type_data import get_booster_data
from data.data_utils import prep_helper
from data.feature_engi import feature_engi_regular_data
from training.tuning import Boostertune
from training.trainer_booster import BoosterTrain
import inference_saving.inference_functions as pu

from models.registry import regi_model_types, regi_booster, regi_infer_types, BOOSTER_REGISTRY


# Booster class --------------------------------------------------------------------------------------------------------------


class BoosterMixing(MethodMixing):
    def get_data(self, data, bank_indices):
        df = data[self.args.data_type]
        return get_booster_data(self.args, df, bank_indices)
    
    def _prep_helper(self, train, vali):
        return prep_helper(train, vali)
    
    def _feature_engi(self, data, **kwargs):
        return feature_engi_regular_data(data, **kwargs)


@regi_model_types('booster')
class Booster(BoosterMixing, Model):

    def __init__(self, args):
        super().__init__(args)

    def tuning(self, train_data, vali_data):
        tuner = Boostertune.from_model_type(self.args)
        return tuner.tune(train_data, vali_data)

    def train(self, train_data, vali_data, test_data, hp):
        train_data, test_data = self.pred_data_for_tr_inf(train_data, vali_data, test_data)
        trainer = BoosterTrain.from_model_type(self.args, train_data, test_data)
        return trainer.train(hp)
    
    @staticmethod
    def from_model_type(args):
        return Booster(args)
    


@regi_infer_types('booster')
class BoosterInf(BoosterMixing, InferenceModel):

    def __init__(self, args):
        super().__init__(args)
        self.main_folder, self.model_settings = pu.get_folder_path_booster(self.args)
    
    def get_model(self, test_data, model_parameters, tmp_folder):
        return pu.get_model_booster(tmp_folder)
    
    def get_predictions(self, model, test_data, model_parameters, tmp_folder, f1_values = None):
        preds = self.predict(model, test_data)
        return pu.get_preditcions_booster(preds, test_data, tmp_folder, f1_values)

    def get_test_indices_data(self, raw_data, bank):
        train_data, vali_data, test_data, bank_indices = self.get_indices_data(raw_data, bank)
        train_data, test_data = self.pred_data_for_tr_inf(train_data, vali_data, test_data)
        return test_data, bank_indices['test_indices']
    
    @staticmethod
    def from_model_type(args):
        booster_cls = BOOSTER_REGISTRY.get(args.model)
        if booster_cls is None:
            raise ValueError(f'Unknown model type: {args.model}')
        return booster_cls(args)

import models.model_classes as mcls

@regi_booster('xgboost')
class XGBoostInfer(mcls.XGBoostMixin, BoosterInf):
    pass

@regi_booster('light_gbm')
class Light_gbmInfer(mcls.LightGBMMixin, BoosterInf):
    pass

 