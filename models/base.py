from abc import ABC, abstractmethod
import utils
from data.get_indices_type_data import get_indices_bdt
from models.registry import MODEL_TYPE_REGISTRY, INFERENCE_REGISTRY
import inference_saving.inference_functions as pu


# Main classes --------------------------------------------------------------------------------------------------------------

class BaseModel:
    REGISTRY = None

    def __init__(self, args):
        self.args = args
        self.args.model_type = utils.model_types.get(args.model)
        self.args.data_type = utils.data_types.get(self.args.model_type)

    def get_indices(self, data, bank=None):
        return get_indices_bdt(data, bank = bank)
    
    @classmethod
    def from_model_type(cls, args):
        model_type = utils.model_types.get(args.model)
        if model_type not in cls.REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        base_cls = cls.REGISTRY[model_type]
        return base_cls.from_model_type(args)


class Model(BaseModel, ABC):
    REGISTRY = MODEL_TYPE_REGISTRY

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def tuning(self):
        pass

    @abstractmethod
    def train(self):
        pass


class InferenceModel(BaseModel, ABC):
    REGISTRY = INFERENCE_REGISTRY

    def get_folder_params(self, bank = None):
        return pu.get_indi_para(self.main_folder, bank = bank)
    
    def get_indices_data(self, raw_data, bank = None):
        return pu.get_indices_data(self, raw_data, bank)

    @abstractmethod
    def get_data(self): pass

    @abstractmethod
    def get_model(self): pass


class MethodMixing:
    
    def feature_engineering(self, train_data, vali_data):
        train_data = self._feature_engi(train_data)
        vali_data = self._feature_engi(vali_data, scaler_encoders=train_data.get('scaler_encoders'))
        return train_data, vali_data
    
    def pred_data_for_tr_inf(self, train_data, vali_data, test_data):
        train_data = self._prep_helper(train_data, vali_data)
        return self.feature_engineering(train_data, test_data)
    
    def _prep_helper(self, train, vali):
        raise NotImplementedError

    def _feature_engi(self, data, **kwags):
        raise NotImplementedError
    

    




