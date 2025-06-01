import utils
from data.feature_engi import general_feature_engineering
import training.tuning as tune
from abc import ABC, abstractmethod
import models.model_classes as mcls


# ---------------------------------------------------------------------------------------------------------------------------------------
# Code for training boosters ------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


class BoosterTrain(ABC):

    def __init__(self, args, train_data, test_data):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        
    def train(self, hp):
        utils.set_seed(1)
        models = {}
        booster_data = self.pred_data(self.train_data)
        model = self.train_model(hp['params'], booster_data, hp['num_rounds'])
        preds = self.predict(model, self.test_data)
        f1 = tune.f1_eval(preds, self.test_data)
        models['seed_1'] = {'model': model, 'f1': f1}

        return models
    
    @classmethod
    def from_model_type(cls, args, *train_args):
        model_cls = BOOSTER_REGISTRY.get(args.model)
        if model_cls is None:
            raise ValueError(f'Unknown model type: {args.model}')
        return model_cls(args, *train_args)

BOOSTER_REGISTRY = {}
def regi_booster(name):
    def wrapper(cls):
        BOOSTER_REGISTRY[name] = cls
        return cls
    return wrapper

@regi_booster('xgboost')
class XGBoosttrain(mcls.XGBoostMixin, BoosterTrain):
    pass

@regi_booster('light_gbm')
class XGBoosttrain(mcls.LightGBMMixin, BoosterTrain):
    pass


