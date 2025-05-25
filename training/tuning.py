#import trainer_utils as tu
from data.feature_engi import general_feature_engineering
from sklearn.metrics import f1_score
#import training.trainer_gnn as tg
import training.hyperparams as tune_u
import utils
from abc import ABC, abstractmethod
import models.model_classes as mcls
from training.trainer_gnn import GNNTrain

class GNNTune(ABC):
    
    def __init__(self, args, train_data, vali_data):
        self.args = args
        self.train_data = train_data
        self.vali_data = vali_data

    def tune(self):
        top_params = self._loop_tuning(top=5)
        search_space = self._get_search_space(top_params)
        best_params = self._loop_tuning(sampler_intervals=search_space, top=1)
        return best_params[0]

    def _loop_tuning(self, sampler_intervals = None, top = 5):
        
        x_0 = utils.get_tuning_configs(self.args).get(self.args.scenario).get(self.args.size).get('x_0')
        params_list = [tune_u.hyper_sampler(self.args, self.train_data['df'].num_nodes, sample_intervals = sampler_intervals) for i in range(x_0)]

        scores = []
        for i, param in enumerate(params_list):
            utils.set_seed(self.args.seed + 1)
            trainer = GNNTrain(self.args, self.train_data, self.vali_data, param)
            _, f1 = trainer.train()
            scores.append(f1)

        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top]
        
        return [params_list[i] for i in params_to_keep]
    
    def _get_search_space(self, params_list):

        intervals = {
            'hid_em_size_interval': [1e9, -1e9],
            'lr_interval': [1e9, -1e9],
            'gnn_layer_interval': [1e9, -1e9],
            'dropout_interval': [1e9, -1e9],
            'w_ce2_interval': [1e9, -1e9],
        }

        for params in params_list:
            p = params['params']
            intervals['hid_em_size_interval'] = tune_u.update_interval(p['hidden_embedding_size'], intervals['hid_em_size_interval'])
            intervals['lr_interval'] = tune_u.update_interval(p['learning rate'], intervals['lr_interval'])
            intervals['gnn_layer_interval'] = tune_u.update_interval(p['gnn_layers'], intervals['gnn_layer_interval'])
            intervals['dropout_interval'] = tune_u.update_interval(p['dropout'], intervals['dropout_interval'])
            intervals['w_ce2_interval'] = tune_u.update_interval(p['w_ce2'], intervals['w_ce2_interval'])
        
        return intervals


# Tuning functions for xgboost ----------------------------------------------------------------------------------------------------------------

def f1_eval(pred, vali_data):
    return f1_score(vali_data['y'], (pred >= 0.5).astype(int), average='binary', zero_division=0)

class Boostertune(ABC):

    def __init__(self, args):
        self.args = args
        tuning_configs = utils.get_tuning_configs(args)
        self.x_0 = tuning_configs[args.scenario][args.size]['x_0']
        self.eta = tuning_configs[args.scenario][args.size]['eta']
        self.r_0 = tuning_configs[args.scenario][args.size]['r_0']
    
    @abstractmethod
    def pred_data(self, tune_train_data): pass

    @abstractmethod
    def train_model(self, params, tune_train_data, num_rounds): pass

    @abstractmethod
    def predict(self, model, tune_vali_data): pass

    def tune(self, train_data, vali_data):

        frac_not_reached = True
        model_hyper_params = [tune_u.hyper_sampler(self.args, None) for _ in range(self.x_0)]

        while frac_not_reached:

            f1s = []
            self.r_0 = min(self.r_0, 1)
            index_slice = round(train_data['x'].shape[0] * self.r_0)
            sliced_data = {'x': train_data['x'].iloc[:index_slice,:], 'y': train_data['y'][:index_slice]}
            tune_train_data, tune_vali_data = general_feature_engineering('booster', sliced_data, vali_data)
            booster_data = self.pred_data(tune_train_data)
            
            for hp in model_hyper_params:
                try: 
                    model = self.train_model(hp['params'], booster_data, hp['num_rounds'])
                    preds = self.predict(model, tune_vali_data)
                    f1s.append(f1_eval(preds, tune_vali_data))
                except Exception as e:
                    print(f"Error with hyperparameters {hp}: {str(e)}")
                    f1s.append(0)  # Append 0 for F1 score if training or prediction fails

            self.x_0 = max(1, round(self.x_0/self.eta))
            params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:self.x_0]
            model_hyper_params = [model_hyper_params[i] for i in params_to_keep]            

            if self.r_0 >= 1 or self.x_0 == 1:
                frac_not_reached = False
            self.r_0 *= self.eta
        
        return model_hyper_params[0]
    
    @staticmethod
    def from_model_type(args):
        cls = BOOSTER_REGISTRY.get(args.model)
        if cls is None:
            raise ValueError(f'Unknown model type: {args.model}')
        return cls(args)


BOOSTER_REGISTRY = {}
def regi_booster(name):
    def wrapper(cls):
        BOOSTER_REGISTRY[name] = cls
        return cls
    return wrapper

@regi_booster('xgboost')
class XGBoosttune(mcls.XGBoostMixin, Boostertune):
    pass

@regi_booster('light_gbm')
class Light_gbmtune(mcls.LightGBMMixin, Boostertune):
    pass

    




"""

#get_top = max(1, round(self.x_0/self.eta)) if self.r_0 != 0 else 1
#params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:get_top]
#model_hyper_params = [model_hyper_params[i] for i in params_to_keep]

def tuning(args, train_data, vali_data, bank_indices):

    model_type = tu.model_types.get(args.model)

    if model_type == 'graph':

        # apply feature engineering / preprocessing
        train_data, vali_data = data_funcs.general_feature_engineering(model_type, train_data, vali_data)
        
        # tune to obtain tuned hyperparameters
        hyper_params = tune_gnn(args, train_data, vali_data)
        tuned_model = hyper_params[0]
        
    elif model_type == 'booster':
        hyper_params = tune_booster(args, train_data, vali_data)
        tuned_model = hyper_params[0]
    
    return tuned_model


    def tune_booster(args, train_data, vali_data):

    tuning_configs = tune_u.get_tuning_configs(args)
    x_0, eta, r_0 = tuning_configs.get(args.scenario).get(args.size).get('x_0'), tuning_configs.get(args.scenario).get(args.size).get('eta'), tuning_configs.get(args.scenario).get(args.size).get('r_0')
    
    frac_not_reached = True
    model_hyper_params = [tune_u.hyper_sampler(args, None) for _ in range(x_0)]

    while frac_not_reached:

        f1s = []

        if r_0 > 1:
            r_0 = 1

        index_slice = round(train_data['x'].shape[0] * r_0)
        tmp_data = {'x': train_data['x'].iloc[:index_slice,:], 'y': train_data['y'][:index_slice]}
        tune_train_data, tune_vali_data = data_funcs.general_feature_engineering('booster', tmp_data, vali_data)

        dtrain = xgb.DMatrix(tune_train_data['x'], tune_train_data['y'])
        
        for i in range(x_0):
                model = xgb.train(model_hyper_params[i]['params'], dtrain, model_hyper_params[i]['num_rounds'])
                preds = model.predict(xgb.DMatrix(tune_vali_data['x']))
                f1s.append(f1_eval(preds, tune_vali_data))

        get_top = max(1, round(round(x_0/eta))) if r_0 != 0 else 1
        params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:get_top]
        model_hyper_params = [model_hyper_params[i] for i in params_to_keep]

        if r_0 >= 1 or get_top == 1:
            frac_not_reached = False

        x_0 = max(1, round(round(x_0/eta)))
        r_0 *= eta
    
    return model_hyper_params[0]


"""


"""

def tuning(args, data, bank_indices):

    model_type = tu.model_types.get(args.model)
    if model_type == 'graph':
        data1 = data_funcs.get_train_vali_test_data(args, data, bank_indices)
        
        hyper_params = tune_gnn(args, data)
        tuned_model = hyper_params[0]
    elif model_type == 'booster':
        tuned_model = tune_booster(args, data, bank_indices)
    
    return tuned_model


"""


"""
    #temp_model = tra.train_model(args, data, train_data_indices, model_hyper_params[0], bank = bank, r_0 = 1)
    #return temp_model, model_hyper_params[0]
    
    while frac_not_reached:
        
        scores_test = []
        if r_0 > 1:
            r_0 = 1
    
        for i in range(len(model_hyper_params)):

            temp_model = tra.train_model(args, data, train_data_indices, model_hyper_params[i], bank = bank, r_0 = r_0)

            model_type = tu.model_types.get(args.model)
            vali_data = data[tu.data_types.get(model_type)]['vali_data']

            data_processor = tu.data_functions.get(model_type)
            vali_data = data_processor(vali_data, vali_data_indices, args, temp_model.scaler, temp_model.encoder_pay, temp_model.encoder_cur)

            pred = temp_model.model.predict(xgb.DMatrix(vali_data['X']))
            scores_test.append(f1_score(vali_data['y'], (pred >= 0.5).astype(int), average='binary'))

        get_top = 1 if r_0 == 1 else round(x_0/eta)
        params_to_keep = sorted(range(len(scores_test)), key=lambda i: scores_test[i], reverse=True)[:get_top]
        model_hyper_params = [models_paras for index, models_paras in enumerate(model_hyper_params) if index in params_to_keep]

        if r_0 >= 1:
            frac_not_reached = False

        x_0 = round(x_0/eta)
        r_0 *= eta
    
    temp_model = tra.train_model(args, data, train_data_indices, model_hyper_params[0], bank = bank, r_0 = 1)

    return temp_model, model_hyper_params[0]
"""


