import utils
import trainer_gnn as tg
import trainer_utils as tu
import data_functions as data_funcs
import pandas as pd
import numpy as np
import xgboost as xgb
import tuning as tune

def train_model(args, train_data, vali_data, test_data, hyperparameters):
    
    model_type = tu.model_types.get(args.model)

    if model_type == 'graph':
        models = train_gnn(args, vali_data, test_data, hyperparameters)
    elif model_type == 'booster':
        models = train_booster(train_data, vali_data, test_data, hyperparameters)
    
    return models


def train_gnn(args, train_data, test_data, hyperparameters, seeds = 4):
    model_type = 'graph'
    train_data, test_data = data_funcs.general_feature_engineering(model_type, train_data, test_data)

    models = {}
    for seed in range(1, seeds + 1):
        utils.set_seed(seed)
        model, f1 = tg.gnn_trainer(args, train_data, test_data, hyperparameters, None)
        models[f'seed_{seed}'] = None
        models[f'seed_{seed}'] = {'model': model, 'f1': f1}

    return models


def train_booster(train_data, vali_data, test_data, hyperparameters):

    train_data = {
        'x': pd.concat([train_data['x'], vali_data['x']]).reset_index(drop=True),
        'y': np.concatenate([train_data['y'], vali_data['y']])
    }
    train_data, test_data = data_funcs.general_feature_engineering('booster', train_data, test_data)
    
    utils.set_seed(1)
    models = {}
    dtrain = xgb.DMatrix(train_data['x'], train_data['y'])
    model = xgb.train(hyperparameters['params'], dtrain, hyperparameters['num_rounds'])
    preds = model.predict(xgb.DMatrix(test_data['x']))
    f1 = tune.f1_eval(preds, test_data)
    models['seed_1'] = {'model': model, 'f1': f1}

    return models
    
    





