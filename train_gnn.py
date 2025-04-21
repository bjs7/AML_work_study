import utils
import trainer_gnn as tg
import data_functions as data_funcs


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


