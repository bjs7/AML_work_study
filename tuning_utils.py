import random
import json

def get_tuning_configs(args):

    with open('tuning_configs.json', 'r') as file:
        model_parameters = json.load(file)

    return model_parameters.get(args.model)


def operator(param, parameters, j):
    if j == 0:
        return param if param < parameters else parameters
    elif j == 1:
        return param if param > parameters else parameters



def hyper_sampler(args, num_nodes = None, sample_intervals = None):

    if args.model == 'xgboost':
        parameters = {
            "num_rounds": random.randint(10, 1000),
            "params": {
                "objective":  "binary:logistic",
                "eval_metric": "logloss",
                
                "max_depth": random.randint(1, 15), 
                "learning_rate": random.uniform(10**(-2.5), 10**(-1)),
                "lambda": random.uniform(10**(-2), 10**(2)),
                "scale_pos_weight": random.uniform(1, 10),
                "colsample_bytree": random.uniform(0.5, 1.0),
                "subsample": random.uniform(0.5, 1.0),
                "random_state": 1
                }
            }
    elif args.model == 'GINe':

        if not sample_intervals:
            hid_em_size_interval = [16, 72]
            lr_interval = [0.005, 0.05]
            gnn_layer_interval = [2, 4]
            dropout_interval = [0, 0.5]
            w_ce2_interval = [6,8]
        else:
            hid_em_size_interval = sample_intervals.get('hid_em_size_interval')
            lr_interval = sample_intervals.get('lr_interval')
            gnn_layer_interval = sample_intervals.get('gnn_layer_interval')
            dropout_interval = sample_intervals.get('dropout_interval')
            w_ce2_interval = sample_intervals.get('w_ce2_interval')

        parameters = {
            'params': {
            'batch_size': 4096,
            'num_neighbors': [100],

            'hidden_embedding_size': random.randint(hid_em_size_interval[0], hid_em_size_interval[1]),
            'learning rate': random.uniform(lr_interval[0], lr_interval[1]),
            'gnn_layers': random.randint(gnn_layer_interval[0], gnn_layer_interval[1]),
            'dropout': random.uniform(dropout_interval[0], dropout_interval[1]),
            'w_ce1': 1.0000182882773443,
            'w_ce2': random.uniform(w_ce2_interval[0], w_ce2_interval[1])

            }
            #,'model_settings': {'index_masking': False, 'include_time': False}

        }
        if args.scenario == 'full_info':
            parameters.get('params')['num_neighbors'] *= parameters.get('params')['gnn_layers']
        else:
            if num_nodes >= 25e3:
                #if plus 25k
                num_neighbors = [20, 20, 10, 5]
            elif num_nodes >= 10e3:
                # if plus 10k
                num_neighbors = [20, 15, 5, 5]
            else:
                num_neighbors = None
                
            #num_neighbors = [20, 15, 10, 5]

            if num_neighbors:
                parameters.get('params')['num_neighbors'] = num_neighbors[0:parameters.get('params')['gnn_layers']]
            else:
                parameters.get('params')['num_neighbors'] = num_neighbors
                parameters.get('params')['batch_size'] = 1

    return parameters


"""

elif num_nodes >= 7.5e3:
                # if plus 7.5k
                num_neighbors = [15, 15, 5, 5]
            elif num_nodes >= 5e3:
                # if plus 5k
                num_neighbors = [15, 10, 5, 5]
            elif num_nodes >= 4e3:
                # if plus 4k
                num_neighbors = [25, 5, 5, 5]
            elif num_nodes >= 3e3:
                # if plus 3k
                num_neighbors = [20, 5, 5, 5]
            elif num_nodes >= 2e3:
                # if plus 2k
                num_neighbors = [15, 5, 5, 5]
            elif num_nodes >= 1e3:
                # if plus 1k
                num_neighbors = [10, 5, 5, 5]
            else:
                # if less than 1k
                num_neighbors = [5, 5, 5, 5]



"""