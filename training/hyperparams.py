import random
import utils


# ---------------------------------------------------------------------------------------------------------------------------------------
# functions for sampling and setting hyperparameters ------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------


# simple helper -------------------------------------------------------------------------------------------------------------------------

def operator(param, parameters, j):
    if j == 0:
        return param if param < parameters else parameters
    elif j == 1:
        return param if param > parameters else parameters


def update_interval(param, interval):
        new_interval = interval.copy()
        if param < new_interval[0]:
            new_interval[0] = param
        if param > new_interval[1]:
            new_interval[1] = param
        return new_interval

# ----------------------------------------------------------------------------------------------------------------------------------------
# function for sampling hyperparameter values for gnn and boosters, used in the tuning phase ---------------------------------------------


def hyper_sampler(args, num_nodes = None, sample_intervals = None):

    if args.model == 'xgboost':
        
        device = None
        if utils.get_data_path == "/data/leuven/362/vsc36278":
            device = "cuda"

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
                "tree_method": "hist",
                "device": device,
                "random_state": 1
                }
            }
    
    elif args.model == 'light_gbm':
        
        if args.size == 'large':

            parameters = {
                'num_rounds': random.randint(32, 512),
                'params': {

                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',

                'num_leaves': random.randint(32, 256),
                'learning_rate':  random.uniform(0.001, 0.01),
                'lambda_l2': random.uniform(0.01, 0.5),
                'scale_pos_weight': random.uniform(1,10),
                'lambda_l1': random.uniform(10**(0.01), 10**(0.5)),
                "tree_method": "hist",
                "device": "gpu", 
                'random_state': 1,
                'verbose': -1
                }

            }
        
        else:
            parameters = {
                'num_rounds': random.randint(10, 1000),
                'params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',

                'num_leaves': random.randint(1, 16384),
                'learning_rate':  random.uniform(10**(-2.5), 10**(-1)),
                'lambda_l2': random.uniform(10**(-2), 10**(2)),
                'scale_pos_weight': random.uniform(1,10),
                'lambda_l1': random.uniform(10**(0.01), 10**(0.5)),
                "tree_method": "hist",
                "device": "gpu",
                'random_state': 1,
                'verbose': -1
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




# ----------------------------------------------------------------------------------------------------------------------------------------
# setting for the gfp model, that extracts graph features in the data used for the boosters ----------------------------------------------


gfpparams = {
    "num_threads": 4,
    #"time_window": 21600,
    #"time_window": 86400,

    "vertex_stats": True,         # produce vertex statistics
    "vertex_stats_cols": [3,4],     # produce vertex statistics using the selected input columns
    
    # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
    "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],  # fan,deg,ratio,avg,sum,var,skew,kurtosis

    # scatter gather parameters
    "scatter-gather": True,
    "scatter-gather_tw": 21600,
    "scatter-gather_bins": [y+2 for y in range(2)],

    # length-constrained simple cycle parameters
    "lc-cycle": False,
    "lc-cycle_tw": 86400,
    "lc-cycle_len": 10,
    "lc-cycle_bins": [y+2 for y in range(2)],

    # fan in/out parameters
    "fan": True,
    "fan_tw": 86400,
    "fan_bins": [y+2 for y in range(2)],
    
    # in/out degree parameters
    "degree": True,
    "degree_tw": 86400,
    "degree_bins": [y+2 for y in range(2)],

    # temporal cycle parameters
    "temp-cycle": True,
    "temp-cycle_tw": 86400,
    "temp-cycle_bins": [y+2 for y in range(2)],
}

