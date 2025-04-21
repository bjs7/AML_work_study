import trainer_utils as tu
import data_functions as data_funcs
import xgboost as xgb
from sklearn.metrics import f1_score
import trainer_gnn as tg
import tuning_utils as tune_u


# General tuning functions ------------------------------------------------------------------------------------------------------------


def tuning(args, train_data, vali_data, bank_indices):

    model_type = tu.model_types.get(args.model)

    if model_type == 'graph':

        # apply feature engineering / preprocessing
        train_data, vali_data = data_funcs.general_feature_engineering(model_type, train_data, vali_data)
        
        # tune to obtain tuned hyperparameters
        hyper_params = tune_gnn(args, train_data, vali_data)
        tuned_model = hyper_params[0]
        
    elif model_type == 'booster':
        tuned_model = tune_booster(args, data, bank_indices)
    
    return tuned_model


# Tuning functions for GNN ------------------------------------------------------------------------------------------------------------



def loop_tuning(args, train_data, vali_data, sampler_intervals = None, top = 5):

    f1s = []
    x_0 = args.graph_tuning_x_0
    model_hyper_params = [tune_u.hyper_sampler(args, sample_intervals = sampler_intervals) for i in range(x_0)]
    models = []

    for i in range(x_0):
        model, f1 = tg.gnn_trainer(args, train_data, vali_data, model_hyper_params[i])
        models.append(model)
        f1s.append(f1)

    params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:top]
    #model_hyper_params = [models_paras for index, models_paras in enumerate(model_hyper_params) if index in params_to_keep]
    model_hyper_params = [model_hyper_params[i] for i in params_to_keep]

    return model_hyper_params


def tune_gnn(args, train_data, vali_data):
    
    #################
    #### ROUND 1 ####
    #################

    # set up list of hyperparameters
    model_hyper_params = loop_tuning(args, train_data, vali_data, top = 5)
    
    ##########################################
    ## Prepare sample intervals for round 2 ##
    ##########################################

    hidden_embedding_size_interval = [1e9,-1e9]
    learning_rate_interval = [1e9,-1e9]
    gnn_layers_interval = [1e9,-1e9]
    dropout_interval = [1e9,-1e9]
    w_ce2_interval = [1e9,-1e9]    

    for i in range(len(model_hyper_params)):
        tmp_params = model_hyper_params[i].get('params')
        for j in range(2):
            hidden_embedding_size_interval[j] = tune_u.operator(tmp_params.get('hidden_embedding_size'), hidden_embedding_size_interval[j], j)
            learning_rate_interval[j] = tune_u.operator(tmp_params.get('learning rate'), learning_rate_interval[j], j)
            gnn_layers_interval[j] = tune_u.operator(tmp_params.get('gnn_layers'), gnn_layers_interval[j], j)
            dropout_interval[j] = tune_u.operator(tmp_params.get('dropout'), dropout_interval[j], j)
            w_ce2_interval[j] = tune_u.operator(tmp_params.get('w_ce2'), w_ce2_interval[j], j)

    sampler_intervals = {'hid_em_size_interval': hidden_embedding_size_interval, 
                        'lr_interval': learning_rate_interval, 'gnn_layer_interval': gnn_layers_interval, 
                        'dropout_interval': dropout_interval, 'w_ce2_interval': w_ce2_interval}
    
    #################
    #### ROUND 2 ####
    #################

    model_hyper_params = loop_tuning(args, train_data, vali_data, sampler_intervals, top = 1)

    return model_hyper_params



# Tuning functions for xgboost ----------------------------------------------------------------------------------------------------------------

def f1_eval(pred, vali_data):
    return f1_score(vali_data['y'], (pred >= 0.5).astype(int), average='binary')

def tune_booster(args, data, bank_indices = None):

    x_0 = 10
    r_0 = 0.5
    eta = 1.5
    frac_not_reached = True

    # need to select data over and over, as the fraction increases for train, 
    # and therefore also effects validation

    #models = [trs.xgboost(args, tune_u.hyper_sampler(args)) for _ in range(x_0)]

    while frac_not_reached:

        scores_test = []
        if r_0 > 1:
            r_0 = 1

        print(f'Fraction is currently at {r_0}')
        
        tmp_data = data_funcs.get_train_vali_test_data(args, data, bank_indices, r_0=r_0)
        train_data = tmp_data.get('train_data')
        vali_data = tmp_data.get('vali_data')
        
        for i in range(x_0):

                models[i].train(train_data)
                pred = models[i].model.predict(xgb.DMatrix(vali_data['X']))
                scores_test.append(f1_eval(pred, vali_data))
                #scores_test.append(f1_score(vali_data['y'], (pred >= 0.5).astype(int), average='binary'))

        get_top = 1 if r_0 == 1 else round(x_0/eta)
        params_to_keep = sorted(range(len(scores_test)), key=lambda i: scores_test[i], reverse=True)[:get_top]
        models = [models[i] for i in params_to_keep]

        if r_0 >= 1:
            frac_not_reached = False

        x_0 = round(x_0/eta)
        r_0 *= eta
    
    return models[0]



""""

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


