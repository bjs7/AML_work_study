

import process_data_type as pdt
import trainer as tra

from pathlib import Path
import json
import save_load_models as slm

import random
import tuning as tnt


from torch_geometric.loader import LinkNeighborLoader
import trainer_utils as tu
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
import numpy as np
import xgboost as xgb
import copy


import os
import joblib
from datetime import date
import process_data_type as pdt
import trainer_utils as tu
import torch
from IPython import display
import math


import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_utils as du
import tuning as tune
import scartch_file as sf
import evaluation as eval


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    #parser.add_argument('--scenario', default='full info', type=str, help='Select the scenario to study')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    #parser.add_argument('--data_split', default=[0.60, 0.20], type=utils.parse_data_split)
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')

    parser.add_argument('--graph_tuning_x_0', default = 25, type=int, help='Amount of models to train on for tuning GNN')
    parser.add_argument('--seed', default=0, type=int, help="Select the random seed for reproducability")
    #parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    
    return parser



def main():

    ###########################
    ## NEED TO FIX SEEDS!!!! ##
    ###########################
    # FIT / TRANSFORM AF Graph feature process
    # switch the position of graph feature process and the other features? To include those features in graph features processing?

    # ALSO ENSURE THE SAVE MODELS HAVE THE RIGHT FORMAT!

    # Set hyper parameters, loop values etc. to the real ones

    #check batch size again?

    # set logging
    utils.logger_setup()
    
    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    # set seed
    utils.set_seed(args.seed)

    # load data
    logging.info("load_data")
    df = pd.read_csv("/home/nam_07/AML_work_study/formatted_transactions.csv")
    #df = pd.read_csv("/data/leuven/362/vsc36278/AML_work_study/formatted_transactions.csv")
    data = dp.get_data(df, split_perc = configs.split_perc)

    if args.scenario == 'individual_banks':
        Banks = list((df.loc[:, 'From Bank'])) + list((df.loc[:, 'To Bank']))
        unique_banks = list(set(Banks))
        args.banks = unique_banks

    if args.banks:
        args.scenario = 'individual_banks'

    args.scenario = 'individual_banks'
    bank = 2
    #args.model = 'xgboost'

    #args.scenario = 'full_info'
    #bank = None

    # should only do this once, in order to save time etc.
    #model_type = tu.model_types.get(args.model)
    bank_indices = du.get_indices_bdt(data, args, bank = bank)

    # tune the model
    args.graph_tuning_x_0 = 25
    args.graph_tuning_x_0 = 2
    tuned_params = tune.tuning(args, data, bank_indices)

    # train on train and validation, and evalute on test after hyperparameters have been found
    
    # once we ahve the best models we can train on each one of them, but with different
    # initialization

    #args.model = 'GINe'
    #args.model = 'xgboost'
    data1 = sf.get_train_vali_test_data(args, data, bank_indices, tuning=False)

    folder_name = f'bank_{bank}' if bank else args.scenario
    seeds = range(1,5)
    for ele in seeds:
        utils.set_seed(ele)
        model, f1 = eval.combined_training_eval_gnn(args, data1, tuned_params)
        file_name = f'seed_{ele}'
        tra.save_model(model, hyper_params, folder_name, file_name)




    import numpy as np
    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']
    
    
    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    x = torch.reshape(torch.tensor([1] * 257544).float(), (257544, 1))
    
    torch.manual_seed(1)
    tester = nn.Linear(1, 2)
    tester(x)
    tester(z_norm(x))







        

        





    # train model

    def simple_trainer(args, data, model_configs):

        model_type = tu.model_types.get(args.model)

        if model_type == 'graph':
            model, f1 = eval.combined_training_eval_gnn(args, data1, model_hyper_params[i])
        elif model_type == 'booster':
            # need boost functino here
            return 0
            #model, f1 = eval.combined_training_eval_gnn(args, data1, model_hyper_params[i])

        trainer_class = tu.trainer_classes.get(model_type)       
        #trainer_class = tu.trainer_classes.get(model_type)
        #trainer = trainer_class(args, data, model_configs)
        #wrapped_model = trainer.train()
        return wrapped_model
    
    wrapped_model = simple_trainer(args, train_data, model_configs)

    import evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f1s = []

    for i in range(x_0):

        wrapped_model = simple_trainer(args, train_data, model_hyper_params[i])
        f1 = evaluation.eval_func(wrapped_model, vali_data, model_hyper_params[i], args, device)
        f1s.append(f1)










    # -------------------------

    data_processor = tu.data_functions.get(model_type)

    # data for training
    train_data = data[tu.data_types.get(model_type)]['train_data']
    train_data = data_processor(train_data, bank_indices, args)


    x_0 = 25
    model_hyper_params = [utils.hyper_sampler(args) for i in range(x_0)]
    model_configs = model_hyper_params[0]

    m_param = model_configs.get('params')
    m_settings = model_configs.get('model_settings')

    m_param['num_neighbors'] *= m_param['gnn_layers']

    pred_indices = train_data['pred_indices']
    train_data = train_data['df']

    # need to make dynamic here?
    batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    batch_size = 10
    

    # loader
    #transform = partial(account_for_time, main_data=train_data)
    train_loader = LinkNeighborLoader(train_data, num_neighbors=m_param.get('num_neighbors'), batch_size=batch_size, shuffle=True, transform=None)
    sample_batch = next(iter(train_loader))

    batch = sample_batch
    tr_inds = bank_indices['train_data_indices']
    tr_loader = train_loader

    inds = torch.tensor(tr_inds)

    # -----------------
    inds = pred_indices
    batch_edge_inds = inds[batch.input_id]
    batch_edge_ids = tr_loader.data.edge_attr[batch_edge_inds, 0]

    mask = torch.isin(batch.edge_attr[:, 0], batch_edge_ids)
    sum(mask)

    mask = torch.isin(batch.edge_attr[:, 0], batch.input_id)
    sum(mask)








    # data for validation
    vali_data = data[tu.data_types.get(model_type)]['vali_data']
    vali_data = data_processor(vali_data, bank_indices, args, is_eval=True)

    # data for testing


    # train the model

    x_0 = 25
    model_hyper_params = [utils.hyper_sampler(args) for i in range(x_0)]
    model_configs = model_hyper_params[0]
    
    #train_indices = bank_indices.get('train_data_indices')
    trainer_class = tu.trainer_classes.get(model_type)

    import evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f1s = []

    for i in range(3):

        model_configs = model_hyper_params[i]
        trainer = trainer_class(args, train_data, model_configs)
        temp_model = trainer.train()
        #temp_model = tra.train_model(args, data, bank_indices, model_configs, bank = bank)
        f1 = evaluation.eval_func(temp_model, vali_data, model_configs, args, device)
        f1s.append(f1)
    
    params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:5]
    model_hyper_params = [models_paras for index, models_paras in enumerate(model_hyper_params) if index in params_to_keep]

    hidden_embedding_size_interval = [1e9,-1e9]
    learning_rate_interval = [1e9,-1e9]
    gnn_layers_interval = [1e9,-1e9]
    dropout_interval = [1e9,-1e9]
    w_ce2_interval = [1e9,-1e9]


    def operator(param, parameters, j):
        if j == 0:
            return param if param < parameters else parameters
        elif j == 1:
            return param if param > parameters else parameters        

    for i in range(len(model_hyper_params)):
        tmp_params = model_hyper_params[i].get('params')
        for j in range(2):
            hidden_embedding_size_interval[j] = operator(tmp_params.get('hidden_embedding_size'), hidden_embedding_size_interval[j], j)
            learning_rate_interval[j] = operator(tmp_params.get('learning rate'), learning_rate_interval[j], j)
            gnn_layers_interval[j] = operator(tmp_params.get('gnn_layers'), gnn_layers_interval[j], j)
            dropout_interval[j] = operator(tmp_params.get('dropout'), dropout_interval[j], j)
            w_ce2_interval[j] = operator(tmp_params.get('w_ce2'), w_ce2_interval[j], j)

    sampler_intervals = {'hid_em_size_interval': hidden_embedding_size_interval, 
                        'lr_interval': learning_rate_interval, 'gnn_layer_interval': gnn_layers_interval, 
                        'dropout_interval': dropout_interval, 'w_ce2_interval': w_ce2_interval}
    
    
    
    x_0 = 25
    model_hyper_params = [utils.hyper_sampler(args, sampler_intervals) for i in range(x_0)]
    model_configs = model_hyper_params[0]


    f1s = []
    for i in range(3):
        model_configs = model_hyper_params[i]
        trainer = trainer_class(args, train_data, model_configs)
        temp_model = trainer.train()
        #temp_model = tra.train_model(args, data, bank_indices, model_configs, bank = bank)
        f1 = evaluation.eval_func(temp_model, vali_data, model_configs, args, device)
        f1s.append(f1)
    
    params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:5]
    model_hyper_params = [models_paras for index, models_paras in enumerate(model_hyper_params) if index in params_to_keep]

    # set seed for each model
    # combine train and vali data, to train on

    # train on validation which includes train and validation data
    
    # evaluate the model


























    train_data = data[tu.data_types.get(model_type)]['train_data']
    train_data.edge_index


    data_processor = tu.data_functions.get(model_type)
    graph_vali_indices = train_data_indices + vali_data_indices

    #vali_data_indices
    vali_data = data_processor(vali_data, graph_vali_indices, args, temp_model.scaler, temp_model.encoder_pay, temp_model.encoder_cur)

    m_param = model_configs.get('params')
    m_settings = model_configs.get('model_settings')

    m_param['num_neighbors'] *= m_param['gnn_layers']

    batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    
    vali_data = vali_data['df']
    LinkNeighborLoader(vali_data, num_neighbors=m_param.get('num_neighbors'), edge_label_index=vali_data.edge_index[:, vali_data_indices]
                       batch_size=batch_size, shuffle=False, transform=None)



    train_data = data[tu.data_types.get(model_type)]['train_data']
    train_data = data_processor(train_data, train_indices, args)

    train_data
    vali_data

    train_data['df'].edge_index[:,52510:52519]
    vali_data['df'].edge_index[:,52510:52519]
    
    #model = 'GINe'
    #args.model = 'xgboost'
    
    # train the model and get args, configs etc. incase changes, and save direction of save files
    #args, configs, save_direc = tra.train_model(args, data, configs)


    # In case there is no data for validation for a certain bank, skip it, and pick the parameters from the biggest dataset?

    # createa a 'bank' function, might not be needed

    # currently working on bank 1
    # skip that one, because no validation. Use the bank with largest dataset parameters instead?
    # then process train data, save scalers, encoders etc., train model, and validate etc.
    

    no_train_data = []
    no_vali_data = []
    no_tr_va_data = []

    for bank in args.banks:
        
        if bank % 1000 == 0:
            print(bank)
        train_data_indices, vali_data_indices, test_data_indices = get_indices_bdt(data, bank = bank)

        if not train_data_indices and not vali_data_indices:
            no_tr_va_data.append(bank)
            continue
        elif not train_data_indices:
            no_train_data.append(bank)
            continue
        elif not vali_data_indices:
            no_vali_data.append(bank)
            continue

        
        bank = 2
        train_data_indices, vali_data_indices, test_data_indices = tnt.get_indices_bdt(data, bank = bank)

        data1 = copy.copy(data)
        
        x_0 = 25
        model_hyper_params = [utils.hyper_sampler(args) for i in range(x_0)]
        
        train_indices = train_data_indices
        model_configs = model_hyper_params[0]
        temp_model = tra.train_model(args, data, train_data_indices, model_hyper_params[0], bank = bank)
        
        model_configs

        
        torch.FloatTensor([6, 8])

        torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([6, 8]))

            



        

        bank = 2

        # 
        test_model = tnt.boost_tune_train(args, data, 2)

        # save the model
        file_name = f'bank_{bank}' if bank else 'full_info'

        #file_name = f'bank_{bank}' if bank else args.scenario
        save_direc = save_model(test_model, file_name, args, configs)
        
        
        # then save the model?
        # should it be trained on training and validation data?

        #pd.to_datetime(vali_data['x']['Timestamp'], unit='s')
    

    return 0
    
    if args.model_configs:
        return 0
    
    else:
        return 0 
        x_0 = 3


    # for all banks
    #eta = 2
    #r_0 = 0.1

    # for single bank
    #eta = 1.5
    # r_0 = 0.3

    #while eta * r_0 < 1:
        #train model
        # select relevant models
        


    # train model
    #tra.train_model(args, data, configs, bank = bank)




    if args.scenario == 'individual_banks':
        for index, bank in enumerate(args.banks):
            save_direc = tra.train_model(args, data, configs, bank = bank)
    else:
        save_direc = tra.train_model(args, data, configs)
    
    slm.save_configs(args, save_direc)




if __name__ == '__main__':
    main()







