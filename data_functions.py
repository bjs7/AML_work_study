import numpy as np
import process_data_type as pdt
import trainer_utils as tu
import pandas as pd
import torch
import copy

def general_feature_engineering(model_type, train_data, vali_data):
    train_data = pdt.feature_engi_graph_data(train_data)
    vali_data = pdt.feature_engi_graph_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))
    return train_data, vali_data

def get_graph_data(data, args, bank_indices = None):

    model_type = tu.model_types.get(args.model)
    df = data[tu.data_types.get(model_type)]
    #data_processor = tu.data_functions.get(model_type)

    if model_type == 'graph':
        if args.scenario == 'individual_banks':
            train_data, vali_data, test_data = pdt.update_data(df['test_data']['df'], bank_indices, args)
        else:
            train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']        
        
        return train_data, vali_data, test_data



# indices functions ---------

def get_bank_indices(df, bank, model_type = None):

    listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == bank)[0])]
    listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == bank)[0])]
    if model_type == 'graph':
        listwith_frombank = list(df.index[listwith_frombank])
        listwith_tobank = list(df.index[listwith_tobank])
    indices_to_get = list(set(listwith_frombank + listwith_tobank))
    bank_indices = sorted(indices_to_get)

    return bank_indices

def get_indices_bdt(data, args, bank = None):

    model_type = tu.model_types.get(args.model)

    train_data_indices = data['regular_data']['train_data']['x'][['From Bank', 'To Bank']]
    vali_data_indices = data['regular_data']['vali_data']['x'][['From Bank', 'To Bank']]
    test_data_indices = data['regular_data']['test_data']['x'][['From Bank', 'To Bank']]
    
    train_data_indices = get_bank_indices(train_data_indices, bank, model_type) if isinstance(bank, int) else train_data_indices.index.tolist()
    vali_data_indices = get_bank_indices(vali_data_indices, bank, model_type) if isinstance(bank, int) else vali_data_indices.index.tolist()
    test_data_indices = get_bank_indices(test_data_indices, bank, model_type) if isinstance(bank, int) else test_data_indices.index.tolist()

    return {'train_data_indices': train_data_indices, 'vali_data_indices': vali_data_indices, 'test_data_indices': test_data_indices}

