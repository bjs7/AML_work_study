import numpy as np
import data.feature_engi as fe
#import pandas as pd


# --------------------------------------------------------------------------------------------------------------------------
# Indices functions --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

def get_bank_indices(df, bank):

    listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == bank)[0])]
    listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == bank)[0])]
    
    listwith_frombank = list(df.index[listwith_frombank])
    listwith_tobank = list(df.index[listwith_tobank])
    indices_to_get = list(set(listwith_frombank + listwith_tobank))
    bank_indices = sorted(indices_to_get)

    return bank_indices

def get_indices_bdt(data, bank = None):

    train_data_indices = data['regular_data']['train_data']['x'][['From Bank', 'To Bank']]
    vali_data_indices = data['regular_data']['vali_data']['x'][['From Bank', 'To Bank']]
    test_data_indices = data['regular_data']['test_data']['x'][['From Bank', 'To Bank']]
    
    train_data_indices = get_bank_indices(train_data_indices, bank) if isinstance(bank, int) else train_data_indices.index.tolist()
    vali_data_indices = get_bank_indices(vali_data_indices, bank) if isinstance(bank, int) else vali_data_indices.index.tolist()
    test_data_indices = get_bank_indices(test_data_indices, bank) if isinstance(bank, int) else test_data_indices.index.tolist()

    return {'train_indices': train_data_indices, 'vali_indices': vali_data_indices, 'test_indices': test_data_indices}




# --------------------------------------------------------------------------------------------------------------------------
# Get model type data functions --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# Graph data function -------------------------------------

def get_graph_data(args, df, bank_indices = None):

    if args.scenario == 'individual_banks':
        train_data, vali_data, test_data = fe.update_data(df['test_data']['df'], bank_indices, args)
    else:
        train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']
        
    return train_data, vali_data, test_data



# Booster data function -------------------------------------

def get_booster_data(args, df, bank_indices = None):

    if args.scenario == 'individual_banks':
        train_data, vali_data, test_data = fe.update_regular_data(df, bank_indices, args)
    else:
        df['vali_data']['x'] = df['vali_data']['x'].reset_index(drop=True)
        df['test_data']['x'] = df['test_data']['x'].reset_index(drop=True)

        train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']

    return train_data, vali_data, test_data
















"""


def get_graph_data(data, args, bank_indices = None):

    model_type = tu.model_types.get(args.model)
    df = data[tu.data_types.get(model_type)]
    #df = raw_data[tu.data_types.get(model_type)]

    if model_type == 'graph':
        if args.scenario == 'individual_banks':
            train_data, vali_data, test_data = pdt.update_data(df['test_data']['df'], bank_indices, args)
        else:
            train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']
        
        return train_data, vali_data, test_data
    
    elif model_type == 'booster':
        if args.scenario == 'individual_banks':
            train_data, vali_data, test_data = pdt.update_regular_data(df, bank_indices, args)
        else:
            df['vali_data']['x'] = df['vali_data']['x'].reset_index(drop=True)
            df['test_data']['x'] = df['test_data']['x'].reset_index(drop=True)

            train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']

        #train_data, vali_data, test_data = pdt.update_regular_data(data, bank_indices, args)

        return train_data, vali_data, test_data






def get_indices_bdt(data, args, bank = None):

    model_type = tu.model_types.get(args.model)

    train_data_indices = data['regular_data']['train_data']['x'][['From Bank', 'To Bank']]
    vali_data_indices = data['regular_data']['vali_data']['x'][['From Bank', 'To Bank']]
    test_data_indices = data['regular_data']['test_data']['x'][['From Bank', 'To Bank']]
    
    train_data_indices = get_bank_indices(train_data_indices, bank, model_type) if isinstance(bank, int) else train_data_indices.index.tolist()
    vali_data_indices = get_bank_indices(vali_data_indices, bank, model_type) if isinstance(bank, int) else vali_data_indices.index.tolist()
    test_data_indices = get_bank_indices(test_data_indices, bank, model_type) if isinstance(bank, int) else test_data_indices.index.tolist()

    return {'train_indices': train_data_indices, 'vali_indices': vali_data_indices, 'test_indices': test_data_indices}
    #return {'train_data_indices': train_data_indices, 'vali_data_indices': vali_data_indices, 'test_data_indices': test_data_indices}




"""

