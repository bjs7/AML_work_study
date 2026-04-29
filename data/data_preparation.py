import numpy as np
import data.feature_engineering as fe
import data.data_utils as du
#import pandas as pd


# --------------------------------------------------------------------------------------------------------------------------
# Indices functions --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

def get_bank_indices(df, bank):
    mask = (df.loc[:, 'From Bank'] == bank) | (df.loc[:, 'To Bank'] == bank)
    return sorted(df[mask].index.tolist())

def get_indices_bdt(data, bank = None):

    df = {}

    for split in ['train', 'vali', 'test']:
        split_indices = data['regular_data'][f'{split}_data']['x'][['From Bank', 'To Bank']]
        df[f'{split}_indices'] = get_bank_indices(split_indices, bank) if isinstance(bank, int) else split_indices.index.tolist()

    return df


# --------------------------------------------------------------------------------------------------------------------------
# Get model type data functions --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# Graph data function -------------------------------------

def get_graph_data(parsers, df, bank_indices = None):

    if parsers['data_parser'].scenario == 'individual_banks':
        train_data, vali_data, test_data = fe.update_data(df['test_data']['df'], bank_indices)
    else:
        train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']
    

    if parsers['gnn_parser'].reverse_mp:
        for data in (train_data, vali_data, test_data):
            data['df'] = du.create_hetero_obj(
                data['df'].x,  data['df'].y,  data['df'].edge_index,  
                data['df'].edge_attr, data['df'].timestamps
                )

    #args.ports = True
    if parsers['gnn_parser'].ports:
        for data in (train_data, vali_data, test_data):
            data['df'].add_ports()

    if parsers['gnn_parser'].tds:
        for data in (train_data, vali_data, test_data):
            data['df'].add_time_deltas()
        
    return train_data, vali_data, test_data



# Booster data function -------------------------------------

def get_booster_data(args, df, bank_indices = None):

    if args.scenario == 'individual_banks':
        return fe.update_regular_data(df, bank_indices)
    else:
        df['vali_data']['x'] = df['vali_data']['x'].reset_index(drop=True)
        df['test_data']['x'] = df['test_data']['x'].reset_index(drop=True)
        return df['train_data'], df['vali_data'], df['test_data']
