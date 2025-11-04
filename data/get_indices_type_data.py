import numpy as np
import data.feature_engi as fe
import data.data_utils as du
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
    

    if args.reverse_mp:
        train_data['df'] = du.create_hetero_obj(train_data['df'].x,  train_data['df'].y,  train_data['df'].edge_index,  train_data['df'].edge_attr, train_data['df'].timestamps, args)
        vali_data['df'] = du.create_hetero_obj(vali_data['df'].x,  vali_data['df'].y,  vali_data['df'].edge_index,  vali_data['df'].edge_attr, vali_data['df'].timestamps, args)
        test_data['df'] = du.create_hetero_obj(test_data['df'].x,  test_data['df'].y,  test_data['df'].edge_index,  test_data['df'].edge_attr, test_data['df'].timestamps, args)

    #args.ports = True
    if args.ports:
        train_data['df'].add_ports()
        vali_data['df'].add_ports()
        test_data['df'].add_ports()
    if args.tds:
        train_data['df'].add_time_deltas()
        vali_data['df'].add_time_deltas()
        test_data['df'].add_time_deltas()
        
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

    test123 = du.create_hetero_obj(train_data['df'].x,  train_data['df'].y,  train_data['df'].edge_index,  train_data['df'].edge_attr, train_data['df'].timestamps, args)
    test123.add_ports()

    test123
    
    train_data['df'].edge_attr

    test123['node']
    test123['node', 'to', 'node'].edge_index
    test123['node', 'rev_to', 'node'].edge_index

    test123['node', 'to', 'node'].edge_attr
    test123['node', 'rev_to', 'node'].edge_attr

    test123['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]]
    test123['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

    test123['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = test123['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

    test123['node', 'rev_to', 'node'].edge_attr[:, [5]] = test123['node', 'rev_to', 'node'].edge_attr[:, [4]]

    test123['node', 'rev_to', 'node'].edge_attr[:, [4, 5]] = test123['node', 'rev_to', 'node'].edge_attr[:, [5, 4]]

"""