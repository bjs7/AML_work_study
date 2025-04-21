import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from snapml import GraphFeaturePreprocessor
from gp_params import params
from torch_geometric.data import Data


def process_regular_data(data, bank_indices, args, scaler_encoders = None):

    df = copy.copy(data)

    scaler = scaler_encoders.get('scaler') if scaler_encoders else None
    encoder_pay = scaler_encoders.get('encoder_pay') if scaler_encoders else None
    encoder_cur = scaler_encoders.get('encoder_cur') if scaler_encoders else None

    data_features = ['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 'Sent Currency', 'Payment Format']
    X = df['x'].iloc[bank_indices,:].loc[:,data_features]
    X = X.reset_index(drop=True)
    y = df['y'][bank_indices].copy()

    # graph_feature_preprocessing using snapML

    gp = GraphFeaturePreprocessor()
    gp.set_params(params)

    X_gf = gp.fit_transform(X[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent']].astype("float64"))
    #gp.fit_transform(X[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent']].astype("float64"))
    X_gf = X_gf[:,5:]
    
    # remove EdgeID as it is no longer needed
    X = X.drop('EdgeID', axis='columns')

    # Count the amount of times an account sends or receives
    for ID in ['from_id', 'to_id']:
        X.iloc[:, np.where(X.columns == ID)[0][0]] = X.iloc[:, np.where(X.columns == ID)[0][0]].map(X.iloc[:, np.where(X.columns == ID)[0][0]].value_counts())

    period = 86400
    timestamps = X.loc[:, 'Timestamp']
    sin_component = np.sin(2 * np.pi * timestamps / period)
    cos_component = np.cos(2 * np.pi * timestamps / period)

    if not scaler:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(X.loc[:,['Timestamp', 'Amount Sent']])
    else:
        scaled_values = scaler.transform(X.loc[:,['Timestamp', 'Amount Sent']])
    X['Timestamp'] = X['Timestamp'].astype(float)
    X.loc[:,['Timestamp']] = scaled_values[:,0]
    X.loc[:, ['Amount Sent']] = scaled_values[:,1]

    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_pay.fit_transform(np.array([X.loc[:, 'Payment Format']]).T)
    else:
        encoded = encoder_pay.transform(np.array([X.loc[:, 'Payment Format']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_pay.get_feature_names_out(["Payment Format"]))
    X = pd.concat([X.drop(columns = ['Payment Format']), encoded_df], axis=1)

    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_cur.fit_transform(np.array([X.loc[:, 'Sent Currency']]).T)
    else:
        encoded = encoder_cur.transform(np.array([X.loc[:, 'Sent Currency']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_cur.get_feature_names_out(["Sent Currency"]))
    X = pd.concat([X.drop(columns=['Sent Currency']), encoded_df], axis=1)

    sin_cos = pd.concat([pd.DataFrame({'sin_component': sin_component.values}), pd.DataFrame({'cos_component': cos_component.values})], axis=1)
    X = pd.concat([X, sin_cos], axis = 1)

    # finally concat X and the graph features
    X_gf = pd.DataFrame(X_gf, columns=[f'graph_feature_{i + 1}' for i in range(X_gf.shape[1])])
    X = pd.concat([X, X_gf], axis=1)

    return {'X': X, 'y': y, 'bank_indices': np.array(bank_indices), 'scaler_encoders': {'scaler': scaler, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}}


# function to first 'update' data



def get_updated_bank_indices(bank_indices):

    #bank_indices = copy.copy(bank_indices)
    train_indices = bank_indices.get('train_data_indices', None)
    vali_indices = bank_indices.get('vali_data_indices', None)
    test_indices = bank_indices.get('test_data_indices', None)
    
    # update the indices so they 'match' the bank
    updated_train_indices = np.arange(len(train_indices))
    updated_vali_indices = np.arange(len(vali_indices)) + len(train_indices)
    updated_test_indices = np.arange(len(test_indices)) + len(train_indices) + len(vali_indices)
    
    # indices for the whole bank
    bank_indices = train_indices + vali_indices + test_indices

    return updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices


def update_nr_nodes(df):
    max_n_id = np.array(df.edge_index.max() + 1)
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    x = torch.tensor(df_nodes.loc[:, ['Feature']].to_numpy()).float()
    df.x = x


def update_data(data, bank_indices, args, mode = 'train', scaler_encoders = None, train_plus_vali = False):

    df = copy.copy(data)
    #df = copy.copy(df1['train_data'])
    #df = copy.copy(df1['test_data']['df'])

    # get indices for the bank
    updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices = get_updated_bank_indices(bank_indices)

    df.edge_index = df.edge_index[:,bank_indices].clone()
    df.edge_attr = df.edge_attr[bank_indices,:].clone()
    df.y = df.y[bank_indices].clone()
    df.timestamps = df.timestamps[bank_indices].clone()

    def get_dict_val(name, collection):
        return collection.setdefault(name, len(collection))

    reset_indices = dict()
    for j in range(df.edge_index.shape[1]):
        df.edge_index[:,j] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[:,j].tolist()])

    el_vali = np.concatenate([updated_train_indices, updated_vali_indices])
    train_edges, train_attr, train_y, train_ts = df.edge_index[:,updated_train_indices], df.edge_attr[updated_train_indices,:], df.y[updated_train_indices], df.timestamps[updated_train_indices]
    vali_edges, vali_attr, vali_y, vali_ts = df.edge_index[:,el_vali], df.edge_attr[el_vali,:], df.y[el_vali], df.timestamps[el_vali]

    train_data = Data(x = df.x, edge_index=train_edges, edge_attr=train_attr, y=train_y, timestamps=train_ts)
    vali_data = Data(x = df.x, edge_index=vali_edges, edge_attr=vali_attr, y=vali_y, timestamps=vali_ts)
    test_data = df
    
    update_nr_nodes(train_data)
    update_nr_nodes(vali_data)
    update_nr_nodes(test_data)

    #train_data.x, vali_data.x, test_data.x = z_norm(train_data.x), z_norm(vali_data.x), z_norm(test_data.x)

    #{'df': df, 'pred_indices': torch.tensor(pred_indices),'scaler_encoders': {'scaler': scaler, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}}
    return {'df': train_data, 'pred_indices': torch.tensor(updated_train_indices)}, {'df': vali_data, 'pred_indices': torch.tensor(updated_vali_indices)}, {'df': test_data, 'pred_indices': torch.tensor(updated_test_indices)}


# function to process it, standardize etc.


def feature_engi_graph_data(data, scaler_encoders = None):

    data = copy.deepcopy(data)
    #data = copy.deepcopy(train_data)
    df = data['df']

    scaler = scaler_encoders.get('scaler') if scaler_encoders else None
    encoder_pay = scaler_encoders.get('encoder_pay') if scaler_encoders else None
    encoder_cur = scaler_encoders.get('encoder_cur') if scaler_encoders else None

    # time periods ---------------------------------------------------------------------------------------------------------

    timestamps = df.edge_attr[:, 0]

    hour_period = 60*60
    sin_component_hour = torch.sin(2 * np.pi * timestamps / hour_period).unsqueeze(1)
    cos_component_hour = torch.cos(2 * np.pi * timestamps / hour_period).unsqueeze(1)

    day_period = 60*60*24
    sin_component_day = torch.sin(2 * np.pi * timestamps / day_period).unsqueeze(1)
    cos_component_day = torch.cos(2 * np.pi * timestamps / day_period).unsqueeze(1)

    week_period = 60*60*24*7
    sin_component_week = torch.sin(2 * np.pi * timestamps / week_period).unsqueeze(1)
    cos_component_week = torch.cos(2 * np.pi * timestamps / week_period).unsqueeze(1)

    # FOR NOW TIME IS NOT KEPT!
    # standardization ------------------------------------------------------------------------------------------------------
    if not scaler:
        scaler = StandardScaler()
        #scaled_values = scaler.fit_transform(df.edge_attr[:,0:2])
        scaled_values = scaler.fit_transform(torch.reshape(df.edge_attr[:,1], (df.edge_attr[:,1].shape[0],1)))
    else:
        #scaled_values = scaler.transform(df.edge_attr[:,0:2])
        scaled_values = scaler.transform(torch.reshape(df.edge_attr[:,1], (df.edge_attr[:,1].shape[0],1)))
    #df.edge_attr[:,0:2] = torch.tensor(scaled_values)
    df.edge_attr[:,1] = torch.reshape(torch.tensor(scaled_values), (-1,))

    df.x = z_norm(df.x)
    
    # Encoding -------------------------------------------------------------------------------------------------------------
    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_payment = encoder_pay.fit_transform(np.array([df.edge_attr[:,2]]).T)
    else:
        encoded_payment = encoder_pay.transform(np.array([df.edge_attr[:,2]]).T)

    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_currency = encoder_cur.fit_transform(np.array([df.edge_attr[:,3]]).T)
    else:
        encoded_currency = encoder_cur.transform(np.array([df.edge_attr[:,3]]).T)

    # Pack -----------------------------------------------------------------------------------------------------------------
    df.edge_attr = torch.cat([torch.tensor(np.arange(df.edge_attr.shape[0])).unsqueeze(1), df.edge_attr[:, [0, 1]],
                            torch.tensor(encoded_payment), torch.tensor(encoded_currency),
                            sin_component_day, cos_component_day, sin_component_hour, cos_component_hour, sin_component_week, cos_component_week], axis=1).float()
    
    data['scaler_encoders'] = {'scaler': scaler, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}

    return data


#feature_engi_graph_data(train_data)

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std








def process_graph_data(data, bank_indices, args, mode = 'train', scaler_encoders = None, train_plus_vali = False):


    # this approach nodes are only included in a split if they have transactions there
    df = copy.copy(data)

    scaler = scaler_encoders.get('scaler') if scaler_encoders else None
    encoder_pay = scaler_encoders.get('encoder_pay') if scaler_encoders else None
    encoder_cur = scaler_encoders.get('encoder_cur') if scaler_encoders else None

    if mode == 'validation':
        train_indices = bank_indices.get('train_data_indices', None)
        vali_indices = bank_indices.get('vali_data_indices', None)
        bank_indices = train_indices + vali_indices
        pred_indices = np.arange(len(vali_indices)) + len(train_indices)
    elif mode == 'test':
        train_indices = bank_indices.get('train_data_indices', None)
        vali_indices = bank_indices.get('vali_data_indices', None)
        test_indices = bank_indices.get('test_data_indices', None)
        bank_indices = train_indices + vali_indices + test_indices
        pred_indices = np.arange(len(test_indices)) + len(train_indices) + len(vali_indices)
    else:
        bank_indices = bank_indices.get('train_data_indices', None) + bank_indices.get('vali_data_indices', None) if train_plus_vali else bank_indices.get('train_data_indices', None)
        pred_indices = np.arange(len(bank_indices))


    if args.scenario == 'individual_banks': #or (len(bank_indices) < len(data['y']))

        def get_dict_val(name, collection):
            return collection.setdefault(name, len(collection))
        
        df.edge_index = df.edge_index[:,bank_indices].clone()
        df.edge_attr = df.edge_attr[bank_indices,:].clone()
        df.y = df.y[bank_indices].clone()
        df.timestamps = df.timestamps[bank_indices].clone()

        
        reset_indices = dict()
        #df.edge_index[0,:] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[0,:].tolist()])
        #df.edge_index[1,:] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[1, :].tolist()])

        for j in range(df.edge_index.shape[1]):
            df.edge_index[:,j] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[:,j].tolist()])


        max_n_id = np.array(df.edge_index.max() + 1)
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})

        x = torch.tensor(df_nodes.loc[:, ['Feature']].to_numpy()).float()
        df.x = x



    period = 86400
    timestamps = df.edge_attr[:, 0]
    sin_component = torch.sin(2 * np.pi * timestamps / period).unsqueeze(1)
    cos_component = torch.cos(2 * np.pi * timestamps / period).unsqueeze(1)

    df.edge_attr[:, 0:2] = z_norm(df.edge_attr[:, 0:2])
    
    # old code, doesn't seem to work for pytorch 2.5.0, instead 2.6.0 is required. Might be able to run if environment from
    # AML data generating paper is used.

    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_payment = encoder_pay.fit_transform(np.array([df.edge_attr[:,2]]).T) # old
        #encoded_payment = encoder.fit_transform(np.array(df.edge_attr[:,2]).reshape(-1,1)) # new
    else:
        encoded_payment = encoder_pay.transform(np.array([df.edge_attr[:,2]]).T) # old

    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_currency = encoder_cur.fit_transform(np.array([df.edge_attr[:,3]]).T) # old
        #encoded_currency = encoder.fit_transform(np.array(df.edge_attr[:,3]).reshape(-1,1)) # new
    else:
        encoded_currency = encoder_cur.transform(np.array([df.edge_attr[:,3]]).T)

    df.edge_attr = torch.cat([torch.tensor(np.arange(df.edge_attr.shape[0])).unsqueeze(1), df.edge_attr[:, [0, 1]],
                              torch.tensor(encoded_payment), torch.tensor(encoded_currency),
                              sin_component, cos_component], axis=1).float()

    return {'df': df, 'pred_indices': torch.tensor(pred_indices),'scaler_encoders': {'scaler': scaler, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}}


def process_graph_data_for_eval(data, bank_indices, args, scaler = None, encoder_pay = None, encoder_cur = None):

    df = copy.copy(data)

    #df = copy.copy(data[tu.data_types.get(model_type)]['train_data'])
    #df = copy.copy(data[tu.data_types.get(model_type)]['vali_data'])
    #df = copy.copy(vali_data)

    #


    #new_train_indices = np.arange(len(test_indices)) + (len(train_indices) + len(new_vali_indices))

    if args.scenario == 'individual_banks': #or (len(bank_indices) < len(data['y']))

        df.edge_index = df.edge_index[:,bank_indices].clone()
        df.edge_attr = df.edge_attr[bank_indices,:].clone()
        df.y = df.y[bank_indices].clone()
        df.timestamps = df.timestamps[bank_indices].clone()

        bank_indices = np.arange(len(bank_indices))

        def get_dict_val(name, collection):
            return collection.setdefault(name, len(collection))
        
        reset_indices = dict()
        df.edge_index[0,:] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[0,:].tolist()])
        df.edge_index[1,:] = torch.tensor([get_dict_val(ele, reset_indices) for ele in df.edge_index[1, :].tolist()])

        max_n_id = np.array(df.edge_index.max() + 1)
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})

        x = torch.tensor(df_nodes.loc[:, ['Feature']].to_numpy()).float()
        df.x = x

    period = 86400
    timestamps = df.edge_attr[:, 0]
    sin_component = torch.sin(2 * np.pi * timestamps / period).unsqueeze(1)
    cos_component = torch.cos(2 * np.pi * timestamps / period).unsqueeze(1)

    df.edge_attr[:, 0:2] = z_norm(df.edge_attr[:, 0:2])
    
    # old code, doesn't seem to work for pytorch 2.5.0, instead 2.6.0 is required. Might be able to run if environment from
    # AML data generating paper is used.

    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_payment = encoder_pay.fit_transform(np.array([df.edge_attr[:,2]]).T) # old
        #encoded_payment = encoder.fit_transform(np.array(df.edge_attr[:,2]).reshape(-1,1)) # new
    else:
        encoded_payment = encoder_pay.transform(np.array([df.edge_attr[:,2]]).T) # old


    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_currency = encoder_cur.fit_transform(np.array([df.edge_attr[:,3]]).T) # old
        #encoded_currency = encoder.fit_transform(np.array(df.edge_attr[:,3]).reshape(-1,1)) # new
    else:
        encoded_currency = encoder_cur.transform(np.array([df.edge_attr[:,3]]).T)

    df.edge_attr = torch.cat([torch.tensor(np.arange(df.edge_attr.shape[0])).unsqueeze(1), df.edge_attr[:, [0, 1]],
                              torch.tensor(encoded_payment), torch.tensor(encoded_currency),
                              sin_component, cos_component], axis=1).float()

    return {'df': df, 'pred_indices': torch.tensor(pred_indices), 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}




def banks_indices(df, unique_banks, include_tobank = True):
    banks_data_indices = {}
    for bank in unique_banks:
        listwith_tobank = []
        listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == bank)[0])]
        if include_tobank:
            listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == bank)[0])]
        indices_to_get = list(set(listwith_frombank + listwith_tobank))
        banks_data_indices[bank] = sorted(indices_to_get)
    return banks_data_indices



