import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from snapml import GraphFeaturePreprocessor
#from gp_params import params
#from training.tuning_utils import gfpparams
import training.hyperparams as tu
from torch_geometric.data import Data
import data.data_utils as du


# Function for feature engineering, GNN and Booster --------------------------------------------------------------------------------------------

def general_feature_engineering(model_type, train_data, vali_data):

    if model_type == 'graph':
        train_data = feature_engi_graph_data(train_data)
        vali_data = feature_engi_graph_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))
    elif model_type == 'booster':
        train_data = feature_engi_regular_data(train_data)
        vali_data = feature_engi_regular_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))
    
    return train_data, vali_data


def update_regular_data(df, bank_indices, args):
    
    #df = copy.deepcopy(data)

    train_data, vali_data, test_data = [        
        {
            'x': df[data_key]['x'].loc[bank_indices[indices_key], :].reset_index(drop=True),
            'y': df[data_key]['x'].loc[bank_indices[indices_key], 'Is Laundering'].to_numpy()
        } 
        for data_key, indices_key in zip(list(df.keys()), list(bank_indices.keys()))
        ]
    
    return train_data, vali_data, test_data


# For now no update to the from_id and to_id variables, as these are just id's so the value should be irrelevant
def feature_engi_regular_data(df, scaler_encoders = None):

    #df = copy.copy(data)
    #df = train_data

    scaler = scaler_encoders.get('scaler') if scaler_encoders else None
    encoder_pay = scaler_encoders.get('encoder_pay') if scaler_encoders else None
    encoder_cur = scaler_encoders.get('encoder_cur') if scaler_encoders else None
    gp = scaler_encoders.get('gfp') if scaler_encoders else None

    data_features = ['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 'Sent Currency', 'Payment Format']
    x = df['x'].loc[:,data_features]
    y = df['y']

    # GFP -----------------------------------------------------------------------------------------------------------------
    # graph_feature_preprocessing using snapML

    # switch the position of graph feature process and the other features? To include those features in graph features processing?

    if not gp:
        gp = GraphFeaturePreprocessor()
        gp.set_params(tu.gfpparams)
        x_gf = gp.fit_transform(x[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent']].astype("float64"))
    else:
        x_gf = gp.transform(x[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent']].astype("float64"))
    x_gf = x_gf[:,5:]
    
    # remove EdgeID as it is no longer needed
    x = x.drop('EdgeID', axis='columns')

    # Count the amount of times an account sends or receives
    for ID in ['from_id', 'to_id']:
        x.iloc[:, np.where(x.columns == ID)[0][0]] = x.iloc[:, np.where(x.columns == ID)[0][0]].map(x.iloc[:, np.where(x.columns == ID)[0][0]].value_counts())


    # time periods ---------------------------------------------------------------------------------------------------------

    timestamps = x.loc[:, 'Timestamp']

    hour_period = 60*60
    sin_component_hour = np.sin(2 * np.pi * timestamps / hour_period)
    cos_component_hour = np.cos(2 * np.pi * timestamps / hour_period)

    day_period = 60*60*24
    sin_component_day = np.sin(2 * np.pi * timestamps / day_period)
    cos_component_day = np.cos(2 * np.pi * timestamps / day_period)

    week_period = 60*60*24*7
    sin_component_week = np.sin(2 * np.pi * timestamps / week_period)
    cos_component_week = np.cos(2 * np.pi * timestamps / week_period)


    # standardization ------------------------------------------------------------------------------------------------------

    if not scaler:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(x.loc[:,['Amount Sent']])
    else:
        scaled_values = scaler.transform(x.loc[:,['Amount Sent']])
    #x['Timestamp'] = x['Timestamp'].astype(float)
    #x.loc[:,['Timestamp']] = scaled_values[:,0]
    x.loc[:, ['Amount Sent']] = scaled_values[:,0]

    # Encoding -------------------------------------------------------------------------------------------------------------

    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_pay.fit_transform(np.array([x.loc[:, 'Payment Format']]).T)
    else:
        encoded = encoder_pay.transform(np.array([x.loc[:, 'Payment Format']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_pay.get_feature_names_out(["Payment Format"]))
    x = pd.concat([x.drop(columns = ['Payment Format']), encoded_df], axis=1)

    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_cur.fit_transform(np.array([x.loc[:, 'Sent Currency']]).T)
    else:
        encoded = encoder_cur.transform(np.array([x.loc[:, 'Sent Currency']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_cur.get_feature_names_out(["Sent Currency"]))
    x = pd.concat([x.drop(columns=['Sent Currency']), encoded_df], axis=1)


    # Pack -----------------------------------------------------------------------------------------------------------------

    sin_cos = pd.concat([
                         pd.DataFrame({'sin_component_hour': sin_component_hour.values}), 
                         pd.DataFrame({'cos_component_hour': cos_component_hour.values}),
                         pd.DataFrame({'sin_component_day': sin_component_day.values}),
                         pd.DataFrame({'cos_component_day': cos_component_day.values}),
                         pd.DataFrame({'sin_component_week': sin_component_week.values}),
                         pd.DataFrame({'cos_component_week': cos_component_week.values}),
                         ], axis=1)
    x = pd.concat([x, sin_cos], axis = 1)

    # finally concat X and the graph features
    x_gf = pd.DataFrame(x_gf, columns=[f'graph_feature_{i + 1}' for i in range(x_gf.shape[1])])
    x = pd.concat([x, x_gf], axis=1)

    return {'x': x, 'y': y, 'scaler_encoders': {'scaler': scaler, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur, 'gfp': gp}} #'bank_indices': np.array(bank_indices)


# function to first 'update' data

def get_updated_bank_indices(bank_indices):

    #bank_indices = copy.copy(bank_indices)
    train_indices = bank_indices.get('train_indices', None)
    vali_indices = bank_indices.get('vali_indices', None)
    test_indices = bank_indices.get('test_indices', None)
    
    # update the indices so they 'match' the bank
    updated_train_indices = np.arange(len(train_indices))
    updated_vali_indices = np.arange(len(vali_indices)) + len(train_indices)
    updated_test_indices = np.arange(len(test_indices)) + len(train_indices) + len(vali_indices)
    
    # indices for the whole bank
    bank_indices = train_indices + vali_indices + test_indices

    return updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices


def update_data(data, bank_indices, args, mode = 'train', scaler_encoders = None, train_plus_vali = False):

    df = copy.copy(data)

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
    
    du.update_nr_nodes(train_data)
    du.update_nr_nodes(vali_data)
    du.update_nr_nodes(test_data)

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

    df.x = du.z_norm(df.x)
    
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


