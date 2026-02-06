import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from snapml import GraphFeaturePreprocessor
#from gp_params import params
#from training.tuning_utils import gfpparams
import training.utils as tu
#from torch_geometric.data import Data
import data.data_utils as du
from configs import configs

# Function for feature engineering, GNN and Booster --------------------------------------------------------------------------------------------

def general_feature_engineering(model_type, train_data, vali_data):

    if model_type == 'graph':
        train_data = feature_engi_graph_data(train_data)
        vali_data = feature_engi_graph_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))
    elif model_type == 'booster':
        train_data = feature_engi_regular_data(train_data)
        vali_data = feature_engi_regular_data(vali_data, scaler_encoders = train_data.get('scaler_encoders'))
    
    return train_data, vali_data


def update_regular_data(df, bank_indices):
    
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

    data = copy.deepcopy(data)
    df = data['df']

    scl_enc = {ele: scaler_encoders.get(ele) for ele in ('scaler_amt', 'encoder_currency', 'encoder_payment_format', 'gfp')}

    data_features = ['EdgeID', 'from_id', 'to_id', 'Amount_Sent_Normalized_Log', 'Amount_Received_Normalized_Log',
                     'Sent Currency', 'Received Currency', 'Payment Format', 
                     'Amount_Difference_Pct', 'is_currency_exchange']
    x = df['x'].loc[:,data_features]
    y = df['y']

    # GFP -----------------------------------------------------------------------------------------------------------------
    # graph_feature_preprocessing using snapML

    # switch the position of graph feature process and the other features? To include those features in graph features processing?
    if not scl_enc['gfp']:
        scl_enc['gfp'] = GraphFeaturePreprocessor()
        scl_enc['gfp'].set_params(tu.gfpparams)
        x_gf = scl_enc['gfp'].fit_transform(x[['EdgeID', 'from_id', 'to_id', 'Timestamp']].astype("float64"))
    else:
        x_gf = scl_enc['gfp'].transform(x[['EdgeID', 'from_id', 'to_id', 'Timestamp']].astype("float64"))
    x_gf = x_gf[:,5:]
    
    # remove EdgeID as it is no longer needed
    x = x.drop('EdgeID', axis='columns')

    # Count the amount of times an account sends or receives
    for ID in ['from_id', 'to_id']:
        x.iloc[:, np.where(x.columns == ID)[0][0]] = x.iloc[:, np.where(x.columns == ID)[0][0]].map(x.iloc[:, np.where(x.columns == ID)[0][0]].value_counts())


    # time periods ---------------------------------------------------------------------------------------------------------

    timestamps = x.loc[:, 'Timestamp']
    time_freqs = {'hour': 60*60, 'day': 60*60*24, 'week': 60*60*24*7}

    sin_comp = {key: torch.sin(2 * np.pi * timestamps / val) 
            for key, val in time_freqs.items()}
    cos_comp = {key: torch.cos(2 * np.pi * timestamps / val) 
                for key, val in time_freqs.items()}

    # standardization ------------------------------------------------------------------------------------------------------

    if not scl_enc['scaler_amt']:
        scl_enc['scaler_amt'] = StandardScaler()
        scaled_values = scl_enc['scaler_amt'].fit_transform(x.loc[:,['Amount Received']])
    else:
        scaled_values = scl_enc['scaler_amt'].transform(x.loc[:,['Amount Received']])
    x.loc[:, ['Amount Received']] = scaled_values[:,0]

    # Encoding -------------------------------------------------------------------------------------------------------------

    if not encoder_cur:
        encoder_cur = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_cur.fit_transform(np.array([x.loc[:, 'Received Currency']]).T)
    else:
        encoded = encoder_cur.transform(np.array([x.loc[:, 'Received Currency']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_cur.get_feature_names_out(["Received Currency"]))
    x = pd.concat([x.drop(columns=['Received Currency']), encoded_df], axis=1)

    if not encoder_pay:
        encoder_pay = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder_pay.fit_transform(np.array([x.loc[:, 'Payment Format']]).T)
    else:
        encoded = encoder_pay.transform(np.array([x.loc[:, 'Payment Format']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder_pay.get_feature_names_out(["Payment Format"]))
    x = pd.concat([x.drop(columns = ['Payment Format']), encoded_df], axis=1)

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
    
    bank_indices = train_indices + vali_indices + test_indices

    return updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices


def update_data(data, bank_indices):

    df = copy.deepcopy(data)

    # get indices for the bank
    updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices = get_updated_bank_indices(bank_indices)


    train_mapping = dict(enumerate(bank_indices[:len(updated_train_indices)]))
    vali_mapping = dict(enumerate(bank_indices[:len(updated_train_indices) + len(updated_vali_indices)]))
    test_mapping = dict(enumerate(bank_indices))


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

    train_data = du.GraphData(x = df.x, y=train_y, edge_index=train_edges, edge_attr=train_attr, timestamps=train_ts)
    vali_data = du.GraphData(x = df.x, y=vali_y, edge_index=vali_edges, edge_attr=vali_attr, timestamps=vali_ts)
    test_data = df

    if len(updated_train_indices) > 0:
        du.update_nr_nodes(train_data)
        train_data.num_nodes = int(train_data.x.shape[0])
        du.update_nr_nodes(vali_data)
        vali_data.num_nodes = int(vali_data.x.shape[0])
        du.update_nr_nodes(test_data)
        test_data.num_nodes = int(test_data.x.shape[0])
        
    elif len(updated_vali_indices) > 0:
        train_data = None
        du.update_nr_nodes(vali_data)
        vali_data.num_nodes = int(vali_data.x.shape[0])
        du.update_nr_nodes(test_data)
        test_data.num_nodes = int(test_data.x.shape[0])

    else:
        train_data = None
        vali_data = None
        du.update_nr_nodes(test_data)
        test_data.num_nodes = int(test_data.x.shape[0])

    return {'df': train_data, 'pred_indices': torch.tensor(updated_train_indices), 'indices_mapping': train_mapping}, {'df': vali_data, 'pred_indices': torch.tensor(updated_vali_indices), 'indices_mapping': vali_mapping}, {'df': test_data, 'pred_indices': torch.tensor(updated_test_indices), 'indices_mapping': test_mapping}


# function to process it, standardize etc.

def feature_engi_graph_data(data, args, scaler_encoders = None):

    data = copy.deepcopy(data)
    df = data['df']

    # the names in this one also have to change accordingly
    scl_enc = {ele: scaler_encoders.get(ele) for ele in ('scaler_amt_sent', 'scaler_amt_rec', 'scaler_ports_tds', 'encoder_currency', 'encoder_payment_format')} if scaler_encoders else {}

    # time periods ---------------------------------------------------------------------------------------------------------    

    timestamps = df.edge_attr[:, 0]
    time_freqs = {'hour': 60*60, 'day': 60*60*24, 'week': 60*60*24*7}

    sin_comp = {key: torch.sin(2 * np.pi * timestamps / val).unsqueeze(1) 
                for key, val in time_freqs.items()}
    cos_comp = {key: torch.cos(2 * np.pi * timestamps / val).unsqueeze(1) 
                for key, val in time_freqs.items()}
    
    # standardization ------------------------------------------------------------------------------------------------------
    
    for feat, index in zip(['scaler_amt_sent', 'scaler_amt_rec'], [1,2]):
        if not scl_enc.get(feat):
            scl_enc[feat] = StandardScaler()
            scaler_amt_values = scl_enc[feat].fit_transform(torch.reshape(df.edge_attr[:,index], (df.edge_attr[:,index].shape[0],1)))
        else:
            scaler_amt_values = scl_enc[feat].transform(torch.reshape(df.edge_attr[:,index], (df.edge_attr[:,index].shape[0],1)))
        df.edge_attr[:,index] = torch.reshape(torch.tensor(scaler_amt_values), (-1,))

    df.x = du.z_norm(df.x)

    if args.ports or args.tds:

        if not scl_enc.get('scaler_ports_tds'):
            scl_enc['scaler_ports_tds'] = StandardScaler()        
            scaler_ports_tds_values = scl_enc['scaler_ports_tds'].fit_transform(df.edge_attr[:,4:])
        else:
            scaler_ports_tds_values = scl_enc['scaler_ports_tds'].transform(df.edge_attr[:,4:])
        
        df.edge_attr[:,4:] = torch.tensor(scaler_ports_tds_values)

    # Encoding -------------------------------------------------------------------------------------------------------------

    column_encoders = {'currency': 3, 'currency': 4, 'payment_format': 5}
    encoded = {}

    for feature, col_idx in column_encoders.items():
        encoder_key = f'encoder_{feature}'

        column_data = df.edge_attr[:, col_idx].reshape(-1, 1)
        if not scl_enc.get(encoder_key): #hasattr(scl_enc[encoder_key], 'categories_')
            scl_enc[encoder_key] = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded[feature] = scl_enc[encoder_key].fit_transform(column_data)
            #raise ValueError('Encoders not predefined')
        else:
            encoded[feature] = scl_enc[encoder_key].transform(column_data)

    # Pack -----------------------------------------------------------------------------------------------------------------

    cols_to_keep = [0,1,2,6,7]
    if configs.include_time:
        cols_to_keep.append(0)

    df.edge_attr = torch.cat([df.edge_attr[:, cols_to_keep], 
                            *[torch.tensor(encoded_val) for encoded_val in encoded.values()],
                            *sin_comp.values(), *cos_comp.values()], axis=1).float()
    
    data['scaler_encoders'] = scl_enc

    return data



# Functions for data analysis ------------------------------------------------------------------------------------------------


def get_exchange_rates(df, reference_currency = 0):

    # first check that both columns hold the same type of currencies.
    currency_symmetric_diff = set(df['Sent Currency']) ^ set(df['Received Currency'])
    if currency_symmetric_diff != set():
        raise('There is a difference in currencies in the two currency columns')

    #{'US Dollar': 0, 'Bitcoin': 1, 'Euro': 2, 'Australian Dollar': 3, 'Yuan': 4, 
    # 'Rupee': 5, 'Yen': 6, 'Mexican Peso': 7, 
    # 'UK Pound': 8, 'Ruble': 9, 'Canadian Dollar': 10, 
    # 'Swiss Franc': 11, 'Brazil Real': 12, 'Saudi Riyal': 13, 'Shekel': 14}
    currencies = df['Sent Currency'].unique()
    exchange_rates = {}

    for curr in currencies:

        if curr == reference_currency:
            exchange_rates[curr] = 1.0
            continue

        mask1 = (df['Sent Currency'] == reference_currency) & (df['Received Currency'] == curr)
        if mask1.sum() > 0:
            rate1 = (df.loc[mask1, 'Amount Received'] / df.loc[mask1, 'Amount Sent']).median()
        else:
            rate1 = None

        mask2 = (df['Sent Currency'] == curr) & (df['Received Currency'] == reference_currency)
        if mask2.sum() > 0:
            rate2 = (1 / (df.loc[mask2, 'Amount Received'] / df.loc[mask2, 'Amount Sent'])).median()
        else:
            rate2 = None

        if rate1 is not None and rate2 is not None:
            exchange_rates[curr] = (rate1 + rate2) / 2  # Average both directions
        elif rate1 is not None:
            exchange_rates[curr] = rate1
        elif rate2 is not None:
            exchange_rates[curr] = rate2
        else:
            # Fallback: No direct conversion found
            # Could use 1.0 or try to infer from indirect conversions
            exchange_rates[curr] = 1.0
            print(f"Warning: No conversion data found for currency {curr}")

    return exchange_rates


def normalize_amounts(df, exchange_rates):

    df['Amount_Sent_Normalized'] = df.apply(
        lambda row: row['Amount Sent'] / exchange_rates[row['Sent Currency']], 
        axis=1
    )

    df['Amount_Received_Normalized'] = df.apply(
        lambda row: row['Amount Received'] / exchange_rates[row['Received Currency']], 
        axis=1
    )

    return df


def log_transformer(df):

    any_zeros = (df['Amount_Sent_Normalized'] <= 0).any() or (df['Amount_Received_Normalized'] <= 0).any()
    log_func = np.log1p if any_zeros else np.log

    df[['Amount_Sent_Normalized_Log', 'Amount_Received_Normalized_Log']] = df[['Amount_Sent_Normalized', 
                                                                    'Amount_Received_Normalized']].apply(lambda x: log_func(x))
    
    return df

def universal_features_restructure(df):

    # Apply feature engineering and add more features

    # Include the percentage difference between sent and received, for the non-log transformed values
    df['Amount_Difference_Pct'] = (df['Amount_Sent_Normalized'] - df['Amount_Received_Normalized']) / df['Amount_Sent_Normalized']

    # Boolean to check whether the currency sent is the same as the received one.
    df['is_currency_exchange'] = (df['Sent Currency'] != df['Received Currency']).astype(int)

    # One could include currency but different amounts (suspicious), but there are not such 
    # observations, so this is not included as it would just result in a column of 0's

    # restructure the dataframe
    df = df[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent',
        'Sent Currency', 'Amount Received', 'Received Currency',
        'Payment Format', 'Amount_Sent_Normalized', 'Amount_Received_Normalized',
        'Amount_Sent_Normalized_Log', 'Amount_Received_Normalized_Log',
        'Amount_Difference_Pct', 'is_currency_exchange',
        'From Bank', 'To Bank', 'Is Laundering', 'Pattern']]
    
    return df


def universal_feature_engi(df):

    exchange_rates = get_exchange_rates(df)
    df = normalize_amounts(df, exchange_rates)
    df = log_transformer(df)
    df = universal_features_restructure(df)
    return df


