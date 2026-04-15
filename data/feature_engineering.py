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

def feature_engi_regular_data(data, data_parser, scaler_encoders = None):

    #df = copy.copy(data)
    #df = train_data

    df = copy.deepcopy(data)

    # Guard: return as-is if the split is empty (some parties have no data in a given split)
    if df.get('x') is None or df['x'].shape[0] == 0:
        return df
    #df = data['df']

    scl_enc = {ele: scaler_encoders.get(ele) for ele in 
               ('scaler_amount', 'encoder_currency', 'encoder_payment_format', 'gfp')
               } if scaler_encoders else {}
    
    data_features = ['EdgeID', 'from_id', 'to_id']
    if not data_parser.ibm_fe:
        data_features += ['Timestamp', 'Amount Sent', 'Amount Received', 'Sent Currency', 
                          'Received Currency', 'Payment Format', 'is_currency_exchange']
    else:
        if data_parser.normalize_currency:
            data_features += ['Timestamp', 'Amount_Received_Normalized', 'Received Currency', 'Payment Format']
        else:
            data_features += ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']

    x = df['x'].loc[:,data_features]
    y = df['y']

    # GFP -----------------------------------------------------------------------------------------------------------------
    # graph_feature_preprocessing using snapML

    # switch the position of graph feature process and the other features? To include those features in graph features processing?
    # IBM paper: vertex stats use Amount and Timestamp (vertex_stats_cols=[3,4]).
    # Amount Received is at index 4; without it column 4 doesn't exist and the
    # first derived feature would also be silently dropped by the [:,5:] slice.
    if 'Amount Received' in x.columns:
        gfp_amount = 'Amount Received'
    elif 'Amount_Received_Normalized' in x.columns:
        gfp_amount = 'Amount_Received_Normalized'
    else:
        gfp_amount = None

    gfp_cols = ['EdgeID', 'from_id', 'to_id', 'Timestamp']
    if gfp_amount:
        gfp_cols.append(gfp_amount)

    if not scl_enc.get('gfp'):
        scl_enc['gfp'] = GraphFeaturePreprocessor()
        scl_enc['gfp'].set_params(tu.gfpparams)
        x_gf = scl_enc['gfp'].fit_transform(x[gfp_cols].astype("float64"))
    else:
        x_gf = scl_enc['gfp'].transform(x[gfp_cols].astype("float64"))
    x_gf = x_gf[:, len(gfp_cols):]  # strip input columns, keep only derived features
    
    # remove EdgeID as it is no longer needed
    x = x.drop('EdgeID', axis='columns')
    x = x.drop('from_id', axis='columns')
    x = x.drop('to_id', axis='columns')

    if data_parser.ibm_fe:
        x_gf = pd.DataFrame(x_gf, columns=[f'graph_feature_{i + 1}' for i in range(x_gf.shape[1])])
        x = pd.concat([x, x_gf], axis=1)
        return {'x': x, 'y': y, 'scaler_encoders': scl_enc}        
    
    # Timestamp: scale to days ------------------------------------------------------------------------------
    x.loc[:, 'Timestamp'] = x.loc[:, 'Timestamp'] / 86400.0

    # Amounts: log transform then standardize (shared scaler for sent & received) ---------------------------
    x.loc[:,'Amount Sent'] = np.log(x.loc[:,'Amount Sent'])
    x.loc[:,'Amount Received'] = np.log(x.loc[:,'Amount Received'])

    if not scl_enc.get('scaler_amount'):
        scl_enc['scaler_amount'] = StandardScaler()
        combined = np.concatenate([x.loc[:,'Amount Sent'], x.loc[:,'Amount Received']]).reshape(-1,1)
        scl_enc['scaler_amount'].fit(combined)

    for index in ['Amount Sent', 'Amount Received']:
        reshaped = np.array(x.loc[:,index]).reshape(-1,1)
        scaled = scl_enc['scaler_amount'].transform(reshaped)
        x.loc[:,index] = scaled

    # Encoding: OneHotEncode currencies and payment format -------------------------------------------------
    features = ['Sent Currency', 'Received Currency', 'Payment Format']
    encoded = {}

    for feature in features:
        if feature in ('Sent Currency', 'Received Currency'):
            encoder_key = 'encoder_currency'
        else:
            encoder_key = 'encoder_payment_format'
        column_data = x.loc[:,feature]
        if not scl_enc.get(encoder_key):
            scl_enc[encoder_key] = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded[feature] = scl_enc[encoder_key].fit_transform(np.array(column_data).reshape(-1,1))
        else:
            encoded[feature] = scl_enc[encoder_key].transform(np.array(column_data).reshape(-1,1))

    # Pack -----------------------------------------------------------------------------------------------------------------

    for feature in features:
        if feature in ('Sent Currency', 'Received Currency'):
            encoder_key = 'encoder_currency'
        else:
            encoder_key = 'encoder_payment_format'
        encoded_df = pd.DataFrame(encoded[feature], columns=scl_enc[encoder_key].get_feature_names_out([feature]))
        x = pd.concat([x.drop(columns=[feature]), encoded_df], axis=1)

    # finally concat X and the graph features
    x_gf = pd.DataFrame(x_gf, columns=[f'graph_feature_{i + 1}' for i in range(x_gf.shape[1])])
    x = pd.concat([x, x_gf], axis=1)

    return {'x': x, 'y': y, 'scaler_encoders': scl_enc}


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

def feature_engi_graph_data(data, args, scaler_encoders=None, edge_feat_start=0):
    """Per-party feature engineering for graph data.

    Standard layout (edge_feat_start=0):
      0=Timestamp, 1=Amount Sent, 2=Amount Received,
      3=Sent Currency, 4=Received Currency, 5=Payment Format, 6=is_currency_exchange.

    With edge_feat_start=1 (FedGraph/vertical FL, arange ID prepended):
      0=ID, 1=Timestamp, 2=Amount Sent, ..., 7=is_currency_exchange.
    The ID column is preserved as-is in the output.
    """
    data = copy.deepcopy(data)
    df = data['df']

    if df is None:
        return data
    # Skip if too few edges to fit scalers AND no pre-fitted encoders to use
    if df.edge_attr.shape[0] < 2 and not scaler_encoders:
        return data

    scl_enc = {ele: scaler_encoders.get(ele) for ele in
               ('scaler_amount', 'encoder_currency', 'encoder_payment_format')
               } if scaler_encoders else {}

    s = edge_feat_start  # column offset

    # Timestamp: scale to days ------------------------------------------------------------------------------
    df.edge_attr[:, s + 0] = df.edge_attr[:, s + 0] / 86400.0

    # Amounts: log transform then standardize (shared scaler for sent & received) ---------------------------
    df.edge_attr[:, s + 1] = torch.log(df.edge_attr[:, s + 1])
    df.edge_attr[:, s + 2] = torch.log(df.edge_attr[:, s + 2])

    if not scl_enc.get('scaler_amount'):
        scl_enc['scaler_amount'] = StandardScaler()
        combined = torch.cat([df.edge_attr[:, s + 1], df.edge_attr[:, s + 2]]).reshape(-1, 1)
        scl_enc['scaler_amount'].fit(combined)

    for index in [s + 1, s + 2]:
        reshaped = df.edge_attr[:, index].reshape(-1, 1)
        scaled = scl_enc['scaler_amount'].transform(reshaped)
        df.edge_attr[:, index] = torch.reshape(torch.tensor(scaled), (-1,))

    # Node features: z-normalize ---------------------------------------------------------------------------
    if df.x.shape[0] > 1:
        df.x = du.z_norm(df.x)
    else:
        df.x = torch.tensor([[0.]])

    # Encoding: OneHotEncode currencies and payment format -------------------------------------------------
    # Use distinct keys so both Sent Currency and Received Currency are encoded
    column_encoders = {'currency_sent': s + 3, 'currency_received': s + 4, 'payment_format': s + 5}
    encoded = {}

    for feature, col_idx in column_encoders.items():
        if feature in ('currency_sent', 'currency_received'):
            encoder_key = 'encoder_currency'
        else:
            encoder_key = 'encoder_payment_format'

        column_data = df.edge_attr[:, col_idx].reshape(-1, 1)
        if not scl_enc.get(encoder_key):
            scl_enc[encoder_key] = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded[feature] = scl_enc[encoder_key].fit_transform(column_data)
        else:
            encoded[feature] = scl_enc[encoder_key].transform(column_data)

    # Pack: ID col(s) (if any) + Timestamp, Amount Sent, Amount Received, is_currency_exchange + encoded cats
    cols_to_keep = list(range(edge_feat_start)) + [s + i for i in [0, 1, 2, 6]]

    df.edge_attr = torch.cat([df.edge_attr[:, cols_to_keep],
                              *[torch.tensor(enc_val).float() for enc_val in encoded.values()]
                              ], dim=1).float()

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

    # Boolean to check whether the currency sent is the same as the received one.
    df['is_currency_exchange'] = (df['Sent Currency'] != df['Received Currency']).astype(int)

    # Restructure the dataframe
    columns = ['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent',
               'Sent Currency', 'Amount Received', 'Received Currency',
               'Payment Format', 'is_currency_exchange',
               'From Bank', 'To Bank', 'Is Laundering']
    if 'Pattern' in df.columns:
        columns.append('Pattern')

    df = df[columns]
    return df


def universal_feature_engi(df, normalize_currency=False):

    if normalize_currency:
        # normalize_amounts already called in get_data, just overwrite raw columns
        df['Amount Sent'] = df['Amount_Sent_Normalized']
        df['Amount Received'] = df['Amount_Received_Normalized']

    df = universal_features_restructure(df)
    return df


