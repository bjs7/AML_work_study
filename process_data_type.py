import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from snapml import GraphFeaturePreprocessor
from gp_params import params


def get_bank_indices(df, bank, include_tobank = True):

    listwith_tobank = []
    listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == bank)[0])]
    if include_tobank:
        listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == bank)[0])]
    indices_to_get = list(set(listwith_frombank + listwith_tobank))
    bank_indices = sorted(indices_to_get)

    return bank_indices


def process_regular_data(data, bank_indices, args):

    df = copy.copy(data)
    #df = copy.copy(unfil_data)

    data_features = ['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 'Sent Currency', 'Payment Format']
    X = df['x'].iloc[bank_indices,:].loc[:,data_features]
    X = X.reset_index(drop=True)
    y = df['y'][bank_indices].copy()

    # graph_feature_preprocessing using snapML

    gp = GraphFeaturePreprocessor()
    gp.set_params(params)

    X_gf = gp.fit_transform(X[['EdgeID', 'from_id', 'to_id', 'Timestamp']].astype("float64"))
    X_gf = X_gf[:,4:]
    
    # remove EdgeID as it is no longer needed
    X = X.drop('EdgeID', axis='columns')

    # Count the amount of times an account sends or receives
    for ID in ['from_id', 'to_id']:
        X.iloc[:, np.where(X.columns == ID)[0][0]] = X.iloc[:, np.where(X.columns == ID)[0][0]].map(X.iloc[:, np.where(X.columns == ID)[0][0]].value_counts())

    period = 86400
    timestamps = X.loc[:, 'Timestamp']
    sin_component = np.sin(2 * np.pi * timestamps / period)
    cos_component = np.cos(2 * np.pi * timestamps / period)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(X.loc[:,['Timestamp', 'Amount Sent']])
    X['Timestamp'] = X['Timestamp'].astype(float)
    X.loc[:,['Timestamp']] = scaled_values[:,0]
    X.loc[:, ['Amount Sent']] = scaled_values[:,1]

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(np.array([X.loc[:, 'Payment Format']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Payment Format"]))
    X = pd.concat([X, encoded_df], axis=1)

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(np.array([X.loc[:, 'Sent Currency']]).T)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Sent Currency"]))
    X = pd.concat([X, encoded_df], axis=1)

    sin_cos = pd.concat([pd.DataFrame({'sin_component': sin_component.values}), pd.DataFrame({'cos_component': cos_component.values})], axis=1)
    X = pd.concat([X, sin_cos], axis = 1)

    # finally concat X and the graph features
    X_gf = pd.DataFrame(X_gf, columns=[f'graph_feature_{i + 1}' for i in range(X_gf.shape[1])])
    X = pd.concat([X, X_gf], axis=1)

    return {'X': X, 'y': y, 'bank_indices': bank_indices, 'scaler': scaler}


def process_graph_data(data, bank_indices, args):

    df = copy.copy(data)
    #df = copy.copy(unfil_data)
    
    if args.scenario == 'individual banks':
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
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    #encoded_payment = encoder.fit_transform(np.array(df.edge_attr[:,2]).reshape(-1,1)) # new
    encoded_payment = encoder.fit_transform(np.array([df.edge_attr[:,2]]).T) # old

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    #encoded_currency = encoder.fit_transform(np.array(df.edge_attr[:,3]).reshape(-1,1)) # new
    encoded_currency = encoder.fit_transform(np.array([df.edge_attr[:,3]]).T) # old

    df.edge_attr = torch.cat([torch.tensor(np.arange(df.edge_attr.shape[0])).unsqueeze(1), df.edge_attr[:, [0, 1]],
                              torch.tensor(encoded_payment), torch.tensor(encoded_currency),
                              sin_component, cos_component], axis=1).float()

    return {'df': df, 'bank_indices': torch.tensor(bank_indices)}

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


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



