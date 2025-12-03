import pandas as pd
import numpy as np
import itertools
import inspect
import torch
from torch_geometric.data import Data
import data.data_utils as du
import data.feature_engi as fe
import data.data_functions as dfn


#from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --------------------------------------------------------------------------------------------------------------------------
# functions for processing the 'raw' data, splitting it into training, validation, and testing. ----------------------------
# packing it into 'regular' data and graph data. ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# main function for extracting the data, used in main scripth -------------------------------------

def get_data(df, data_paser, **kwargs):

    if not data_paser.ibm_fe:
        df = fe.universal_feature_engi(df)
        scaler_encoders = dfn.extract_enc_cats(df)
        edge_features = ['Timestamp', 'Amount_Sent_Normalized_Log', 'Amount_Received_Normalized_Log',
                        'Sent Currency', 'Received Currency', 'Payment Format', 
                        'Amount_Difference_Pct', 'is_currency_exchange']
    else:
        scaler_encoders = None
        edge_features = ['Timestamp', 'Amount Sent', 'Sent Currency', 
                         'Amount Received', 'Received Currency', 'Payment Format']
        
    # temporary that this is used
    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    
    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()

    # get timestamps and labels
    timestamps = torch.Tensor(df['Timestamp'].to_numpy())
    y = torch.LongTensor(df['Is Laundering'].to_numpy())

    valid_keys = inspect.signature(split_indices).parameters.keys()
    args = {key: kwargs[key] for key in valid_keys if key in kwargs}

    split_inds, test_perc = split_indices(timestamps, y, **args)
    #split_inds, test_perc = split_indices(timestamps, y, [0.6, 0.2])

    indices = [np.concatenate(split_inds[i]) for i in range(0,3)]

    packed_data = {}

    # pack graph data
    if data_paser.data_type == 'graph_data':
        packed_data['graph_data'] = pack_graph_data(df, y, timestamps, indices, edge_features)

    # pack non-graph data
    packed_data['regular_data'] = pack_regular_data(df, y, indices)

    return packed_data, scaler_encoders



# function for splitting the data into training, validation and testing -------------------------------------

def split_indices(timestamps, y, split_perc = [0.6, 0.2]):

    # Obtained indices for train, validation and testing
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]

    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []  # irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    test_perc = round(1 - sum(split_perc), 10)

    if test_perc > 0:
        split_perc.append(test_perc)

        for i, j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v / split_totals_sum for v in split_totals]
                split_error = [abs(v - t) / t for v, t in zip(split_props, split_perc)]
                score = max(split_error)  # - (split_totals_sum/total) + 1
                split_scores[(i, j)] = score
            else:
                continue

        i, j = min(split_scores, key=split_scores.get)
        # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]

    else:
        for i in I:
            split_totals = [d_ts[:i].sum(), d_ts[i:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v / split_totals_sum for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, split_perc)]
            score = max(split_error)  # - (split_totals_sum/total) + 1
            split_scores[i] = score

        i = min(split_scores, key=split_scores.get)
        # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, len(daily_totals)))]


    split_inds = {k: [] for k in range(len(split_perc))}
    for i in range(len(split_perc)):
        for day in split[i]:
            split_inds[i].append(daily_inds[day])

    return split_inds, test_perc



# pack regular data function -------------------------------------

def pack_regular_data(df, y, indices):

    if isinstance(y, torch.Tensor):
        y = np.array(y)

    packed_data = {}
    for idx, data  in enumerate(['train_data', 'vali_data', 'test_data']):
        packed_data[data] = {'x': df.iloc[indices[idx],:], 'y': y[indices[idx]]}

    return packed_data



# pack graph data function -------------------------------------

def pack_graph_data(df_edges, y, timestamps, indices, edge_features):

    train_indices, vali_indices,test_indices = indices

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})

    # set edge and node features
    node_features = ['Feature']

    # Extract node features from nodes, without their ID
    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()

    # Create the edge_index, a 2 x number_of_edges matrix. First row is the source node, second row is the destination,
    # turn it into a torch, and create a torch of edge_attributes
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    train_x, vali_x, test_x = x, x, x
    edge_train = train_indices
    edge_vali = np.concatenate([train_indices, vali_indices])

    train_edge_index, train_edge_attr, train_y, train_edge_times = edge_index[:, edge_train], edge_attr[edge_train], y[edge_train], timestamps[edge_train]
    vali_edge_index, vali_edge_attr, vali_y, vali_edge_times = edge_index[:, edge_vali], edge_attr[edge_vali], y[edge_vali], timestamps[edge_vali]
    test_edge_index, test_edge_attr, test_y, test_edge_times = edge_index, edge_attr, y, timestamps

    train_data = du.GraphData(x=train_x, y=train_y, edge_index=train_edge_index, edge_attr=train_edge_attr, timestamps=train_edge_times)
    vali_data = du.GraphData(x=vali_x, y=vali_y, edge_index=vali_edge_index, edge_attr=vali_edge_attr, timestamps=vali_edge_times)
    test_data = du.GraphData(x=test_x, y=test_y, edge_index=test_edge_index, edge_attr=test_edge_attr, timestamps=test_edge_times)

    for data in [train_data, vali_data, test_data]:
        du.update_nr_nodes_for_gd(data)

    packed_data = {'train_data': {'df': train_data}, 
                'vali_data': {'df': vali_data},
                'test_data': {'df': test_data}}

    for index, name in zip([train_indices, vali_indices, test_indices], list(packed_data)):
        packed_data[name]['pred_indices'] = torch.tensor(index)

    return packed_data

    
    










    #edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    #edge_features = ['Timestamp', 'Amount_Sent_Normalized_Log', 'Amount_Received_Normalized_Log',
                     #'Sent Currency', 'Received Currency', 'Payment Format', 
                     #'Amount_Difference_Pct', 'is_currency_exchange']