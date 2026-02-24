import pandas as pd
import numpy as np
import itertools
import inspect
import torch
from torch_geometric.data import Data
import data.data_utils as du
import data.feature_engineering as fe
import data.data_functions as dfn

from data.relevant_banks import load_relevant_banks

#from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --------------------------------------------------------------------------------------------------------------------------
# functions for processing the 'raw' data, splitting it into training, validation, and testing. ----------------------------
# packing it into 'regular' data and graph data. ---------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# main function for extracting the data, used in main scripth -------------------------------------

def get_data(df, data_parser, **kwargs):

    # Normalize currency if flag is set (heterogeneity experiment, applies to both FE paths)
    if data_parser.normalize_currency:
        from data.exchange_rates import load_or_extract
        exchange_rates = load_or_extract(data_parser)
        df = fe.normalize_amounts(df, exchange_rates)

    if not data_parser.ibm_fe:
        df = fe.universal_feature_engi(df, normalize_currency=data_parser.normalize_currency)
        scaler_encoders = dfn.extract_enc_cats(df)
        edge_features = ['Timestamp', 'Amount Sent', 'Amount Received',
                         'Sent Currency', 'Received Currency', 'Payment Format',
                         'is_currency_exchange']
    else:
        scaler_encoders = None
        if data_parser.normalize_currency:
            edge_features = ['Timestamp', 'Amount_Received_Normalized', 'Received Currency', 'Payment Format']
        else:
            edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']

    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()

    # get timestamps and labels
    timestamps = torch.Tensor(df['Timestamp'].to_numpy())
    y = torch.LongTensor(df['Is Laundering'].to_numpy())

    valid_keys = inspect.signature(split_indices).parameters.keys()
    args = {key: kwargs[key] for key in valid_keys if key in kwargs}


    if not data_parser.testing:
        split_inds, test_perc = split_indices(timestamps, y, **args)
        #split_inds, test_perc = split_indices(timestamps, y, [0.6, 0.2])
        indices = [np.concatenate(split_inds[i]) for i in range(0,3)]
    else:
        indices = sub_indices(df)

    # need to make sure that indices do not need to be reset etc.
    if data_parser.eval_mode == 'comparable':

        individual_indices = load_relevant_banks(data_parser).get('individual').get('indices')
        
        new_train_indices = indices[0][np.isin(indices[0], individual_indices)]
        new_vali_indices = indices[1][np.isin(indices[1], individual_indices)]
        new_test_indices = indices[2][np.isin(indices[2], individual_indices)]

        indices = [new_train_indices, new_vali_indices, new_test_indices]

    packed_data = {}

    # pack graph data
    if data_parser.data_type == 'graph_data':
        packed_data['graph_data'] = pack_graph_data(df, y, timestamps, indices, edge_features, data_parser.eval_mode)

    # pack non-graph data
    packed_data['regular_data'] = pack_regular_data(df, y, indices, data_parser.eval_mode)

    return packed_data, scaler_encoders

#import copy
#indices_holder = copy.deepcopy(indices)
#df_edges = copy.deepcopy(df)

def sub_indices(sub_df):
    
    train_indices = sub_df.iloc[:round(sub_df.shape[0] * 0.6),:].index.to_numpy()
    vali_indices = sub_df.iloc[round(sub_df.shape[0] * 0.6):round(sub_df.shape[0] * 0.8),:].index.to_numpy()
    test_indices = sub_df.iloc[round(sub_df.shape[0] * 0.8):,:].index.to_numpy()

    return [train_indices, vali_indices, test_indices] 


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

def pack_regular_data(df, y, indices, eval_mode = 'system'):

    if isinstance(y, torch.Tensor):
        y = np.array(y)

    max_idx = 0
    packed_data = {}
    for idx, data  in enumerate(['train_data', 'vali_data', 'test_data']):
        packed_data[data] = {'x': df.iloc[indices[idx],:], 'y': y[indices[idx]]}
        if eval_mode != 'system':
            tmp_index = pd.Index(range(max_idx, packed_data[data]['x'].shape[0] + max_idx))
            packed_data[data]['x'] = packed_data[data]['x'].set_index(tmp_index)
            max_idx = packed_data[data]['x'].index.max() + 1
            

    return packed_data



# pack graph data function -------------------------------------

def pack_graph_data(df_edges, y, timestamps, indices, edge_features, eval_mode = 'system'):

    train_indices, vali_indices, test_indices = indices

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
    if eval_mode != 'system':
        edge_test = np.concatenate([train_indices, vali_indices, test_indices])

    train_edge_index, train_edge_attr, train_y, train_edge_times = edge_index[:, edge_train], edge_attr[edge_train], y[edge_train], timestamps[edge_train]
    vali_edge_index, vali_edge_attr, vali_y, vali_edge_times = edge_index[:, edge_vali], edge_attr[edge_vali], y[edge_vali], timestamps[edge_vali]
    if eval_mode != 'system':
        test_edge_index, test_edge_attr, test_y, test_edge_times = edge_index[:, edge_test], edge_attr[edge_test], y[edge_test], timestamps[edge_test]
    else:
        test_edge_index, test_edge_attr, test_y, test_edge_times = edge_index, edge_attr, y, timestamps

    train_data = du.GraphData(x=train_x, y=train_y, edge_index=train_edge_index, edge_attr=train_edge_attr, timestamps=train_edge_times)
    vali_data = du.GraphData(x=vali_x, y=vali_y, edge_index=vali_edge_index, edge_attr=vali_edge_attr, timestamps=vali_edge_times)
    test_data = du.GraphData(x=test_x, y=test_y, edge_index=test_edge_index, edge_attr=test_edge_attr, timestamps=test_edge_times)

    for data in [train_data, vali_data, test_data]:
        du.update_nr_nodes_for_gd(data)

    packed_data = {'train_data': {'df': train_data}, 
                'vali_data': {'df': vali_data},
                'test_data': {'df': test_data}}

    max_idx = 0
    for index, name in zip([train_indices, vali_indices, test_indices], list(packed_data)):
        packed_data[name]['pred_indices'] = torch.tensor(range(max_idx, len(index) + max_idx))
        max_idx = int(packed_data[name]['pred_indices'].max()) + 1
            
    return packed_data


