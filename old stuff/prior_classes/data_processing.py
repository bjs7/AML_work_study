import pandas as pd
import numpy as np
import itertools
import inspect
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch_geometric.data import Data
import data_utils as du

def get_data(df_edges, **kwargs):

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    # get timestamps and labels
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    valid_keys = inspect.signature(split_indices).parameters.keys()
    args = {key: kwargs[key] for key in valid_keys if key in kwargs}

    split_inds, test_perc = split_indices(timestamps, y, **args)
    #split_inds, test_perc = split_indices(timestamps, y, [0.6, 0.2])

    train_indices = np.concatenate(split_inds[0])
    vali_indices = np.concatenate(split_inds[1])
    indices = [train_indices, vali_indices]

    if (test_perc > 0):
        test_indices = np.concatenate(split_inds[2])
        indices.append(test_indices)

    packed_data = {}

    # pack graph data
    packed_data['graph_data'] = pack_graph_data(df_edges, y, timestamps, indices, test_perc)

    # pack non-graph data
    packed_data['regular_data'] = pack_regular_data(df_edges, y, indices, test_perc)

    return packed_data



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


def pack_regular_data(df_edges, y, indices, test_perc):

    if isinstance(y, torch.Tensor):
        y = np.array(y)

    train_indices = indices[0]
    vali_indices = indices[1]

    train_data = {'x': df_edges.iloc[train_indices,:], 'y': y[train_indices]}
    vali_data = {'x': df_edges.iloc[vali_indices, :], 'y': y[vali_indices]}

    if test_perc > 0:
        test_indices = indices[2]
        test_data = {'x': df_edges.iloc[test_indices, :], 'y': y[test_indices]}

        return {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}

    return {'train_data': train_data, 'vali_data': vali_data}



def pack_graph_data(df_edges, y, timestamps, indices, test_perc = 0):

    train_indices = indices[0]
    vali_indices = indices[1]

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})

    # set edge and node features
    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    # Extract node features from nodes, without their ID
    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()

    # Create the edge_index, a 2 x number_of_edges matrix. First row is the source node, second row is the destination,
    # turn it into a torch, and create a torch of edge_attributes
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    train_x, vali_x = x, x
    edge_train = train_indices
    edge_vali = np.concatenate([train_indices, vali_indices])

    train_edge_index, train_edge_attr, train_y, train_edge_times = edge_index[:, edge_train], edge_attr[edge_train], y[edge_train], timestamps[edge_train]
    vali_edge_index, vali_edge_attr, vali_y, vali_edge_times = edge_index[:, edge_vali], edge_attr[edge_vali], y[edge_vali], timestamps[edge_vali]

    #train_data = Data(x=train_x, y=train_y, edge_index=train_edge_index, edge_attr=train_edge_attr, timestamps=train_edge_times)
    train_data = du.GraphData(x=train_x, y=train_y, edge_index=train_edge_index, edge_attr=train_edge_attr, timestamps=train_edge_times)
    #vali_data = Data(x=vali_x, y=vali_y, edge_index=vali_edge_index, edge_attr=vali_edge_attr, timestamps=vali_edge_times)
    vali_data = du.GraphData(x=vali_x, y=vali_y, edge_index=vali_edge_index, edge_attr=vali_edge_attr, timestamps=vali_edge_times)

    du.update_nr_nodes_for_gd(train_data)
    du.update_nr_nodes_for_gd(vali_data)

    train_indices = torch.tensor(train_indices)
    vali_indices = torch.tensor(vali_indices)

    if test_perc > 0:
        test_indices = indices[2]
        test_x = x

        test_edge_index, test_edge_attr, test_y, test_edge_times = edge_index, edge_attr, y, timestamps
        #test_data = Data(x=test_x, y=test_y, edge_index=test_edge_index, edge_attr=test_edge_attr, timestamps=test_edge_times)
        test_data = du.GraphData(x=test_x, y=test_y, edge_index=test_edge_index, edge_attr=test_edge_attr, timestamps=test_edge_times)

        du.update_nr_nodes_for_gd(test_data)
        test_indices = torch.tensor(test_indices)

        #return {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
        return {'train_data': {'df': train_data, 'pred_indices': train_indices}, 
                'vali_data': {'df': vali_data, 'pred_indices': vali_indices}, 
                'test_data': {'df': test_data, 'pred_indices': test_indices}}

    #return {'train_data':train_data, 'vali_data': vali_data}
    return {'train_data': {'df ': train_data, 'pred_indices': train_indices}, 
                'vali_data': {'df ': vali_data, 'pred_indices': vali_indices}}
