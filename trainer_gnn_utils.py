import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import NeighborLoader
import gnn_models as gnn_m

def get_model(sample_batch, nn_size):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)
    #e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    model = gnn_m.GINe(
        num_features=n_feats, num_gnn_layers=nn_size, n_classes=2,
        n_hidden=round(66.00315515631006), residual=False, edge_updates=True, edge_dim=e_dim,
        dropout=0.00983468338330501, final_dropout=0.10527690625126304
    )


    return model

def account_for_time(batch, main_data):

    max_time = main_data.edge_attr[batch.input_id, 1].max()
    mask = batch.edge_attr[:, 1] <= max_time

    #max_index = batch.input_id.max().item()
    #mask = batch.e_id <= max_index

    batch.edge_index = batch.edge_index[:, mask]
    batch.edge_attr = batch.edge_attr[mask]
    batch.y = batch.y[mask]
    batch.timestamps = batch.timestamps[mask]
    batch.e_id = batch.e_id[mask]

    unique_nodes, new_indices = torch.unique(batch.edge_index, return_inverse=True)
    batch.edge_index = new_indices.view(2, -1)
    batch.x = batch.x[unique_nodes]

    return batch




