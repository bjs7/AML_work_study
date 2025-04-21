import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import NeighborLoader
import gnn_models as gnn_m

def get_model(sample_batch, m_param, m_settings, args):

    e_dim_adjust = 1 if m_settings.get('include_time') else 2
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - e_dim_adjust) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - e_dim_adjust)
    #e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    model = gnn_m.GINe(
        num_features=n_feats, num_gnn_layers=m_param.get('gnn_layers'), n_classes=2,
        n_hidden=m_param.get('hidden_embedding_size'), residual=False, edge_updates=args.emlps, edge_dim=e_dim,
        dropout=m_param.get('dropout'), final_dropout=m_param.get('dropout')
    )


    return model

def get_loaders(train_data, vali_data, pred_indices, m_param, batch_size, transform = None):

    train_loader = LinkNeighborLoader(train_data, num_neighbors=m_param.get('num_neighbors'), batch_size=batch_size, shuffle=True, transform=transform)
    vali_loader = LinkNeighborLoader(vali_data, num_neighbors=m_param.get('num_neighbors'), edge_label_index=vali_data.edge_index[:, pred_indices],
                       batch_size=batch_size, shuffle=False, transform=None)
    
    return train_loader, vali_loader

# The loader has an argument called time_attr which could potentially be used to adjust for time?
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




