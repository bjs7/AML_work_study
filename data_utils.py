import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
import numpy as np
import trainer_utils as tu
import pandas as pd

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def update_nr_nodes(df):
    max_n_id = np.array(df.edge_index.max() + 1)
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    x = torch.tensor(df_nodes.loc[:, ['Feature']].to_numpy()).float()
    df.x = x

class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None,
        readout: str = 'edge',
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

