
# packages for FL

from abc import ABC, abstractmethod
import logging
import utils as fl_utils
import torch
from federated_learning.registry import GNN_REGISTRY

import torch
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader

logger = logging.getLogger(__name__)



# potentially split this up, so that it is in two, like one where parameters are extracted once
# and then used to get the model in the register.
# need to sort how to init about parties and whether they need batching or not, and
# also how to adjust the data set size, such that copy can be avoided.

# maybe move get_gnn out of GNN, and into manager 100%, and then send to parties?

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GNN(ABC):

    def __init__(self, manager, hyperparams, node_features, edge_dim):
        super().__init__()
        self._get_gnn_loss_optimizer(manager, hyperparams, node_features, edge_dim)
        # I could potentially also ahve the data in here, if it is just pointers
        # but keep it out for now

    @staticmethod
    def _create_gnn_model(manager, hyperparams, node_features, edge_dim):
        # add the GNN_REGISTRY as an attribute to the manager?
        model_name = manager.args['fl_parser'].model
        if model_name not in GNN_REGISTRY:
            raise ValueError(f"Unknown algo type: {model_name}")
        gnn_init = GNN_REGISTRY[model_name]

        arguments = {'num_features': node_features, 'num_gnn_layers': hyperparams.get('num_gnn_layers'),
                    'n_classes': 2, 'n_hidden': hyperparams.get('hidden_embedding_size'),
                    'residual': False, 'edge_updates': manager.args['gnn_parser'].emlps,
                    'edge_dim': edge_dim, 'dropout': hyperparams.get('dropout'),
                    'final_dropout': hyperparams.get('final_dropout')}
        
        return gnn_init(**arguments)

    def _init_model_configs(self):
        return 0

    def _get_gnn_loss_optimizer(self, manager, hyperparams, node_features, edge_dim):
        self.gnn = self._create_gnn_model(manager, hyperparams, node_features, edge_dim)
        self.gnn.to(device)  # Move model to GPU
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=hyperparams.get('learning_rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparams.get('w_ce1'), hyperparams.get('w_ce2')]).to(device)) 

    def update_w(self, df):

        self.gnn.train()
        self.optimizer.zero_grad()

        # Move data to device
        df = df.to(device)

        pred = self.gnn(df.x, df.edge_index, df.edge_attr)
        loss = self.loss_fn(pred, df.y)

        # Check for unusual loss values
        if torch.isnan(loss):
            logger.error("Loss is NaN! Check for numerical instability, learning rate, or data issues")
        elif torch.isinf(loss):
            logger.error("Loss is infinite! Check for numerical overflow in model or data")
        elif loss.item() > 100:
            logger.warning("Very high loss value: %.4f - may indicate learning issues", loss.item())

        loss.backward()
        self.optimizer.step()

    def predict(self, graph_data):

        df = graph_data.get('df')
        pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            # Move data to device
            df = df.to(device)
            pred = self.gnn(df.x, df.edge_index, df.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred

        predictions = pred.softmax(dim=1)[:,1].cpu().numpy()

        # Check for unusual predictions
        if torch.isnan(pred).any():
            logger.warning("Model predictions contain NaN values!")
        elif (predictions == 0).all():
            logger.warning("All predictions are zero - model may not be learning")

        return predictions

    def predict_binary(self, graph_data):

        df = graph_data.get('df')
        pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            # Move data to device
            df = df.to(device)
            pred = self.gnn(df.x, df.edge_index, df.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred

        return pred.argmax(dim=-1).cpu().numpy()  # Move predictions back to CPU and convert to numpy



#torch.randn(3, 5).softmax(dim=0)
#torch.randn(10, 2).softmax(dim=1)

#pred.softmax(dim=1)[:,1]
#pred.argmax(dim=-1)

#assert ((pred.softmax(dim=1)[:,1] > 0.5) == (pred.argmax(dim=-1) == 1)).all()

#self = manager._party.model
#graph_data = manager._party.get_eval_data()
#pred = self.predict(graph_data)

#pred.softmax(dim=1)

#self = party.model
#graph_data = party.get_eval_data()
#pred = self.predict(graph_data)

#pred.softmax(dim=1)

#self.predict_binary(graph_data)

#torch.exp(pred[0,:])/sum(torch.exp(pred[0,:]))
#torch.exp(torch.tensor(100))

# ---------------------------------------------------------------------------------------------------------------------------------------
# Util functions for the gnn models -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

# select loader for sampling edges ---------------------------------------------------------------------------------------------------------

def get_loaders(train_data, vali_data, pred_indices, m_param, batch_size, transform = None):

    train_loader = LinkNeighborLoader(train_data, num_neighbors=m_param.get('num_neighbors'), edge_label = train_data.y, batch_size=batch_size, shuffle=True, transform=transform)
    vali_loader = LinkNeighborLoader(vali_data, num_neighbors=m_param.get('num_neighbors'), 
                                     edge_label_index=vali_data.edge_index[:, pred_indices],
                                     edge_label=vali_data.y[pred_indices],
                                        batch_size=batch_size, shuffle=False, transform=transform)
    
    return train_loader, vali_loader

class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)



