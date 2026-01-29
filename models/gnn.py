
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

        # FedGraph/FedAvg: parties compute locally with potentially very few observations,
        # so BatchNorm can fail. Always use LayerNorm for these algorithms.
        fl_algo = manager.args['fl_parser'].fl_algo
        use_batchnorm = manager.args['data_parser'].batching and fl_algo not in ('FedGraph', 'FedAvg')

        arguments = {'num_features': node_features, 'num_gnn_layers': hyperparams.get('num_gnn_layers'),
                    'n_classes': 2, 'n_hidden': hyperparams.get('hidden_embedding_size'),
                    'residual': False, 'edge_updates': manager.args['gnn_parser'].emlps,
                    'edge_dim': edge_dim, 'dropout': hyperparams.get('dropout'),
                    'final_dropout': hyperparams.get('final_dropout'),
                    'batching': use_batchnorm}
        
        return gnn_init(**arguments)

    def _init_model_configs(self):
        return 0

    def _get_gnn_loss_optimizer(self, manager, hyperparams, node_features, edge_dim):
        self.gnn = self._create_gnn_model(manager, hyperparams, node_features, edge_dim)
        self.gnn.to(device)  # Move model to GPU
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=hyperparams.get('learning_rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparams.get('w_ce1'), hyperparams.get('w_ce2')]).to(device)) 

    def update_weights(self, gd, mask):

        self.gnn.train()
        self.optimizer.zero_grad()

        # Move data to device
        gd = gd.to(device)

        out = self.gnn(gd.x, gd.edge_index, gd.edge_attr)
        pred = out[mask]

        loss = self.loss_fn(pred, gd.y[mask])

        loss.backward()
        self.optimizer.step()

        return pred, gd.y[mask], loss.item()
    
    def update_weights_no_batching(self, gd):

        #gd = gd.get('df')

        self.gnn.train()
        self.optimizer.zero_grad()

        # Move data to device
        gd = gd.to(device)
        pred = self.gnn(gd.x, gd.edge_index, gd.edge_attr)
        loss = self.loss_fn(pred, gd.y)

        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def predict(self, gd, mask):

        #df = gd.get('df')
        #pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            gd = gd.to(device)
            out = self.gnn(gd.x, gd.edge_index, gd.edge_attr)
            pred = out[mask]
            #pred = pred[pred_indices] if pred_indices is not None else pred

        return pred.softmax(dim=1)[:,1].detach().cpu()
    
    @torch.no_grad()
    def predict_no_batching(self, gd):

        pred_indices = gd.get('pred_indices')
        gd = gd.get('df')        

        self.gnn.eval()
        with torch.no_grad():
            gd = gd.to(device)
            pred = self.gnn(gd.x, gd.edge_index, gd.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred

        return pred.softmax(dim=1)[:,1].detach().cpu().numpy()

    @torch.no_grad()
    def predict_binary(self, graph_data):

        df = graph_data.get('df')
        pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            df = df.to(device)
            pred = self.gnn(df.x, df.edge_index, df.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred

        return pred.argmax(dim=-1).cpu().numpy() 

# ---------------------------------------------------------------------------------------------------------------------------------------
# Util functions for the gnn models -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import torch

# if using batch.e_id then one would also need to add the original edges indices for the added edges

#data = eval_data
#loader = eval_loader
#indices = eval_indices
# current approach doesn't work
# compare 
# batch_edge_ids = torch.concat([ids_in_batch, missing_ids])
# and 
# batch.edge_attr[mask,0].int()
# and what they drag out
# and how it is used "outside"


def batching_masker(batch, data, loader, indices):

    indices = indices.detach().cpu()
    batch_edge_inds = indices[batch.input_id.detach().cpu()]
    batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
    #mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
    
    missing_seed_edges = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

    if missing_seed_edges.sum() != 0:

        missing_ids = batch_edge_ids[missing_seed_edges].int()
        ids_in_batch = batch_edge_ids[~missing_seed_edges].int()

        edge_labels_add = batch.edge_label_index[:,missing_seed_edges].detach().clone()
        edge_attr_add = data.edge_attr[missing_ids, :].detach().clone()
        y_add = data.y[missing_ids].detach().clone()

        batch.edge_index = torch.cat([batch.edge_index, edge_labels_add], dim=1)
        batch.edge_attr = torch.cat([batch.edge_attr, edge_attr_add], dim=0)
        batch.y = torch.cat([batch.y, y_add], dim=0)
        
        #batch_edge_ids = torch.concat([ids_in_batch, missing_ids])
        #mask = torch.cat((mask, torch.ones(y_add.shape[0], dtype=torch.bool)))
    #torch.all(torch.sort(batch.edge_attr[mask,0])[0] == torch.sort(batch_edge_ids)[0])
    
    mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
    #batch.edge_attr[mask,0].int()
    pred_ids = batch.edge_attr[mask,0].int()
    batch.edge_attr = batch.edge_attr[:, 1:]

    return mask, pred_ids


#inds = self._party.procs_data['train_data']['pred_indices'].detach().cpu()
#batch_edge_inds = inds[batch.input_id]
#batch_edge_ids = train_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]

#mask = torch.isin(batch.edge_attr[:, 0], batch_edge_ids)
#mask.sum()


#inds = self._party.procs_data['train_data']['pred_indices'].detach().cpu()
#batch_edge_inds = inds[batch.input_id.detach().cpu()]
#batch_edge_ids = train_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]

#mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch.input_id.detach().cpu())
#sum(mask)


# select loader for sampling edges ---------------------------------------------------------------------------------------------------------



#train_loader = LinkNeighborLoader(tr_data, num_neighbors=num_neighbors, 
#                                    edge_label_index = tr_data.edge_index,
#                                    edge_label = tr_data.y, 
#                                    batch_size=8192, shuffle=True, transform=None)

#eval_loader = LinkNeighborLoader(ev_data, num_neighbors=num_neighbors, 
#                                    edge_label_index=ev_data.edge_index[:, ev_pred_indices],
#                                    edge_label=ev_data.y[ev_pred_indices],
#                                    batch_size=8192, shuffle=False, transform=None)


def get_loaders(train_data, eval_data, eval_indices, num_neighbors, transform = None):

    train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors, 
                                          edge_label_index = train_data.edge_index,
                                          edge_label = train_data.y, 
                                          batch_size=8192, shuffle=True, transform=transform)


    eval_loader = LinkNeighborLoader(eval_data, num_neighbors=num_neighbors, 
                                        edge_label_index=eval_data.edge_index[:, eval_indices],
                                        edge_label=eval_data.y[eval_indices],
                                        batch_size=8192, shuffle=False, transform=transform)
    
    return train_loader, eval_loader


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



