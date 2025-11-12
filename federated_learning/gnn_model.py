
# packages for FL

from abc import ABC, abstractmethod
import federated_learning.FL_utils as fl_utils
import torch
from federated_learning.registry import GNN_REGISTRY



# potentially split this up, so that it is in two, like one where parameters are extracted once
# and then used to get the model in the register.
# need to sort how to init about parties and whether they need batching or not, and
# also how to adjust the data set size, such that copy can be avoided.

# maybe move get_gnn out of GNN, and into manager 100%, and then send to parties?
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

        arguments = {'num_features': node_features, 'num_gnn_layers': hyperparams.get('gnn_layers'),
                    'n_classes': 2, 'n_hidden': hyperparams.get('hidden_embedding_size'),
                    'residual': False, 'edge_updates': manager.args['gnn_parser'].emlps, 
                    'edge_dim': edge_dim, 'dropout': hyperparams.get('dropout'), 
                    'final_dropout': hyperparams.get('dropout')}
        
        return gnn_init(**arguments)

    def _init_model_configs(self):
        return 0

    def _get_gnn_loss_optimizer(self, manager, hyperparams, node_features, edge_dim):
        self.gnn = self._create_gnn_model(manager, hyperparams, node_features, edge_dim)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=hyperparams.get('learning rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparams.get('w_ce1'), hyperparams.get('w_ce2')])) 

    def update_w(self, gd_data):

        self.gnn.train()
        self.optimizer.zero_grad()

        pred = self.gnn(gd_data.x, gd_data.edge_index, gd_data.edge_attr, 
                            gd_data.edge_index, gd_data.edge_attr)
        loss = self.loss_fn(pred, gd_data.y)

        loss.backward()
        self.optimizer.step()
    
    def predict_binary(self, graph_data):

        gd_data = graph_data.get('gd_data')
        pred_indices = graph_data.get('pred_indices')

        self.gnn.eval()
        with torch.no_grad():
            #data.to(device)
            pred = self.gnn(gd_data.x, gd_data.edge_index, gd_data.edge_attr, 
                    gd_data.edge_index, gd_data.edge_attr)
            pred = pred[pred_indices] if pred_indices is not None else pred
            
        return pred.argmax(dim=-1)
