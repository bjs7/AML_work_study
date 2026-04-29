import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv, LayerNorm
import torch.nn.functional as F
import torch
from federated_learning.registry import register_gnn

#############################################################################################
# IMPORTANT !!!!!!!
# WHEN ADDING NEW MODELS CHECK IF THEY HAVE MORE/DIFFERENT ARGUMENTS/PARAMETERS THAN THE GINE
# IF SO THEN THEY NEED TO BE ADDED TO THE SAMPLING OF SAMPLING HYPERPARAMETERS!!
#############################################################################################
# ALSO NEED TO CHANGE THE ELIF CONDITION TO MODEL.TYPE
# Or whether another type of data (holder) for the graph data is needed

@register_gnn('GINe')
class GINe(torch.nn.Module):

    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                 n_hidden=100, edge_updates=False, residual=True,
                 edge_dim=None, dropout=0.0, final_dropout=0.5, batching=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            if batching:
                self.batch_norms.append(BatchNorm(n_hidden))
            else:
                self.batch_norms.append(LayerNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(25, n_classes))
        self.mlp_vert = nn.Sequential(Linear(n_hidden * 3 * 2, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(25, n_classes))

    def emed_features(self, nodes, edges):
        nodes = self.node_emb(nodes)
        edges = self.edge_emb(edges)
        return {'nodes': nodes, 'edges': edges}
    
    def apply_gnn_layer(self, nodes, edges, edge_index, layer_idx):
        src, dst = edge_index
        nodes = (nodes + F.relu(self.batch_norms[layer_idx](self.convs[layer_idx](nodes, edge_index, edges)))) / 2
        if self.edge_updates:
            edges = edges + self.emlps[layer_idx](torch.cat([nodes[src], nodes[dst], edges], dim=-1)) / 2
        return {'nodes': nodes, 'edges': edges}
    
    def prep_nodes_edges(self, nodes, edges, edge_index):
        nodes = nodes[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        out = torch.cat((nodes, edges.view(-1, edges.shape[1])), 1)
        return out
    
    def final_layer(self, embeddings):
        return self.mlp(embeddings)
    
    def forward(self, nodes, edge_index, edges):

        embeddings = self.emed_features(nodes, edges)
        for i in range(self.num_gnn_layers):
            embeddings = self.apply_gnn_layer(
                embeddings['nodes'],
                embeddings['edges'],
                edge_index,
                i
            )
        embeddings = self.prep_nodes_edges(embeddings['nodes'], embeddings['edges'], edge_index)
        output = self.final_layer(embeddings)

        return output






class GATe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                 n_hidden=100, n_heads=4, edge_updates=False,
                 edge_dim=None, dropout=0.0, final_dropout=0.5, batching=True):
        super().__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            if batching:
                self.batch_norms.append(BatchNorm(n_hidden))
            else:
                self.batch_norms.append(LayerNorm(n_hidden))
                
        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
            
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x

        return self.mlp(out)






