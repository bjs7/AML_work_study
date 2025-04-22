import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging


class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                 n_hidden=66, edge_updates=True, residual=True,
                 edge_dim=None, dropout=0.0, final_dropout=0.5):
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
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden)
            ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25),
                                 nn.ReLU(), nn.Dropout(self.final_dropout),
                                 Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr, edge_label_index, target_edge_attr, index_mask=True):
        src, dst = edge_index
        target_src, target_dst = edge_label_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        target_edge_attr = self.edge_emb(target_edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                if index_mask:
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                else:
                    target_edge_attr = target_edge_attr + self.emlps[i](torch.cat([x[target_src], x[target_dst], target_edge_attr], dim=-1)) / 2

        if index_mask:
            x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
            x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
            out = x
        else:
            x = x[edge_label_index.T].reshape(-1, 2 * self.n_hidden).relu()
            x = torch.cat((x, target_edge_attr.view(-1, target_edge_attr.shape[1])), 1)
            out = x

        return self.mlp(out)

#seed_src, seed_dst = edge_label_index
#seed_emb_1 = x[seed_src]
#seed_emb_2 = x[seed_dst]
#out = torch.cat([seed_emb_1, seed_emb_2, torch.abs(seed_emb_1 - seed_emb_2)], dim=-1)
