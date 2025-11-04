import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging


bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = 2)
len(bank_indices['train_data_indices'])
len(bank_indices['vali_data_indices'])
len(bank_indices['test_data_indices'])


            
args.scenario = 'individual_banks'
if bank % 1000 == 0: print(bank)
bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = 2)
#len(bank_indices['train_data_indices'])
#len(bank_indices['vali_data_indices'])
train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
train_data

if train_data['df'].num_nodes >= 5000:
    banks_plus_5k.append(bank)
elif train_data['df'].num_nodes >= 4000:
    banks_plus_4k.append(bank)
elif train_data['df'].num_nodes >= 3000:
    banks_plus_3k.append(bank)
elif train_data['df'].num_nodes >= 2000:
    banks_plus_2k.append(bank)
elif train_data['df'].num_nodes >= 1000:
    banks_plus_1k.append(bank)
else:
    banks_less_1k.append(bank)


len(banks_plus_5k) # 12 #[0, 2, 4, 12, 50, 68, 541, 1677, 1698, 2273, 9872, 11273]
len(banks_plus_4k) # 11
len(banks_plus_3k) # 23
len(banks_plus_2k) # 74
len(banks_plus_1k) # 300
len(banks_less_1k) # 30050


[0, 2, 4, 12, 50, 68, 541, 1677, 1698, 2273, 9872, 11273]
bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = 0)
train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
train_data
test123 = hyper_sampler(args, 0e3)
test123.get('params').get('batch_size')[0]
# max 40k, else around 10k


banks_less_100k = []
banks_plus_100k = []
banks_plus_200k = []
banks_plus_300k = []
banks_plus_400k = []
banks_plus_500k = []

banks_less_1k = []
banks_plus_1k = []
banks_plus_2k = []
banks_plus_3k = []
banks_plus_4k = []
banks_plus_5k = []

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                 n_hidden=66, edge_updates=True, residual=True,
                 edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        n_hidden = 70
        num_gnn_layers = 3
        edge_updates = True
        final_dropout = 0

        node_emb = nn.Linear(1, n_hidden)
        edge_emb = nn.Linear(19, n_hidden)

        convs = nn.ModuleList()
        emlps = nn.ModuleList()
        batch_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden)
            ), edge_dim=n_hidden)
            if edge_updates: emlps.append(nn.Sequential(
                nn.Linear(3 * n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
            ))
            convs.append(conv)
            batch_norms.append(BatchNorm(n_hidden))

        mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(final_dropout), Linear(50, 25),
                                 nn.ReLU(), nn.Dropout(final_dropout),
                                 Linear(25, 2))

    def forward(self, x, edge_index, edge_attr, edge_label_index, target_edge_attr, index_mask=True):


        #target_edge_attr = train_data.edge_attr[batch.input_id, :]
        target_edge_attr = train_data.edge_attr
        target_edge_attr = target_edge_attr[:, 1:] if m_settings['include_time'] else target_edge_attr[:, 2:]

        edge_index = train_data.edge_index
        src, dst = edge_index
        target_src, target_dst = train_data.edge_index
        edge_label_index = train_data.edge_index

        x = node_emb(train_data.x)
        edge_attr = edge_emb(train_data.edge_attr)
        target_edge_attr = edge_emb(target_edge_attr)
        edge_index = train_data.edge_index

        for i in range(num_gnn_layers):
            x = (x + F.relu(batch_norms[i](convs[i](x, edge_index, edge_attr)))) / 2
            if edge_updates:
                edge_attr = edge_attr + emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                target_edge_attr = target_edge_attr + emlps[i](torch.cat([x[target_src], x[target_dst], target_edge_attr], dim=-1)) / 2


            sorted(edge_attr[:,18])[0:10]
            sorted(target_edge_attr[:,18])[0:10]

            sorted(edge_attr[:,69])[0:10]
            sorted(target_edge_attr[:,69])[0:10]

            sorted(edge_attr[:,0])[0:10]
            sorted(target_edge_attr[:,0])[0:10]

            sorted(x[mask][:,209])[0:10]
            sorted(x1[:,209])[0:10]


            
            sorted(batch.edge_attr[mask][:,18])[0:10]
            sorted(target_edge_attr[mask1][:,18])[0:10]

            sorted(edge_attr[mask][:,69])[0:10]
            sorted(target_edge_attr[mask1][:,69])[0:10]

            sorted(edge_attr[mask][:,0])[0:10]
            sorted(target_edge_attr[mask1][:,0])[0:10]

            sorted(x[mask][:,209])[0:10]
            sorted(x1[mask1][:,209])[0:10]


        if index_mask:
            x1 = x[edge_label_index.T].reshape(-1, 2 * n_hidden).relu()
            x1 = torch.cat((x1, target_edge_attr.view(-1, target_edge_attr.shape[1])), 1)
            x = x[edge_index.T].reshape(-1, 2 * n_hidden).relu()
            x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
            out = x
        else:
            seed_src, seed_dst = batch.edge_label_index
            seed_emb_1 = x[seed_src]
            seed_emb_2 = x[seed_dst]
            seed_emb_2.shape
            out = torch.cat([seed_emb_1, seed_emb_2, torch.abs(seed_emb_1 - seed_emb_2)], dim=-1)

        
        
        x[mask][3][0:10]
        x1[3][0:10]

        mlp(x[mask])
        mlp(x1[mask1])

        return mlp(out)
