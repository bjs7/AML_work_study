import copy
import numpy as np
import tqdm
import torch
from torch_geometric.nn.models.metapath2vec import sample
#from model import edge_index
from functools import partial
import trainer_utils as tu
from torch_geometric.loader import LinkNeighborLoader
import trainer_gnn_utils as tgu
import configs

def train_gnn(args, train_data, **kwargs):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_indices = train_data['bank_indices']
    train_data = train_data['df']

    ###################################################################
    ##### Might be come relevant to make some changes here later! #####
    ###################################################################

    m_param = tu.get_model_configs(args).get('params')
    m_settings = tu.get_model_configs(args).get('model_settings')

    # need to make dynamic here?
    num_neighbors = m_param.get('num_neighbors')
    batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    nn_size = len(num_neighbors)

    # loader
    #transform = partial(account_for_time, main_data=train_data)
    train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=True, transform=None)
    sample_batch = next(iter(train_loader))
    model = tgu.get_model(sample_batch, nn_size)

    # stepup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_param.get('lr'))
    sample_batch.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([m_param.get('w_ce1'), m_param.get('w_ce2')]).to(device))
    model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, m_settings)

    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, **kwargs)
    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device)

    return model


def train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, m_settings):
    
    # training
    epochs = configs.epochs
    best_val_f1 = 0

    for epoch in range(epochs):

        print(f'Epoch number {epoch+1}')

        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(train_loader):
            #batch = next(iter(train_loader))
            optimizer.zero_grad()

            if m_settings['index_masking']:
                mask = torch.isin(batch.edge_attr[:, 0].to(torch.int), batch.input_id)
                # remove the unique edge id from the edge features, as it's no longer neededd
                batch.edge_attr = batch.edge_attr[:, 1:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = True)
                pred = out[mask]
                ground_truth = batch.y[mask]
            else:
                batch.edge_attr = batch.edge_attr[:, 1:]
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = False)
                ground_truth = train_data.y[batch.input_id]

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    return model

