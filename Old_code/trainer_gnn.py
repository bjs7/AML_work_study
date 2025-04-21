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

import evaluation as eva
from sklearn.metrics import f1_score

def train_gnn(args, train_data, model_configs, **kwargs):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_indices = train_data['pred_indices']
    train_data = train_data['df']

    ###################################################################
    ##### Might be come relevant to make some changes here later! #####
    ###################################################################

    #m_param = tu.get_model_configs(args).get('params')
    #m_settings = tu.get_model_configs(args).get('model_settings')

    m_param = model_configs.get('params')
    m_settings = model_configs.get('model_settings')

    #m_param['num_neighbors'] *= m_param['gnn_layers']

    # need to make dynamic here?
    batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    

    # loader
    #transform = partial(account_for_time, main_data=train_data)
    train_loader = LinkNeighborLoader(train_data, num_neighbors=m_param.get('num_neighbors'), batch_size=batch_size, shuffle=True, transform=None)
    sample_batch = next(iter(train_loader))

    # continue from here
    model = tgu.get_model(sample_batch, m_param, m_settings)

    # setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_param.get('learning rate'))
    sample_batch.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([m_param.get('w_ce1'), m_param.get('w_ce2')]).to(device))
    model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, m_settings)

    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, **kwargs)
    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device)

    #return {'model': model, 'encoder_pay': encoder_pay, 'encoder_cur': encoder_cur}
    return model


def train_homo(train_loader, train_data, train_indices, vali_loader, vali_data, pred_indices, model, optimizer, loss_fn, device, m_settings, args):
    
    # training
    epochs = configs.epochs
    best_val_f1 = -1

    for epoch in range(epochs):

        print(f'Epoch number {epoch+1}')

        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            if m_settings['index_masking']:
                
                #inds = train_indices
                #batch_edge_inds = inds[batch.input_id]
                #batch_edge_inds = batch.input_id
                #batch_edge_ids = tr_loader.data.edge_attr[batch_edge_inds, 0]
                #mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

                #batch_edge_inds = inds[batch.input_id]
                #batch_edge_ids = tr_loader.data.edge_attr[batch_edge_inds, 0]
                #batch = sample_batch
                
                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu().to(torch.int), batch.input_id)
                # remove the unique edge id from the edge features, as it's no longer neededd
                #batch.edge_attr = batch.edge_attr[:, 1:]
                batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = True)
                pred = out[mask]
                ground_truth = batch.y[mask]
            else:
                batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = False)
                ground_truth = train_data.y[batch.input_id]

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        #f1 = f1_score(ground_truth, pred)
        
        # evaluate model
        current_f1 = eval_func(model, vali_loader, vali_data, pred_indices, args, device, m_settings)
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model = model

    return best_model, best_val_f1



def eval_func(model, loader, data, inds, args, device, m_settings):

    args.tqdm = True
    args.data = 'Small_J'

    preds = []
    ground_truths = []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu().to(torch.int), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        #batch.edge_attr = batch.edge_attr[:, 1:]
        batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1