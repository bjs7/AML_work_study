import copy
import numpy as np
import tqdm
import torch
from torch_geometric.nn.models.metapath2vec import sample
#from model import edge_index
import trainer_gnn_utils as tgu
import configs
import evaluation as eval
from sklearn.metrics import f1_score
import tuning_utils as tut


#model_configs = tut.hyper_sampler(args)
def gnn_trainer(args, train_data, vali_data, model_configs, seed = None):

    args.tqdm = True
    args.data = 'Small_J'

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # need data here
    
    train_indices = train_data['pred_indices']
    train_data = train_data['df']

    pred_indices = vali_data['pred_indices']
    vali_data = vali_data['df']  

    # parameters
    m_param = model_configs.get('params')
    m_settings = model_configs.get('model_settings')
    
    # might wanna change the batch_size
    batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]

    # loaders

    if seed:
        utils.set_seed(seed)
    # The loader has an argument called time_attr which could potentially be used to adjust for time?
    train_loader, vali_loader = tgu.get_loaders(train_data, vali_data, pred_indices, m_param, batch_size)
    sample_batch = next(iter(train_loader))
    #batch = next(iter(vali_loader))
    
    # from here we switch between train and validation

    # train
    model = tgu.get_model(sample_batch, m_param, m_settings, args)

    # setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_param.get('learning rate'))
    sample_batch.to(device)
    #batch = sample_batch

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([m_param.get('w_ce1'), m_param.get('w_ce2')]).to(device))
    model, f1 = train_homo(train_loader, train_data, train_indices, vali_loader, vali_data, pred_indices, model, optimizer, loss_fn, device, m_settings, args)

    return model, f1


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
            target_edge_attr = train_data.edge_attr[batch.input_id, :]

            if m_settings['index_masking']:
                #batch = sample_batch
                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu().to(torch.int), batch.input_id)
                # remove the unique edge id from the edge features, as it's no longer neededd
                #batch.edge_attr = batch.edge_attr[:, 1:]
                batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
                target_edge_attr = target_edge_attr[:, 1:] if m_settings['include_time'] else target_edge_attr[:, 2:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr, index_mask = True)
                pred = out[mask]
                ground_truth = batch.y[mask]
            else:
                batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
                target_edge_attr = target_edge_attr[:, 1:] if m_settings['include_time'] else target_edge_attr[:, 2:]
                batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr, index_mask = False)
                ground_truth = batch.edge_label
                #ground_truth = train_data.y[batch.input_id]

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
        current_f1 = eval.eval_func(model, vali_loader, vali_data, pred_indices, args, device, m_settings)
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model = model

    return best_model, best_val_f1
