from sklearn.metrics import f1_score
import trainer_gnn_utils as tgu
import torch
import tqdm
import copy

#model_configs = utils.hyper_sampler(args)
def eval_func(model, loader, data, inds, args, device, m_settings):

    args.tqdm = True
    args.data = 'Small_J'

    preds = []
    ground_truths = []
    #inds = pred_indices
    #loader = vali_loader
    #batch = next(iter(loader))

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]

        if m_settings.get('index_masking'):
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

        target_edge_attr = data.edge_attr[batch_edge_inds, :].to(device)
        batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
        target_edge_attr = target_edge_attr[:, 1:] if m_settings['include_time'] else target_edge_attr[:, 2:]
        
        with torch.no_grad():
            batch.to(device)
            if m_settings.get('index_masking'):
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr, index_mask = True)
                out = out[mask]
                ground_truth = batch.y[mask]
            else:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr, index_mask = False)
                ground_truth = batch.edge_label
                
            pred = out.argmax(dim=-1)
            preds.append(pred)
            #ground_truths.append(batch.y[mask])
            ground_truths.append(ground_truth)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1

    
"""

        batch.edge_label_index
        batch.edge_index
        data.edge_attr[batch.input_id, :][:,0]
        data.edge_attr[batch.input_id, :][:,18]
        train_data.edge_attr[batch.input_id, 20][0:10]
        train_data.edge_attr[batch.input_id, 18][50:55]
        vali_data.edge_attr[batch_edge_inds, 18][50:55]
        data.edge_attr[batch_edge_inds, :][:,0]


        train_data.edge_attr[batch.input_id, 0][0:10]
        vali_data.edge_attr[batch_edge_inds, 0][0:10]

"""

