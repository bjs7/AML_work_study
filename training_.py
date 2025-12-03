import torch
import tqdm
from sklearn.metrics import f1_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import logging

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data):
    #training
    best_val_f1 = 0
    epochs = 15 if args['data_parser'].testing else config['epochs']

    for epoch in range(epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []

        for batch in tqdm.tqdm(tr_loader, disable=False):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device)

        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args['gnn_parser'].model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config['n_gnn_layers'], n_classes=2,
                n_hidden=round(config['n_hidden']), residual=False, edge_updates=args['gnn_parser'].emlps, edge_dim=e_dim, 
                dropout=config['dropout'], final_dropout=config['final_dropout']
                )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyperparameters =  {"lr": 0.006213266113989207, "n_hidden": 66.00315515631006, 
                        "n_mlp_layers": 1, "n_gnn_layers": 2, "loss": "ce", 
                        "w_ce1": 1.0000182882773443, "w_ce2": 6.275014431494497, 
                        "norm_method": "z_normalize", "dropout": 0.00983468338330501, 
                        "final_dropout": 0.10527690625126304}

    #define a model config dictionary and wandb logging at the same time
    config={
    "epochs": 100,
    "batch_size": 8192,
    "model": 'gin',
    "data": 'small',
    "num_neighbors": [100, 100],
    "lr": hyperparameters.get('lr'),
    "n_hidden": hyperparameters.get('n_hidden'),
    "n_gnn_layers": hyperparameters.get('n_gnn_layers'),
    "loss": "ce",
    "w_ce1": hyperparameters.get('w_ce1'),
    "w_ce2": hyperparameters.get('w_ce2'),
    "dropout": hyperparameters.get('dropout'),
    "final_dropout": hyperparameters.get('final_dropout'),
    "n_heads": None
    }

    #set the transform if ego ids should be used

    transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args['gnn_parser'].reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config['w_ce1'], config['w_ce2']]).to(device))

    model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data)
    