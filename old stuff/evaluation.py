from sklearn.metrics import f1_score
import training.gnn_utils as tgu
import torch
import tqdm
import copy
import numpy as np
from abc import ABC, abstractmethod


import copy
#import numpy as np
import tqdm
import torch
from torch_geometric.nn.models.metapath2vec import sample
#from model import edge_index
import training.gnn_utils as tgu
import configs.configs as config
import evaluation as eval
from sklearn.metrics import f1_score
import training.hyperparams as tune_u
import warnings
import utils

from data.feature_engi import general_feature_engineering



#model_configs = utils.hyper_sampler(args)
def eval_func(model, loader, data, inds, args, device, m_settings):

    args.data = 'Small_J'

    preds = []
    ground_truths = []
    #inds = pred_indices
    #loader = vali_loader
    #batch = next(iter(loader))
    model.eval()

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
    f1 = f1_score(ground_truth, pred, average='binary', zero_division=0)

    return f1


def eval_func_no_batching(model, data, inds, args, device, m_settings):

    data = copy.deepcopy(data)
    args.data = 'Small_J'
    data.edge_attr = data.edge_attr[:, 1:] if m_settings['include_time'] else data.edge_attr[:, 2:]

    model.eval()
    with torch.no_grad():
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.edge_index, data.edge_attr, index_mask = m_settings['include_time'])
        out = out[inds]
        pred = out.argmax(dim=-1)
    
    preds = pred.cpu().numpy()
    ground_truth = data.y[inds].cpu().numpy()
    f1 = f1_score(ground_truth, preds, average='binary', zero_division=0)

    return f1

    




# --------------------------------------------------------------------------------

def gnn_trainer(args, train_data, vali_data, model_configs, seed = None):

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
    #m_settings = model_configs.get('model_settings')
    m_settings = utils.get_tuning_configs(args).get('model_settings')
    
    # might wanna change the batch_size
    #batch_size = m_param.get('batch_size')[0] if train_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    batch_size = m_param.get('batch_size')

    # loaders
    if seed:
        utils.set_seed(seed)

    if batch_size > 1:
        # The loader has an argument called time_attr which could potentially be used to adjust for time?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_loader, vali_loader = tgu.get_loaders(train_data, vali_data, pred_indices, m_param, batch_size)
        sample_batch = next(iter(train_loader))


    # from here we switch between train and validation

    # train
    model = tgu.get_model(train_data, m_param, m_settings, args)

    # setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_param.get('learning rate'))
    #sample_batch.to(device)
    #batch = sample_batch

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([m_param.get('w_ce1'), m_param.get('w_ce2')]).to(device))
    model, f1 = train_homo(train_loader, train_data, train_indices, vali_loader, vali_data, pred_indices, model, optimizer, 
                           loss_fn, device, m_settings, args) if batch_size > 1 else train_homo_no_batching(train_data, train_indices, vali_data, pred_indices, 
                                                                                                            model, optimizer, loss_fn, device, m_settings, args)

    return model, f1


def train_homo(train_loader, train_data, train_indices, vali_loader, vali_data, pred_indices, model, optimizer, loss_fn, device, m_settings, args):
    
    # training
    epochs = config.epochs
    best_val_f1 = -1

    for epoch in range(epochs):

        #set the model to training mode
        model.train()

        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(train_loader, disable=not args.tqdm):
            
            optimizer.zero_grad()
            target_edge_attr = train_data.edge_attr[batch.input_id, :].to(device)

            if m_settings['index_masking']:
                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu().to(torch.int), batch.input_id)
                # remove the unique edge id from the edge features, as it's no longer neededd
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
            #best_model = model
            best_model = copy.deepcopy(model.state_dict())

    return best_model, best_val_f1


# function for running full batch

def train_homo_no_batching(train_data, train_indices, vali_data, pred_indices, model, optimizer, loss_fn, device, m_settings, args):
    
    # training
    epochs = config.epochs
    best_val_f1 = -1

    train_data = copy.deepcopy(train_data)
    train_data.edge_attr = train_data.edge_attr[:, 1:] if m_settings['include_time'] else train_data.edge_attr[:, 2:]
    train_data.to(device)

    for epoch in range(epochs):

        #set the model to training mode
        model.train()

        # training step
        optimizer.zero_grad()
        pred = model(train_data.x, train_data.edge_index, train_data.edge_attr, train_data.edge_index, train_data.edge_attr, index_mask = m_settings['index_masking'])
        loss = loss_fn(pred, train_data.y)

        loss.backward()
        optimizer.step()

        # evaluate model
        current_f1 = eval.eval_func_no_batching(model, vali_data, pred_indices, args, device, m_settings)
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model = copy.deepcopy(model.state_dict())

    return best_model, best_val_f1


# old performance utils


def get_folder_direction(args, split = config.split_perc):

    m_settings = utils.get_tuning_configs(args)

    if args.model == 'GINe':
        m_settings['model_settings']['emlps'] = args.emlps
        mask_indexing, transforming = m_settings.get('model_settings').get('index_masking'), m_settings.get('model_settings').get('transforming_of_time')
        path = f'/home/nam_07/AML_work_study/models/{args.model}/{args.size}_{args.ir}/split_{split[0]}_{split[1]}__EU_{args.emlps}__transforming_of_time_{transforming}__mask_indexing_{mask_indexing}'
 
    elif args.model == 'xgboost':
        x_0_fi, r_0_fi = m_settings.get('full_info').get(args.size).get('x_0'), m_settings.get('full_info').get(args.size).get('r_0')
        x_0_in, r_0_in = m_settings.get('individual_banks').get(args.size).get('x_0'), m_settings.get('individual_banks').get(args.size).get('r_0')
        path = f'/home/nam_07/AML_work_study/models/{args.model}/{args.size}_{args.ir}/split_{split[0]}_{split[1]}__full_info_x0_{x_0_fi}_r0_{r_0_fi}__individual_x0_{x_0_in}_r0_{r_0_in}'

    return path

def get_main_folder(args):
    main_folder = get_folder_direction(args)
    with open(os.path.join(main_folder, 'model_settings.json'), 'r') as file:
        model_settings = json.load(file)
    return main_folder, model_settings



def get_indices_feat_data(model, raw_data, bank = None):

    #bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
    #train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
    bank_indices = model.get_indices(raw_data, bank=bank)
    train_data, vali_data, test_data = model.get_data(raw_data, bank_indices=bank_indices)

    #model_type = tu.model_types.get(mol)

    if model.args.model_type == 'graph':
        train_data, test_data = data_funcs.general_feature_engineering('graph', vali_data, test_data)
    elif model.args.model_type == 'booster':
        train_data = {
            'x': pd.concat([train_data['x'], vali_data['x']]).reset_index(drop=True),
            'y': np.concatenate([train_data['y'], vali_data['y']])
        }
        train_data, test_data = data_funcs.general_feature_engineering('booster', train_data, test_data)
        test_data['pred_indices'] = bank_indices['test_indices']

    return test_data, bank_indices['test_indices']


def get_model(args, test_data, model_parameters, model_settings, tmp_folder):

    if args.model == 'GINe':
        model = tgu.get_model(test_data['df'], model_parameters['params'], model_settings['model_settings'], args)
    elif args.model == 'xgboost':
        model = xgb.Booster()
        model.load_model(os.path.join(tmp_folder, 'seed_1.ubj'))
    
    return model


def make_predictions(model_type, model, test_data, model_settings, tmp_folder):

    models_predictions = {}
    if model_type == 'graph':

        #if not edge_dim_adjusted:
        test_data['df'].edge_attr = test_data['df'].edge_attr[:, 1:] if model_settings['model_settings']['include_time'] else test_data['df'].edge_attr[:, 2:]
        true_y = test_data['df'].y[test_data['pred_indices']]
        seeds = sorted([seed for seed in os.listdir(tmp_folder) if '.pth' in seed])

        for seed in seeds:

            model_weights = torch.load(os.path.join(tmp_folder, seed), weights_only=True)
            model.load_state_dict(model_weights)
            model.eval()

            # predictions
            out = model(test_data['df'].x, test_data['df'].edge_index, test_data['df'].edge_attr, 
                    test_data['df'].edge_index, test_data['df'].edge_attr, 
                    index_mask = model_settings['model_settings']['index_masking'])

            out = out[test_data['pred_indices']]
            preds = out.argmax(dim=-1)
            f1 = f1_score(true_y, preds, average='binary')

            models_predictions[seed] = {'preds': preds, 'f1': f1}
    
    elif model_type == 'booster':
        
        preds = model.predict(xgb.DMatrix(test_data['x']))
        preds = (preds >= 0.5).astype(int)
        f1 = f1_score(test_data['y'], preds, average='binary')
        models_predictions['seed_1'] = {'preds': preds, 'f1': f1}

    return models_predictions

def f1_loop(models_predictions):
    max_f1, ave_f1, max_f1_index = -1, [], None
    for key in models_predictions.keys():
        ave_f1.append(models_predictions[key]['f1'])
        if models_predictions[key]['f1'] > max_f1:
            max_f1 = models_predictions[key]['f1']
            max_f1_index = key
    ave_f1 = np.mean(ave_f1)
    return max_f1_index, max_f1, ave_f1

def get_predictions(args, model, test_data, model_settings, tmp_folder, f1_values = None):

    model_type = tu.model_types.get(args.model)
    models_predictions = make_predictions(model_type, model, test_data, model_settings, tmp_folder)

    if model_type == 'graph':
        best_seed, max_f1, ave_f1 = f1_loop(models_predictions)
        f1_values = pd.DataFrame(columns=['bank', 'max_f1', 'ave_f1']) if (f1_values is None) else f1_values
        f1_values.loc[len(f1_values)] = [tmp_folder.split("/")[-1], max_f1, ave_f1]
        predictions = models_predictions[best_seed]['preds']
  
    elif model_type == 'booster':
        f1_values = pd.DataFrame(columns=['bank', 'f1']) if not f1_values else f1_values
        f1_values.loc[len(f1_values)] = [tmp_folder.split("/")[-1], models_predictions['seed_1']['f1']]
        predictions = models_predictions['seed_1']['preds']

    return predictions, f1_values


def make_predictions_graph(model, test_data, model_settings, tmp_folder):
    models_predictions = {}
    
    test_data['df'].edge_attr = test_data['df'].edge_attr[:, 1:] if model_settings['model_settings']['include_time'] else test_data['df'].edge_attr[:, 2:]
    true_y = test_data['df'].y[test_data['pred_indices']]
    seeds = sorted([seed for seed in os.listdir(tmp_folder) if '.pth' in seed])

    for seed in seeds:
        
        model_weights = torch.load(os.path.join(tmp_folder, seed), weights_only=True)
        model.load_state_dict(model_weights)
        model.eval()

        # predictions
        out = model(test_data['df'].x, test_data['df'].edge_index, test_data['df'].edge_attr, 
            test_data['df'].edge_index, test_data['df'].edge_attr, 
            index_mask = model_settings['model_settings']['index_masking'])

        out = out[test_data['pred_indices']]
        preds = out.argmax(dim=-1)
        f1 = f1_score(true_y, preds, average='binary')

        models_predictions[seed] = {'preds': preds, 'f1': f1}

    return models_predictions



