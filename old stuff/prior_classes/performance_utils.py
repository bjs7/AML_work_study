import tuning_utils as tune_u
import trainer_utils as tu
import json
import os
import data_functions as data_funcs
import pandas as pd
import numpy as np
import trainer_gnn_utils as tgu
import xgboost as xgb
import torch
from sklearn.metrics import f1_score


def get_folder_direction(args, split = [0.6, 0.2]):

    m_settings = tune_u.get_tuning_configs(args)

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


def get_indi_para(main_folder, bank = None):
    folder = f'bank_{bank}' if isinstance(bank, int) else 'full_info'
    tmp_folder = os.path.join(main_folder, folder)
    if os.path.exists(tmp_folder):    
        with open(os.path.join(tmp_folder, 'hyper_parameters.json'), 'r') as file:
            model_parameters = json.load(file)
    else:
        model_parameters = None
    return tmp_folder, model_parameters


def get_indices_feat_data(args, raw_data, bank = None):

    bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
    train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)

    model_type = tu.model_types.get(args.model)

    if model_type == 'graph':
        train_data, test_data = data_funcs.general_feature_engineering('graph', vali_data, test_data)
    elif model_type == 'booster':
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







