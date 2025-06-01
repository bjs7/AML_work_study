import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import f1_score
import utils
import configs.configs as config
import copy
from torch_geometric.loader import LinkNeighborLoader
import warnings

def get_folder_path_gnn(args, split = config.split_perc):
    m_settings = utils.get_tuning_configs(args)
    m_settings['model_settings']['emlps'] = args.emlps
    mask_indexing, transforming = m_settings.get('model_settings').get('index_masking'), m_settings.get('model_settings').get('transforming_of_time')
    utils.get_data_path()
    main_folder = f'{utils.get_data_path()}/AML_work_study/models/{args.model}/{args.size}_{args.ir}/split_{split[0]}_{split[1]}__EU_{args.emlps}__transforming_of_time_{transforming}__mask_indexing_{mask_indexing}'

    with open(os.path.join(main_folder, 'model_settings.json'), 'r') as file:
        model_settings = json.load(file)

    return main_folder, model_settings

def get_folder_path_booster(args, split = config.split_perc):
    m_settings = utils.get_tuning_configs(args)
    x_0_fi, r_0_fi = m_settings.get('full_info').get(args.size).get('x_0'), m_settings.get('full_info').get(args.size).get('r_0')
    x_0_in, r_0_in = m_settings.get('individual_banks').get(args.size).get('x_0'), m_settings.get('individual_banks').get(args.size).get('r_0')
    main_folder = f'{utils.get_data_path()}/AML_work_study/models/{args.model}/{args.size}_{args.ir}/split_{split[0]}_{split[1]}__full_info_x0_{x_0_fi}_r0_{r_0_fi}__individual_x0_{x_0_in}_r0_{r_0_in}'

    with open(os.path.join(main_folder, 'model_settings.json'), 'r') as file:
        model_settings = json.load(file)

    return main_folder, model_settings


def get_indices_data(infer, raw_data, bank):
    bank_indices = infer.get_indices(raw_data, bank)
    train_data, vali_data, test_data = infer.get_data(raw_data, bank_indices=bank_indices)
    return train_data, vali_data, test_data, bank_indices

def get_test_indices_data_booster(infer, train_data, vali_data, test_data, bank_indices):
    test_indices = bank_indices['test_indices']
    train_data, test_data = infer.pred_data_for_tr_inf(train_data, vali_data, test_data)
    return test_data, test_indices


def get_indi_para(main_folder, bank = None):
    folder = f'bank_{bank}' if isinstance(bank, int) else 'full_info'
    tmp_folder = os.path.join(main_folder, folder)
    if os.path.exists(tmp_folder):    
        with open(os.path.join(tmp_folder, 'hyper_parameters.json'), 'r') as file:
            model_parameters = json.load(file)
    else:
        model_parameters = None
    return tmp_folder, model_parameters

def get_model_booster(tmp_folder):
    model = xgb.Booster()
    model.load_model(os.path.join(tmp_folder, 'seed_1.ubj'))
    return model


class GNNpredictions:

    def __init__(self, test_data, model_settings, model_parameters, tmp_folder):
        
        self.models_predictions = {}
        self.model_settings = model_settings
        self.model_parameters = model_parameters

        self.tmp_folder = tmp_folder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pred_indices = test_data['pred_indices']
        self.test_data = test_data['df']
        self.true_y = test_data['df'].y[test_data['pred_indices']]
        self.seeds = sorted([seed for seed in os.listdir(self.tmp_folder) if '.pth' in seed])

    def make_predictions(self, model):
        
        if self.model_parameters['params']['batch_size'] > 1:
            make_predict = self._predictions_batching
        else:
            make_predict = self._predictions_no_batching
        
        for seed in self.seeds:
            preds, f1 = make_predict(model, seed)
            self.models_predictions[seed] = {'preds': preds, 'f1': f1}

        return self.models_predictions


    def get_predictions(self, model, f1_values = None):
        models_predictions = self.make_predictions(model)
        best_seed, max_f1, ave_f1 = self.f1_loop(models_predictions)
        f1_values = pd.DataFrame(columns=['bank', 'max_f1', 'ave_f1']) if (f1_values is None) else f1_values
        f1_values.loc[len(f1_values)] = [self.tmp_folder.split("/")[-1], max_f1, ave_f1]
        return models_predictions[best_seed]['preds'], f1_values

    def f1_loop(self, models_predictions):
        max_f1, ave_f1, max_f1_index = -1, [], None
        for key in models_predictions.keys():
            ave_f1.append(models_predictions[key]['f1'])
            if models_predictions[key]['f1'] > max_f1:
                max_f1 = models_predictions[key]['f1']
                max_f1_index = key
        ave_f1 = np.mean(ave_f1)
        return max_f1_index, max_f1, ave_f1

    
    def _predictions_batching(self, model, seed):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = LinkNeighborLoader(self.test_data, 
                                    num_neighbors=self.model_parameters['params']['num_neighbors'], 
                                    edge_label_index=self.test_data.edge_index[:, self.pred_indices],
                                    edge_label=self.test_data.y[self.pred_indices],
                                    batch_size=self.model_parameters['params']['batch_size'], shuffle=False, transform=None)

        preds = []
        ground_truths = []
        model.eval()

        model_path = os.path.join(self.tmp_folder, seed)
        model_weights = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(model_weights)
        model.to(self.device)

        for batch in loader:
            
            #select the seed edges from which the batch was created
            self.pred_indices = self.pred_indices.detach().cpu()
            batch_edge_inds = self.pred_indices[batch.input_id.detach().cpu()]

            #
            target_edge_attr = self.test_data.edge_attr[batch_edge_inds, :].to(self.device)
            batch.edge_attr = batch.edge_attr[:, 1:] if self.model_settings['model_settings']['include_time'] else batch.edge_attr[:, 2:]
            target_edge_attr = target_edge_attr[:, 1:] if self.model_settings['model_settings']['include_time'] else target_edge_attr[:, 2:]
            
            with torch.no_grad():
                batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, 
                    batch.edge_label_index, target_edge_attr, 
                    index_mask = self.model_settings['model_settings']['index_masking'])
                #out = out[self.pred_indices]
                pred = out.argmax(dim=-1)
                preds.append(pred)
                ground_truths.append(batch.edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        f1 = f1_score(ground_truth, pred, average='binary', zero_division=0)
        #f1 = f1_score(self.true_y, preds.cpu().numpy(), average='binary')
        #self.models_predictions[seed] = {'preds': preds, 'f1': f1}
        
        return pred, f1


    def _predictions_no_batching(self, model, seed):
        
        test_data = copy.deepcopy(self.test_data)
        test_data.edge_attr = test_data.edge_attr[:, 1:] if self.model_settings['model_settings']['include_time'] else test_data.edge_attr[:, 2:]

        model_path = os.path.join(self.tmp_folder, seed)
        model_weights = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(model_weights)
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            test_data.to(self.device)
            out = model(test_data.x, test_data.edge_index, test_data.edge_attr, 
                test_data.edge_index, test_data.edge_attr, 
                index_mask = self.model_settings['model_settings']['index_masking'])
            out = out[self.pred_indices]
            preds = out.argmax(dim=-1)

        f1 = f1_score(self.true_y, preds.cpu().numpy(), average='binary')

        return preds, f1


def get_preditcions_booster(preds, test_data, tmp_folder, f1_values=None):

    preds = (preds >= 0.5).astype(int)
    f1 = f1_score(test_data['y'], preds, average='binary', zero_division=0)

    f1_values = pd.DataFrame(columns=['bank', 'f1']) if (f1_values is None) else f1_values #not isinstance(f1_values, pd.DataFrame)
    f1_values.loc[len(f1_values)] = [tmp_folder.split("/")[-1], f1]

    return preds, f1_values



