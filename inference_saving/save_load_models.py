from pathlib import Path
from abc import ABC, abstractmethod
import json
import os
import configs
import re
import pandas as pd
import torch
import utils
import configs.configs as config


def save_model(model, hyper_params, args, folder_name, file_name, q):

    # create folders to store the model(s)
    # hyper_params = tuned_hyperparameters
    save_direc = config.save_direc_training
    save_direc = os.path.join(save_direc, args.model)
    save_direc = os.path.join(save_direc, f'{args.size}_' + args.ir)

    args.split_perc = config.split_perc[0:2]
    str_folder = f'split_{args.split_perc[0]}_{args.split_perc[1]}__'

    m_settings = utils.get_tuning_configs(args) #.get('model_settings')
    if args.model == 'GINe':
        #args.model = 'GINe'
        m_settings['model_settings']['emlps'] = args.emlps
        mask_indexing, transforming = m_settings.get('model_settings').get('index_masking'), m_settings.get('model_settings').get('transforming_of_time')
        str_folder += f'EU_{args.emlps}__' + f'transforming_of_time_{transforming}__' + f'mask_indexing_{mask_indexing}'
 
    elif args.model == 'xgboost' or args.model == 'light_gbm':
        #args.model = 'xgboost'
        x_0_fi, r_0_fi = m_settings.get('full_info').get(args.size).get('x_0'), m_settings.get('full_info').get(args.size).get('r_0')
        x_0_in, r_0_in = m_settings.get('individual_banks').get(args.size).get('x_0'), m_settings.get('individual_banks').get(args.size).get('r_0')
        str_folder += f'full_info_x0_{x_0_fi}_r0_{r_0_fi}' + f'__individual_x0_{x_0_in}_r0_{r_0_in}'

    save_direc = os.path.join(save_direc, str_folder)
    if args.scenario == 'full_info':
        folder_path = Path(save_direc)
        file_path = folder_path / 'model_settings.json'
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(m_settings, indent=4))

    save_direc = os.path.join(save_direc, folder_name)

    if not os.path.exists(save_direc):
        os.makedirs(save_direc, exist_ok=True)
    
    # save the model
    model_type = utils.model_types.get(args.model)
    file_type = utils.file_types.get(model_type)
    file_name = os.path.join(save_direc, file_name + f'.{file_type}')
    if model_type == 'graph':
        #torch.save(model.state_dict(), file_name)
        torch.save(model, file_name)
    elif model_type == 'booster':
        #scaler = model.scaler
        #joblib.dump({"model": model, "scaler": scaler}, file_name) if scaler is not None else model.save_model(file_name)
        model.save_model(file_name)
    if q == 1:
        file_path = os.path.join(save_direc, 'hyper_parameters.json')
        with open(file_path, 'w') as file:
            json.dump(hyper_params, file, indent=4)



def save_configs(args, save_direc):
    
    # prep to save arguments
    args_dict = {'arguments': vars(args), 'model_configs': utils.get_model_configs(args)}
    folder_path = Path(save_direc)
    file_path = folder_path / 'configurations.json'

    if not os.path.exists(file_path):
        print(file_path)
        # ensure folder exists and save the file
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(args_dict, indent=4))
    




"""

    if not os.path.isfile(save_direc + '/m_settings.json'):
        with open(save_direc + '/m_settings.json', 'w') as file:
            json.dump(m_settings, file, indent=4)


"""