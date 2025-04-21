from pathlib import Path
from abc import ABC, abstractmethod
import json
import trainer_utils as tu
import os
import configs
import re
import pandas as pd
import torch

def save_model(model, hyper_params, args, folder_name, file_name):

    # create folders to store the model(s)
    save_direc = configs.save_direc_training
    save_direc = os.path.join(save_direc, args.model)

    args.split_perc = configs.split_perc
    str_folder = f'split_{args.split_perc[0]}_{args.split_perc[1]}-'

    if args.model == 'GINe':
        transforming = 'No_transforming_of_time'
        mask_indexing = hyper_params['model_settings']['index_masking']
        str_folder += f'mask_indexing_{mask_indexing}-{transforming}'  

    save_direc = os.path.join(os.path.join(save_direc, str_folder), folder_name)

    if not os.path.exists(save_direc):
        os.makedirs(save_direc, exist_ok=True)
    
    # save the model
    model_type = tu.model_types.get(args.model)
    file_type = tu.file_types.get(model_type)
    file_name = os.path.join(save_direc, file_name + f'.{file_type}')
    if model_type == 'graph':
        torch.save(model.state_dict(), file_name)
    elif model_type == 'booster':
        scaler = model.scaler
        joblib.dump({"model": model, "scaler": scaler}, file_name) if scaler is not None else model.save_model(file_name)


def save_configs(args, save_direc):
    
    # prep to save arguments
    args_dict = {'arguments': vars(args), 'model_configs': tu.get_model_configs(args)}
    folder_path = Path(save_direc)
    file_path = folder_path / 'configurations.json'

    if not os.path.exists(file_path):
        print(file_path)
        # ensure folder exists and save the file
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(args_dict, indent=4))
    




