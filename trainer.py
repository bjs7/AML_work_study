import os
import joblib
from datetime import date
import process_data_type as pdt
import trainer_utils as tu
import torch
from IPython import display



def train_model(args, data, configs, bank = None):

    #model_type = model_types.get(args.model)
    model_type = tu.model_types.get(args.model)

    # Filter the data
    data_processor = tu.data_functions.get(model_type)
    data_for_indices = data['regular_data']['train_data']['x'][['From Bank', 'To Bank']]
    data = data[tu.data_types.get(model_type)]['train_data']

    # get class used for training
    trainer_class = tu.trainer_classes.get(model_type)
    
    # process data
    data_for_indices = pdt.get_bank_indices(data_for_indices, bank) if bank else data_for_indices.index.tolist()
    train_data = data_processor(data, data_for_indices, args)
    
    # train the model
    trainer = trainer_class(args, train_data)
    trained_model = trainer.train()

    # save the model
    #file_name = f'bank_{bank}' if bank else 'full_info'
    file_name = f'bank_{bank}' if bank else args.scenario
    save_direc = save_model(trained_model, file_name, args, configs)

    return save_direc


def save_model(trained_model, file_name, args, configs):

    # create folders to store the model(s)
    save_direc = configs.save_direc_training
    save_direc = os.path.join(save_direc, args.model)

    args.split_perc = configs.split_perc
    str_folder = f'split_{args.split_perc[0]}_{args.split_perc[1]}-'
    models_configs = tu.get_model_configs(args)

    if args.model == 'GINe':
        num_neighbors = models_configs['params']['num_neighbors']
        mask_indexing = models_configs['model_settings']['index_masking']
        str_folder += f'num_neighbors_{num_neighbors}-mask_indexing_{mask_indexing}'
    
    elif args.model == 'xgboost':
        #models_configs = tu.get_model_configs(args)
        num_rounds = models_configs['params']['num_rounds']
        str_folder += f'num_rounds_{num_rounds}'

    save_direc = os.path.join(save_direc, str_folder)
    if not os.path.exists(save_direc):
        os.makedirs(save_direc, exist_ok=True)
    
    # save the model
    model_type = tu.model_types.get(args.model)
    file_type = tu.file_types.get(model_type)
    file_name = os.path.join(save_direc, file_name + f'.{file_type}')
    if model_type == 'graph':
        torch.save(trained_model.state_dict(), file_name)
    elif model_type == 'booster':
        scaler = trained_model.scaler
        joblib.dump({"model": trained_model, "scaler": scaler}, file_name) if scaler is not None else trained_model.save_model(file_name)

    return save_direc








"""

def train_model(model, data, banks = [], **kwargs):

    trainer_class = trainer_functions.get(model.__name__)
    trained_models = {}

    data_type = data_types.get(model.__name__)
    #data_type = 'graph_data'
    #data_type = 'regular_data'
    data_processor = data_functions.get(data_type)

    data_for_indices = data['regular_data']['train_data']['x']
    unfil_data = data[data_type]['train_data']

    save_direc = "/home/nam_07/AML_work_study/models"
    file_type = file_types.get(model.__name__)

    #bank = 1
    if banks:
        for bank in banks:
            print(f'Currently training bank {bank}')

            # get indices for the given data
            bank_indices = get_bank_indices(data_for_indices, bank)

            # filter and process the data for the given bank
            train_data = data_processor(unfil_data, bank_indices, single_bank=True)

            # train model
            #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data)
            #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data, params=params, num_rounds=num_rounds)
            trainer = trainer_class(model, train_data, **kwargs) if model.__name__ == 'GINe' else trainer_class(model(), train_data, **kwargs)
            trained_models[bank] = trainer.train()
            if model.__name__ == 'GINe':
                save_model(trained_models[bank], save_direc + f'\\{model.__name__}_bank_{bank}.{file_type}')
            else:
                save_model(trained_models[bank].model, save_direc + f'\\{model.__name__}_bank_{bank}.{file_type}', trained_models[bank].scaler)

    else:
        # no filtering, just process the data
        train_data = data_processor(unfil_data, data_for_indices.index.tolist())

        # train model
        #trainer = trainer_class(model, train_data) if model.__name__ == 'GINe' else trainer_class(model(), train_data)
        trainer = trainer_class(model, train_data, **kwargs) if model.__name__ == 'GINe' else trainer_class(model(),train_data, **kwargs)
        trained_models['all_banks'] = trainer.train()

        if model.__name__ == 'GINe':
            save_model(trained_models['all_banks'], save_direc + f'\\{model.__name__}_all_banks.{file_type}')
        else:
            save_model(trained_models['all_banks'].model, save_direc + f'\\{model.__name__}_all_banks.{file_type}', trained_models['all_banks'].scaler)

    return trained_models

"""
