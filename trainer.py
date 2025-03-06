import xgboost as xgb
import joblib

trainer_functions = {
    "simple_nn_full": simple_nn_trainer,
    "Booster": xgboost_trainer,
    'GINe': train_gnn_trainer
}

data_types = {
    'GINe': 'graph_data',
    'Booster': 'regular_data'
}

data_functions = {
    'graph_data': process_graph_data,
    'regular_data': process_regular_data,
}

file_types = {
    'GINe': 'pth',
    'Booster': 'pkl'
}

def train_model(model, data, banks = [], **kwargs):

    trainer_class = trainer_functions.get(model.__name__)
    trained_models = {}

    data_type = data_types.get(model.__name__)
    #data_type = 'graph_data'
    #data_type = 'regular_data'
    data_processor = data_functions.get(data_type)

    data_for_indices = data['regular_data']['train_data']['x']
    unfil_data = data[data_type]['train_data']

    save_direc = 'C:\\Users\\u0168001\\OneDrive - KU Leuven\\Desktop\\Courses\\AML_work_study\\pycharm\\models'
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
