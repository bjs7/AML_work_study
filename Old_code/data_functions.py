import numpy as np
import process_data_type as pdt
import trainer_utils as tu
import pandas as pd
import torch
import copy


def get_graph_data(data, args, bank_indices = None):

    model_type = tu.model_types.get(args.model)
    df = data[tu.data_types.get(model_type)]
    #data_processor = tu.data_functions.get(model_type)

    if model_type == 'graph':
        if args.scenario == 'individual_banks':
            train_data, vali_data, test_data = pdt.update_data(df['test_data']['df'], bank_indices, args)
        else:
            train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']        
        
        return train_data, vali_data, test_data



def get_train_vali_test_data(args, data, bank_indices = None, tuning = True, r_0 = 1):


    # setup
    
    #bank_indices = copy.copy(all_bank_indices)
    model_type = tu.model_types.get(args.model)
    df = data[tu.data_types.get(model_type)]
    data_processor = tu.data_functions.get(model_type)


    #data_by_type = data[tu.data_types.get(model_type)]['test_data']

    # STILL UNSURE IF I NEED TO MAKE A COPY OF THE DATA THAT I AM PROCESSING! AS VALIDAION MIGHT NEED NOT TO BE CHANGED?
    # Nok det bedste.... men foerst tjek om approach for xgboost kan splifies

    if model_type == 'graph':

        if args.scenario == 'individual_banks':
            #bank_indices = copy.copy(bank_indices)
            train_data, vali_data, test_data = pdt.update_data(df['test_data']['df'], bank_indices, args)
            #train_data, vali_data, test_data = update_data(df1['test_data']['df'], bank_indices, args)
        else:
            #updated_train_indices, updated_vali_indices, updated_test_indices, bank_indices1 = pdt.get_updated_bank_indices(bank_indices)
            #train_data, vali_data, test_data = {'df': df1['train_data'], 'pred_indices': torch.tensor(updated_train_indices)}, {'df': df1['vali_data'], 'pred_indices': torch.tensor(updated_vali_indices)}, {'df': df1['test_data'], 'pred_indices': torch.tensor(updated_test_indices)}

            # STILL NEED TO ADJUST TO THE RIGHT AMOUNT OF NODES IN THE SPLITS!
            train_data, vali_data, test_data = df['train_data'], df['vali_data'], df['test_data']

        if tuning:
            # data for training
            #train_data = df['train_data']
            #train_data = data_processor(train_data, bank_indices, args)
            data_processor(train_data)

            # data for validation
            #vali_data = df['vali_data']
            #vali_data = data_processor(vali_data, bank_indices, args, mode = 'validation', scaler_encoders = train_data.get('scaler_encoders'))
            data_processor(vali_data, scaler_encoders = train_data.get('scaler_encoders'))

            return {'train_data': train_data, 'vali_data': vali_data}
        
        else:

            # data for training
            train_data = df['vali_data']
            train_data = data_processor(train_data, bank_indices, args, train_plus_vali = True)

            # data for testing
            test_data = df['test_data']
            test_data = data_processor(test_data, bank_indices, args, mode = 'test', scaler_encoders = train_data.get('scaler_encoders'))

            return {'train_data': train_data, 'vali_data': test_data}
    
    elif model_type == 'booster':
        
        train_indices = bank_indices['train_data_indices']
        vali_indices = bank_indices['vali_data_indices']
        test_indices = bank_indices['test_data_indices']
        
        if tuning:
            if r_0 < 1: 
                train_indices = train_indices[:round(len(train_indices) * r_0)]

            train_data = df['train_data']
            train_data = data_processor(train_data, train_indices, args)

            vali_data = df['vali_data']
            vali_data = data_processor(vali_data, vali_indices, args, scaler_encoders = train_data.get('scaler_encoders'))
            
            return {'train_data': train_data, 'vali_data': vali_data}
        
        else:

            train_indices = bank_indices['train_data_indices'] + bank_indices['vali_data_indices']
            test_indices = bank_indices['test_data_indices']

            train_data = {
                'x': pd.concat([df['train_data']['x'], 
                                df['vali_data']['x']]) ,
                'y': np.concatenate([df['train_data']['y'], 
                                     df['vali_data']['y']])
            }
            
            train_data = data_processor(train_data, train_indices, args)

            test_data = df['test_data']
            test_data = data_processor(test_data, test_indices, args, scaler_encoders = train_data.get('scaler_encoders'))

    
            return {'train_data': train_data, 'vali_data': test_data}
        


"""
def get_train_vali_test_data(args, data, bank_indices, tuning = True, r_0 = 1):

    # setup
    model_type = tu.model_types.get(args.model)
    data_processor = tu.data_functions.get(model_type)

    if model_type == 'graph':

        if tuning:
            # data for training
            train_data = data[tu.data_types.get(model_type)]['train_data']
            test123 = data[tu.data_types.get(model_type)]['train_data']
            train_data = data_processor(train_data, bank_indices, args)
            t1 = data_processor(test123, bank_indices, args)

            # data for validation
            vali_data = data[tu.data_types.get(model_type)]['vali_data']
            test1234 = data[tu.data_types.get(model_type)]['vali_data']
            vali_data = data_processor(vali_data, bank_indices, args, mode = 'validation', scaler_encoders = train_data.get('scaler_encoders'))
            t2 = data_processor(test1234, bank_indices, args, mode = 'validation', scaler_encoders = test1234.get('scaler_encoders'))

            test12345 = data[tu.data_types.get(model_type)]['test_data']
            data_processor(test12345, bank_indices, args, mode = 'test')

            return {'train_data': train_data, 'vali_data': vali_data}
        
        else:

            # data for training
            train_data = data[tu.data_types.get(model_type)]['vali_data']
            train_data = data_processor(train_data, bank_indices, args, train_plus_vali = True)

            # data for testing
            test_data = data[tu.data_types.get(model_type)]['test_data']
            test_data = data_processor(test_data, bank_indices, args, mode = 'test', scaler_encoders = train_data.get('scaler_encoders'))

            return {'train_data': train_data, 'vali_data': test_data}
    
    elif model_type == 'booster':
        
        train_indices = bank_indices['train_data_indices']
        vali_indices = bank_indices['vali_data_indices']
        test_indices = bank_indices['test_data_indices']
        
        if tuning:
            if r_0 < 1: 
                train_indices = train_indices[:round(len(train_indices) * r_0)]

            train_data = data[tu.data_types.get(model_type)]['train_data']
            train_data = data_processor(train_data, train_indices, args)

            vali_data = data[tu.data_types.get(model_type)]['vali_data']
            vali_data = data_processor(vali_data, vali_indices, args, scaler_encoders = train_data.get('scaler_encoders'))
            
            return {'train_data': train_data, 'vali_data': vali_data}
        
        else:

            train_indices = bank_indices['train_data_indices'] + bank_indices['vali_data_indices']
            test_indices = bank_indices['test_data_indices']

            train_data = {
                'x': pd.concat([data[tu.data_types.get(model_type)]['train_data']['x'], 
                                data[tu.data_types.get(model_type)]['vali_data']['x']]) ,
                'y': np.concatenate([data[tu.data_types.get(model_type)]['train_data']['y'], 
                                     data[tu.data_types.get(model_type)]['vali_data']['y']])
            }
            
            train_data = data_processor(train_data, train_indices, args)

            test_data = data[tu.data_types.get(model_type)]['test_data']
            test_data = data_processor(test_data, test_indices, args, scaler_encoders = train_data.get('scaler_encoders'))

    
            return {'train_data': train_data, 'vali_data': test_data}


"""
