import argparse
import data_processing as dp
import pandas as pd
import utils
import process_data_type as pdt
import trainer as tra
import configs
from pathlib import Path
import json
import save_load_models as slm

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    #parser.add_argument('--scenario', default='full info', type=str, help='Select the scenario to study')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    #parser.add_argument('--data_split', default=[0.75, 0.25], type=utils.parse_data_split)

    return parser



def main():

    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    if args.banks:
        args.scenario = 'individual banks'

    #args.split_perc = configs.split_perc
    #args_dict = {'arguments': vars(args), 'model_configs': tu.get_model_configs(args)}
    
    #configs.save_direc_training
    #json.dumps(vars(args), indent=4)

    #config_str = json.dumps(args_dict, sort_keys=True)
    #config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    # I need to save configs etc. after models have been trained...

    
    # load data
    df = pd.read_csv("/home/nam_07/AML_work_study/formatted_transactions.csv")
    data = dp.get_data(df, split_perc = configs.split_perc)

    #model = 'GINe'
    #args.model = 'xgboost'
    
    # train the model and get args, configs etc. incase changes, and save direction of save files
    #args, configs, save_direc = tra.train_model(args, data, configs)

    if args.scenario == 'individual banks':
        for index, bank in enumerate(args.banks):
            save_direc = tra.train_model(args, data, configs, bank = bank)
    else:
        save_direc = tra.train_model(args, data, configs)
    
    slm.save_configs(args, save_direc)




if __name__ == '__main__':
    main()







