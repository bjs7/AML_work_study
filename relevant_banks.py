import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_functions as data_funcs
import tuning as tune
import train_models as tr_models
import save_load_models as slm
import numpy as np
import os
from pathlib import Path
import json

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xgboost', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='individual_banks', type=str, help='Select the scenario to study')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")

    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    #parser.add_argument('--graph_tuning_x_0', default = 10, type=int, help='Amount of models to train on for tuning GNN')
    parser.add_argument('--seed', default=0, type=int, help="Set seed for reproducability")
    
    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--specific_banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    
    return parser


def filter_banks(args):

    #parser = get_parser()
    #args = parser.parse_args()

    df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    raw_data = dp.get_data(df, split_perc = configs.split_perc)

    Banks = list((df.loc[:, 'From Bank'])) + list((df.loc[:, 'To Bank']))
    unique_banks = list(set(Banks))
    args.banks = unique_banks
    args.scenario = 'individual_banks'

    no_train_data = []
    no_vali_data = []
    no_test_data = []
    laundering_in_test_data = []
    laundering_in_vali_data = []
    laundering_in_train_data = []
    first_round_banks = []

    for bank in args.banks:

        if bank % 1000 == 0: print(bank)
        bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
        train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)

        if np.any(train_data['y'] == 1):
            laundering_in_train_data.append(bank)

        if np.any(vali_data['y'] == 1):
            laundering_in_vali_data.append(bank)

        if np.any(test_data['y'] == 1):
            laundering_in_test_data.append(bank)
        
        if not bank_indices['train_indices'] or not bank_indices['vali_indices'] or not bank_indices['test_indices']:
            # there are no banks with no data in the trainin split
            if not bank_indices['train_indices']:
                no_train_data.append(bank)
            # There are 28720 banks with no data in the validation split
            if not bank_indices['vali_indices']:
                no_vali_data.append(bank)
            # there are 14224 banks with no data in the testing split
            if not bank_indices['test_indices']:
                no_test_data.append(bank)
        else:
            first_round_banks.append(bank)


    fr_banks = list(set(unique_banks) - set(no_train_data) - set(no_vali_data) - set(no_test_data))
    sr_banks = list(set(no_train_data + no_vali_data) - set(no_test_data))

    # first round banks

    fr_laundering_in_train = set(fr_banks) & set(laundering_in_train_data)
    fr_laundering_in_vali = set(fr_banks) & set(laundering_in_vali_data)
    fr_laundering_in_test = set(fr_banks) & set(laundering_in_test_data)
    fr_laundering_in_train_test = sorted(list(fr_laundering_in_train & fr_laundering_in_test))
    #fr_laund_in_train_vali_test = sorted(list(set(list(fr_laundering_in_train) + list(fr_laundering_in_vali)) & fr_laundering_in_test))
    fr_laund_in_train_vali = sorted(list(set(fr_banks) & set(laundering_in_train_data + laundering_in_vali_data)))
    pass_to_sr = (set(list(fr_laundering_in_train) + list(fr_laundering_in_vali)) & fr_laundering_in_test) - (fr_laundering_in_train & fr_laundering_in_test)

    #
    fr_laundering_in_train_no_test = fr_laundering_in_train - fr_laundering_in_test
    fr_laundering_in_test_no_train = fr_laundering_in_test - fr_laundering_in_train

    # second round banks
    sr_laundering_in_train = set(sr_banks) & set(laundering_in_train_data)
    sr_laundering_in_vali = set(sr_banks) & set(laundering_in_vali_data)
    sr_laundering_in_test = set(sr_banks) & set(laundering_in_test_data)
    sr_laundering_in_train_test = sorted(list(sr_laundering_in_train & sr_laundering_in_test))
    sr_laundering_in_train_test = sorted(set(sr_laundering_in_train_test + sorted(pass_to_sr)))
    #sr_laund_in_train_vali_test = sorted(list(set(list(sr_laundering_in_train) + list(sr_laundering_in_vali)) & sr_laundering_in_test))
    sr_laund_in_train_vali = sorted(list(set(sr_banks) & set(laundering_in_train_data + laundering_in_vali_data)))

    #
    sr_laundering_in_train_no_test = sr_laundering_in_train - sr_laundering_in_test
    sr_laundering_in_test_no_train = sr_laundering_in_test - sr_laundering_in_train

    # Get number of observations available for banks with train and test data available
    banks = fr_banks + sr_banks
    all_indices_fr_sr = []
    for bank in banks:
        bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
        tmp_indices = bank_indices.get('train_indices') + bank_indices.get('vali_indices') + bank_indices.get('test_indices')
        all_indices_fr_sr += tmp_indices
    obs_ava_fr_sr = len(set(all_indices_fr_sr))

    # Get number of observations available for banks with only laundering
    #banks = fr_laund_in_train_vali + sr_laund_in_train_vali
    banks = fr_laundering_in_train_test + sr_laundering_in_train_test
    all_indices = []
    for bank in banks:
        bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)
        tmp_indices = bank_indices.get('train_indices') + bank_indices.get('vali_indices') + bank_indices.get('test_indices')
        all_indices += tmp_indices
    obs_ava = len(set(all_indices))


    relevant_banks = {
        'total_observations': df.shape[0],
        'full_first_second': {'fr_banks': fr_banks, 'sr_banks': sr_banks, 'obs_ava': obs_ava_fr_sr, 'percentage': obs_ava_fr_sr/df.shape[0]},
        'only_launderings': {'fr_banks': fr_laundering_in_train_test, 
                                        'sr_banks': sr_laundering_in_train_test, 'obs_ava': obs_ava, 'percentage': obs_ava/df.shape[0]},
        #'only_launderings': {'fr_banks': fr_laund_in_train_vali, 
        #                                'sr_banks': sr_laund_in_train_vali, 'obs_ava': obs_ava, 'percentage': obs_ava/df.shape[0]},
        'extras': {'no_data': {'no_train_data': no_train_data, 
                               'no_vali_data': no_vali_data, 
                               'no_test_data': no_test_data},

                    'first_round': {'fr_laundering_in_train': list(fr_laundering_in_train), 
                                    'fr_laundering_in_test': list(fr_laundering_in_test), 
                                    'fr_laundering_in_train_no_test': list(fr_laundering_in_train_no_test),
                                    'fr_laundering_in_test_no_train': list(fr_laundering_in_test_no_train)},
                
                    'second_round': {'sr_laundering_in_train': list(sr_laundering_in_train),
                                    'sr_laundering_in_test': list(sr_laundering_in_test),
                                    'sr_laundering_in_train_no_test': list(sr_laundering_in_train_no_test),
                                    'sr_laundering_in_test_no_train': list(sr_laundering_in_test_no_train)}  
                }
    }   

    # save the data
    save_direc = configs.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    folder_path = Path(save_direc)

    #save_direc = os.path.join(save_direc, f'{args.size}_' + args.ir)
    args.split_perc = configs.split_perc[0:2]
    str_folder = f'{args.size}_' + args.ir + f'__split_{args.split_perc[0]}_{args.split_perc[1]}.json'
    file_path = folder_path / str_folder
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(relevant_banks, indent=4))

    return relevant_banks


def load_relevant_banks(args):

    save_direc = configs.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    
    split = configs.split_perc[0:2]
    str_folder = f'{args.size}_' + args.ir + f'__split_{split[0]}_{split[1]}.json'
    file_location = os.path.join(save_direc, str_folder)

    with open(file_location, 'r') as file:
        relevant_banks = json.load(file)

    return relevant_banks

def get_relevant_banks(args):
    relevant_banks = load_relevant_banks(args).get(args.banks)
    return relevant_banks.get('fr_banks'), relevant_banks.get('sr_banks')


class BanksManager:

    def __init__(self, args):
        self.args = args

    def load_create_file(self):
        save_direc = configs.save_direc_training
        save_direc = os.path.join(save_direc, 'relevant_banks')
        
        split = configs.split_perc[0:2]
        str_folder = f'{self.args.size}_' + self.args.ir + f'__split_{split[0]}_{split[1]}.json'
        file_location = os.path.join(save_direc, str_folder)

        if not os.path.isfile(file_location) or self.args.overwrite:
            self.relvant_banks = filter_banks(self.args)
        else:
            self.relevant_banks = load_relevant_banks(self.args)

    def print_numbers(self):
        fr_banks = len(self.relevant_banks.get('full_first_second').get('fr_banks'))
        sr_banks = len(self.relevant_banks.get('full_first_second').get('sr_banks'))
        obs = self.relevant_banks.get('full_first_second').get('obs_ava')
        perc = self.relevant_banks.get('full_first_second').get('percentage')
        print(f'In the case of training on all possible banks: \nFirst round: {fr_banks} \nSecond round: {sr_banks}')
        print(f'Resulting in a total of {obs} observations, {perc} percentage of the data\n')

        fr_banks = len(self.relevant_banks.get('only_launderings').get('fr_banks'))
        sr_banks = len(self.relevant_banks.get('only_launderings').get('sr_banks'))
        obs = self.relevant_banks.get('only_launderings').get('obs_ava')
        perc = self.relevant_banks.get('only_launderings').get('percentage')
        print(f'In case of training on banks with laundering there are \nFirst round: {fr_banks} \nSecond round: {sr_banks}')
        print(f'Resulting in a total of {obs} observations, {perc} percentage of the data\n')

        no_train_data = len(self.relevant_banks.get('extras').get('no_data').get('no_train_data'))
        no_vali_data = len(self.relevant_banks.get('extras').get('no_data').get('no_vali_data'))
        no_test_data = len(self.relevant_banks.get('extras').get('no_data').get('no_test_data'))

        print(f'Banks with no data in: \nTrain split: {no_train_data} \nVali split: {no_vali_data} \nTest split: {no_test_data}\n')



def main():

    parser = get_parser()
    args = parser.parse_args()

    manager = BanksManager(args)

    manager.load_create_file()

    manager.print_numbers()
    

if __name__ == "__main__":
    main()

