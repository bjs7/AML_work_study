import os
import utils
import pandas as pd
import configs.configs as config
from data.raw_data_processing import get_data
import numpy as np
from pathlib import Path
import json
from configs.configs import split_perc
from data.get_indices_type_data import get_bank_indices
from data.get_indices_type_data import get_indices_bdt, get_booster_data, get_graph_data
import data.data_functions as dfn


def filter_banks(parsers):

    # set up ----------------------------------------------------------------------------------------
    #parsers = utils.parser_all()
    parsers['data_parser'].ibm_fe = True
    parsers['data_parser'].scenario = 'individual_banks'

    # data ----------------------------------------------------------------------------------------
    df = pd.read_csv(utils.get_data_path() + f"/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
    df_length = df.shape[0]
    unique_banks = sorted(pd.concat([df['From Bank'], df['To Bank']]).unique().tolist())
    df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)

    # set lists ----------------------------------------------------------------------------------------
    laundering_in_test_data, laundering_in_vali_data, laundering_in_train_data = [], [], []

    for bank in unique_banks:

        if bank % 1000 == 0: print(bank)

        bank_indices = get_indices_bdt(df, bank = bank)
        train_data, vali_data, test_data = get_booster_data(parsers['data_parser'], df['regular_data'], bank_indices)

        # check for laundering in each split
        if np.any(train_data['x']['Is Laundering'] == 1):
            laundering_in_train_data.append(bank)
        if np.any(vali_data['x']['Is Laundering'] == 1):
            laundering_in_vali_data.append(bank)
        if np.any(test_data['x']['Is Laundering'] == 1):
            laundering_in_test_data.append(bank)

    fr_laundering_banks = sorted(list(set(laundering_in_train_data) & set(laundering_in_vali_data) & set(laundering_in_test_data)))
    sr_laundering_banks = sorted(list((set(laundering_in_train_data) | set(laundering_in_vali_data)) & set(laundering_in_test_data) - set(fr_laundering_banks)))

    # Get number of observations available for banks with only laundering
    all_indices = []
    for bank in (fr_laundering_banks + sr_laundering_banks):
        bank_indices = get_indices_bdt(df, bank = bank)
        all_indices.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])

    obs_available_for_laundering_banks = len(set(all_indices))

    relevant_banks = {
        'total_observations': df_length,
        #'full_first_second': {'fr_banks': fr_banks, 'sr_banks': sr_banks, 'obs_ava': obs_ava_fr_sr, 'percentage': obs_ava_fr_sr/df.shape[0]},
        'only_launderings': {'fr_banks': fr_laundering_banks, 
                                        'sr_banks': sr_laundering_banks, 'obs_ava': obs_available_for_laundering_banks, 'percentage': obs_available_for_laundering_banks/df_length},
    }

    # save the data
    save_direc = config.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    folder_path = Path(save_direc)

    str_folder = f"{parsers['data_parser'].size}_{parsers['data_parser'].ir}__split_{config.split_perc[0:2][0]}_{config.split_perc[0:2][1]}.json"
    file_path = folder_path / str_folder
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(relevant_banks, indent=4))

    return relevant_banks


def load_relevant_banks(parsers):

    save_direc = config.save_direc_training
    save_direc = os.path.join(save_direc, 'relevant_banks')
    str_folder = f"{parsers['data_parser'].size}_{parsers['data_parser'].ir}__split_{config.split_perc[0:2][0]}_{config.split_perc[0:2][1]}.json"
    
    file_location = os.path.join(save_direc, str_folder)

    with open(file_location, 'r') as file:
        relevant_banks = json.load(file)

    return relevant_banks

def get_relevant_banks(parsers):
    relevant_banks = load_relevant_banks(parsers).get(parsers['data_parser'].banks)
    return relevant_banks.get('fr_banks'), relevant_banks.get('sr_banks')


class BanksManager:

    def __init__(self, parsers):
        self.parsers = parsers

    def load_create_file(self):
        save_direc = config.save_direc_training
        save_direc = os.path.join(save_direc, 'relevant_banks')
        
        str_folder = f"{self.parsers['data_parser'].size}_{self.parsers['data_parser'].ir}__split_{config.split_perc[0:2][0]}_{config.split_perc[0:2][1]}.json"
        file_location = os.path.join(save_direc, str_folder)

        if not os.path.isfile(file_location) or self.parsers['data_parser'].overwrite:
            self.relvant_banks = filter_banks(self.parsers)
        else:
            self.relevant_banks = load_relevant_banks(self.parsers)

    def print_numbers(self):

        fr_banks = len(self.relevant_banks.get('only_launderings').get('fr_banks'))
        sr_banks = len(self.relevant_banks.get('only_launderings').get('sr_banks'))
        obs = self.relevant_banks.get('only_launderings').get('obs_ava')
        perc = self.relevant_banks.get('only_launderings').get('percentage')
        print(f'In case of training on banks with laundering there are \nFirst round: {fr_banks} \nSecond round: {sr_banks}')
        print(f'Resulting in a total of {obs} observations, {perc} percentage of the data\n')




def main():

    parsers = utils.parser_all()

    manager = BanksManager(parsers)

    manager.load_create_file()

    manager.print_numbers()
    

if __name__ == "__main__":
    main()

