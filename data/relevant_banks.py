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


    individual_banks = sorted(list(set(laundering_in_train_data) & set(laundering_in_vali_data)))
    FedAvg_banks = sorted(list(laundering_in_train_data))

    # --------------------------------------------------

    # Get number of observations available for individual banks
    all_indices_individual = []
    for bank in individual_banks:
        bank_indices = get_indices_bdt(df, bank = bank)
        all_indices_individual.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])
    obs_available_for_individual_banks = len(set(all_indices_individual))


    all_indices_FedAvg = []
    for bank in FedAvg_banks:
        bank_indices = get_indices_bdt(df, bank = bank)
        all_indices_FedAvg.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])
    obs_available_for_FedAvg_banks = len(set(all_indices_FedAvg))

    relevant_banks = {
        'total_observations': df_length,
        'individual': {'banks': individual_banks, 'obs_ava': obs_available_for_individual_banks, 'percentage': obs_available_for_individual_banks/df_length},
        'FedAvg': {'banks': FedAvg_banks, 'obs_ava': obs_available_for_FedAvg_banks, 'percentage': obs_available_for_FedAvg_banks/df_length}
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
    relevant_banks = load_relevant_banks(parsers).get(parsers['fl_parser'].fl_algo)
    return relevant_banks.get('banks')


class BanksManager:

    def __init__(self, parsers):
        self.parsers = parsers

    def load_create_file(self):
        save_direc = config.save_direc_training
        save_direc = os.path.join(save_direc, 'relevant_banks')
        
        str_folder = f"{self.parsers['data_parser'].size}_{self.parsers['data_parser'].ir}__split_{config.split_perc[0:2][0]}_{config.split_perc[0:2][1]}.json"
        file_location = os.path.join(save_direc, str_folder)

        if not os.path.isfile(file_location) or self.parsers['data_parser'].overwrite:
            self.relevant_banks = filter_banks(self.parsers)
        else:
            self.relevant_banks = load_relevant_banks(self.parsers)

    def print_numbers(self):

        for scenario in ['individual', 'FedAvg']:
            info = self.relevant_banks.get(scenario)
            if info:
                num_banks = len(info.get('banks', []))
                obs = info.get('obs_ava', 0)
                perc = info.get('percentage', 0)
                print(f'{scenario}: {num_banks} banks, {obs} observations ({perc:.2%} of data)')




def main():

    parsers = utils.parser_all()

    manager = BanksManager(parsers)

    manager.load_create_file()

    manager.print_numbers()
    

if __name__ == "__main__":
    main()

