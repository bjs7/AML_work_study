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
    df_raw = pd.read_csv(utils.get_data_path() + f"/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
    df_length = df_raw.shape[0]
    unique_banks = sorted(pd.concat([df_raw['From Bank'], df_raw['To Bank']]).unique().tolist())
    df, scaler_encoders = get_data(df_raw, parsers['data_parser'], split_perc = split_perc)

    # set lists ----------------------------------------------------------------------------------------
    laundering_in_test_data, laundering_in_vali_data, laundering_in_train_data = [], [], []
    has_data_in_vali, has_data_in_test = [], []
    vali_min, test_min = [], []
    vali_min_2, test_min_2 = [], []
    laundering_in_train_data_1, laundering_in_vali_data_1 = [], []

    for bank in unique_banks:

        if bank % 1000 == 0: print(bank)

        bank_indices = get_indices_bdt(df, bank = bank)
        train_data, vali_data, test_data = get_booster_data(parsers['data_parser'], df['regular_data'], bank_indices)

        # check for laundering in each split
        if np.any(train_data['x']['Is Laundering'] == 1) and np.any(train_data['x']['Is Laundering'] == 0):
            laundering_in_train_data.append(bank)
        if np.any(vali_data['x']['Is Laundering'] == 1) and np.any(vali_data['x']['Is Laundering'] == 0):
            laundering_in_vali_data.append(bank)
        if np.any(test_data['x']['Is Laundering'] == 1):
            laundering_in_test_data.append(bank)

        if np.any(train_data['x']['Is Laundering'] == 1):
            laundering_in_train_data_1.append(bank)
        if np.any(vali_data['x']['Is Laundering'] == 1):
            laundering_in_vali_data_1.append(bank)

        # check for any data in vali/test splits
        if bank_indices['vali_indices']:
            has_data_in_vali.append(bank)
        if bank_indices['test_indices']:
            has_data_in_test.append(bank)
        if len(bank_indices['vali_indices']) < 5:
            vali_min.append(bank)
        if len(bank_indices['test_indices']) < 5:
            test_min.append(bank)

    #laundering_in_vali_data_1 == laundering_in_vali_data
    #laundering_in_train_data_1 == laundering_in_train_data
    individual_banks = sorted(list(set(laundering_in_train_data) & set(laundering_in_vali_data) & set(has_data_in_test)))

    # FedAvg: three exclusive groups for add_banks_to_manager
    # Train: both labels required (to learn). Vali/Test: any data (to evaluate with global model).
    fedavg_train_banks = sorted(laundering_in_train_data)
    fedavg_vali_banks = sorted(list(set(has_data_in_vali) - set(fedavg_train_banks)))
    fedavg_test_banks = sorted(list(set(has_data_in_test) - set(fedavg_train_banks) - set(fedavg_vali_banks)))
    
    # --------------------------------------------------

    # Get number of observations available for individual banks
    all_indices_individual = []
    for bank in individual_banks:
        bank_indices = get_indices_bdt(df, bank = bank)
        all_indices_individual.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])
    obs_available_for_individual_banks = len(set(all_indices_individual))

    all_indices_FedAvg = []
    all_fedavg_banks = fedavg_train_banks + fedavg_vali_banks + fedavg_test_banks
    for bank in all_fedavg_banks:
        bank_indices = get_indices_bdt(df, bank = bank)
        all_indices_FedAvg.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])
    obs_available_for_FedAvg_banks = len(set(all_indices_FedAvg))

    alternative_FedAvg = []
    for bank in fedavg_train_banks:
        bank_indices = get_indices_bdt(df, bank = bank)
        alternative_FedAvg.extend(bank_indices['train_indices'] + bank_indices['vali_indices'] + bank_indices['test_indices'])
    obs_available_alternative_FedAvg = len(set(alternative_FedAvg))


    # Laundering counts per split (total)
    split_data = df['regular_data']
    total_laundering = int(df_raw['Is Laundering'].sum())
    train_laundering = int(split_data['train_data']['x']['Is Laundering'].sum())
    vali_laundering = int(split_data['vali_data']['x']['Is Laundering'].sum())
    test_laundering = int(split_data['test_data']['x']['Is Laundering'].sum())

    # Per-split stats for each scenario's banks
    split_totals = {'train': train_laundering, 'vali': vali_laundering, 'test': test_laundering}

    def count_split_stats(bank_indices):
        bank_idx = set(bank_indices)
        counts = {}
        data_pct = {}
        laundering_pct = {}
        for split in ['train', 'vali', 'test']:
            split_df = split_data[f'{split}_data']['x']
            overlap = split_df.index.isin(bank_idx)
            counts[f'{split}_laundering'] = int(split_df.loc[overlap, 'Is Laundering'].sum())
            data_pct[f'{split}_data_pct'] = float(np.mean(overlap))
            laundering_pct[f'{split}_laundering_pct'] = counts[f'{split}_laundering'] / split_totals[split] if split_totals[split] else 0
        laundering_pct['total_laundering_pct'] = sum(counts.values()) / total_laundering if total_laundering else 0
        return counts, data_pct, laundering_pct

    individual_laundering, individual_data_pct, individual_laundering_pct = count_split_stats(all_indices_individual)
    fedavg_laundering, fedavg_data_pct, fedavg_laundering_pct = count_split_stats(all_indices_FedAvg)
    fedavg_alt_laundering, fedavg_alt_data_pct, fedavg_alt_laundering_pct = count_split_stats(alternative_FedAvg)

    relevant_banks = {
        'total_observations': df_length, 'total_laundering': total_laundering,
        'train_laundering': train_laundering, 'vali_laundering': vali_laundering, 'test_laundering': test_laundering,
        'individual': {'banks': individual_banks, 'obs_ava': obs_available_for_individual_banks,
                       'percentage': obs_available_for_individual_banks/df_length,
                       **individual_laundering, **individual_data_pct, **individual_laundering_pct},
        'FedAvg': {'train_banks': fedavg_train_banks, 'vali_banks': has_data_in_vali, 'test_banks': has_data_in_test,
                   'obs_ava': obs_available_for_FedAvg_banks,
                   'percentage': obs_available_for_FedAvg_banks/df_length,
                   **fedavg_laundering, **fedavg_data_pct, **fedavg_laundering_pct},
        'FedAvg_alt': {'banks': fedavg_train_banks,
                       'obs_ava': obs_available_alternative_FedAvg,
                       'percentage': obs_available_alternative_FedAvg/df_length,
                       **fedavg_alt_laundering, **fedavg_alt_data_pct, **fedavg_alt_laundering_pct}
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


def apply_bank_filter(train_banks, df, bank_filter):
    """Filter training bank list by edge count based on bank_filter setting.

    Only filters training banks - vali/test banks are kept to evaluate on all banks.

    Args:
        train_banks: List of training bank IDs
        df: Processed data dict containing regular_data with 'From Bank'/'To Bank' columns
        bank_filter: One of 'no_top10', 'no_top1', 'no_bottom10', 'no_bottom5pct'

    Returns:
        Filtered train_banks
    """
    # Count edges per bank in training split (edge counted once even if bank is both sender and receiver)
    train_data = df['regular_data']['train_data']['x'][['From Bank', 'To Bank']]
    edge_counts = pd.Series({
        bank: ((train_data['From Bank'] == bank) | (train_data['To Bank'] == bank)).sum()
        for bank in train_banks
    })

    if bank_filter == 'no_top10':
        banks_to_remove = set(edge_counts.nlargest(10).index)
    elif bank_filter == 'no_top1':
        banks_to_remove = set(edge_counts.nlargest(1).index)
    elif bank_filter == 'no_bottom10':
        banks_to_remove = set(edge_counts.nsmallest(10).index)
    elif bank_filter == 'no_bottom5pct':
        threshold = np.percentile(edge_counts, 5)
        banks_to_remove = set(edge_counts[edge_counts <= threshold].index)

    return [b for b in train_banks if b not in banks_to_remove]


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

