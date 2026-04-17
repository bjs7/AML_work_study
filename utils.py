import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random
import pandas as pd
import copy

from data.get_indices_type_data import get_indices_bdt
from federated_learning.fl_base import Manager, Party
import data.data_functions as dfn
from data.raw_data_processing import get_data
from configs.configs import split_perc
from configs.paths import get_data_path, get_tuning_configs

# --------------------------------------------------------------------------------------------------
# dictionary holders

model_types = {
    'GINe': 'gnn',
    'GINe_vert': 'gnn',
    'xgboost': 'booster',
    'light_gbm': 'booster',
    'regression': 'regression'
}

data_types = {
    'gnn': 'graph_data',
    'booster': 'regular_data',
    'regression': 'regular_data'
}

file_types = {
    'gnn': 'pth',
    'booster': 'ubj'
}

# --------------------------------------------------------------------------------------------------
# util functions


def setup_get_data():

    parsers = parser_all()
    set_seed(parsers['data_parser'].seed, True)

    df = pd.read_csv(f"{get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

    if parsers['data_parser'].testing:
        # Stratified sample: preserves class ratio (0s and 1s) across the
        # full dataset before the temporal split, then re-sort to keep temporal
        # ordering so the 60/20/20 split remains meaningful.
        frac = parsers['data_parser'].testing_frac
        df = pd.concat([
            df[df['Is Laundering'] == 0].sample(frac=frac, random_state=parsers['data_parser'].seed),
            df[df['Is Laundering'] == 1].sample(frac=frac, random_state=parsers['data_parser'].seed),
        ]).sort_index().reset_index(drop=True)

    df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc=split_perc)

    return parsers, df, scaler_encoders


def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(filename)-20s:%(lineno)-4d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )


def get_cpu_count() -> int:
    """Return available CPUs. On SLURM uses SLURM_CPUS_PER_TASK if set,
    so we don't oversubscribe beyond the allocated job resources."""
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        return int(slurm_cpus)
    return os.cpu_count() or 1


def set_seed(seed: int = 0, log = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if log:
        logging.info(f"Random seed set as {seed}")


def get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim):

    if parsers['data_parser'].data_type == 'graph_data':
        bank_dim = train_data['df']['x'].shape[0]
    else:
        bank_dim = train_data['x'].shape[0]

    if bank_dim < smallest_dim:
        smallest_bank = bank
        smallest_dim = bank_dim

    return smallest_bank, smallest_dim

# --------------------------------------------------------------------------------------------------
# parser function for all the parsers

def parser_all():

    parsers = [
        ("fl_parser", fl_parser()),
        ("data_parser", data_parser()),
        ("gnn_parser", gnn_parser())
    ]

    remaining_args = sys.argv[1:]
    all_parsers = {}

    for name, parser in parsers:
        try:
            parsed, remaining_args = parser.parse_known_args(remaining_args)
            all_parsers[name.lower()] = parsed
        except Exception as e:
            all_parsers[name.lower()] = None

    if remaining_args:
        print(f"Unparsed arguments: {remaining_args}")

    all_parsers['fl_parser'].model_type = model_types.get(all_parsers['fl_parser'].model)
    all_parsers['fl_parser'].data_type = data_types.get(all_parsers['fl_parser'].model_type)
    all_parsers['data_parser'].data_type = data_types.get(all_parsers['fl_parser'].model_type)

    # Apply replicate_ibm settings: full_info, batching, batchnorm, ibm_fe, ibm_hp
    if all_parsers['data_parser'].replicate_ibm:
        all_parsers['fl_parser'].fl_algo = 'full_info'
        all_parsers['data_parser'].batching = True
        all_parsers['data_parser'].batchnorm = True
        all_parsers['data_parser'].ibm_fe = True
        all_parsers['data_parser'].ibm_hp = True

    all_parsers['data_parser'].scenario = 'individual_banks' if all_parsers['fl_parser'].fl_algo != 'full_info' else 'full_info'

    # Default max_workers to CPU count if not specified
    if all_parsers['fl_parser'].max_workers is None:
        all_parsers['fl_parser'].max_workers = os.cpu_count() or 1

    return all_parsers


# fl parser
def fl_parser():

    parser = argparse.ArgumentParser(description="main args for fl")
    parser.add_argument('--fl_algo', default='FedAvg', type=str)
    parser.add_argument('--model', default='GINe', type=str)
    parser.add_argument('--aggregation', default='shared', type=str,
                        help='Weight aggregation method: shared (single model), fedavg, fedprox, etc.')

    # FedAvg / FedProx arguments
    parser.add_argument('--client_fraction', default=1.0, type=float,
                        help='Fraction C of clients sampled per round (1.0 = all clients)')
    parser.add_argument('--num_local_epochs', default=1, type=int,
                        help='Number of local training epochs E per FL round')
    parser.add_argument('--mu', default=0.0, type=float,
                        help='FedProx proximal term coefficient (0.0 = standard FedAvg)')
    parser.add_argument('--weighting', default='proportional', type=str,
                        choices=['proportional', 'uniform'],
                        help='Aggregation weighting: proportional (by dataset size) or uniform (1/K)')
    parser.add_argument('--num_rounds', default=100, type=int,
                        help='Number of FL communication rounds (default: 100)')
    parser.add_argument('--max_workers', default=4, type=int,
                        help='Number of parallel workers for party training (default: number of CPUs)')
    parser.add_argument('--validate_every', default=1, type=int,
                        help='Run validation every N rounds for model selection (default: 1)')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning. For boosters: full_info saves the result; '
                             'other scenarios run their own tuning. Without this flag, '
                             'previously saved full_info HPs are loaded instead.')

    return parser

# data parser
def data_parser():

    parser = argparse.ArgumentParser(description="args for data configs and utils")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--ibm_fe', action='store_true', help='Set to True if the feature engineering should be 1:1 with the IBM paper')
    parser.add_argument('--ibm_hp', action='store_true', help='Set to True if the IBM hyperparameters should be used')
    parser.add_argument('--batching', action='store_true', help='Set to True if batching should be used during training')
    parser.add_argument('--batching_mode', default='lazy_link_neighbor', type=str,
                        choices=['neighbor_sample', 'simple', 'link_neighbor', 'lazy_link_neighbor'],
                        help='Batch generation strategy for vertical FL (default: lazy_link_neighbor)')
    parser.add_argument('--batch_size', default=8192, type=int, help='Batch size for LinkNeighborLoader (default: 8192)')
    parser.add_argument('--use_global_stats', action='store_true', help='Use global statistics for standardization instead of local party statistics')
    parser.add_argument('--eval_mode', default='system', type=str, choices=['system', 'comparable'],
                        help="Evaluation mode: 'system' uses full test set; 'comparable' restricts to individual banks' data")
    parser.add_argument('--testing_seeds', default=4, type=int, help="The amount of seeds tested in the final evaluation of a model")
    parser.add_argument('--sb_num_rounds', default=None, type=int,
                        help='Override SecureBoost num_rounds from tuned HP (useful for timing tests)')
    parser.add_argument('--tune_max_rounds', default=1000, type=int,
                        help='Upper bound for num_rounds in HP sampler (default 1000). Included in saved HP filename.')
    parser.add_argument('--hp_path', default=None, type=str,
                        help='Path to HP JSON file relative to project root (local) or $VSC_DATA/AML_work_study/AML_work_study (HPC). '
                             'Overrides the auto-detected path from eval_mode/ibm_fe/tune_max_rounds.')
    #parser.add_argument("--add_ids", action='store_true', help="Add ids when batching for vertical learning")
    parser.add_argument('--batchnorm', action='store_true', help="Set to True if BatchNorm should be used in the GNN model")
    parser.add_argument('--replicate_ibm', action='store_true',
                        help="Replicate IBM paper settings: sets full_info, batching, batchnorm, ibm_fe, ibm_hp")

    # Heterogeneity experiments
    parser.add_argument('--loss_ratio', default=None, type=int,
                        help='Override w_ce2 loss weight (w_ce1=1). If not set, uses hyperparameter value')
    parser.add_argument('--normalize_currency', action='store_true',
                        help='Normalize amounts to same currency (heterogeneity experiment, applies to both FE paths)')
    parser.add_argument('--bank_filter', default=None, type=str,
                        choices=['no_top10', 'no_top1', 'no_bottom10', 'no_bottom5pct'],
                        help='Filter banks by edge count: no_top10, no_top1 (outlier), no_bottom10, no_bottom5pct')

    # utils
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--testing_frac', default=0.10, type=float,
                        help="Fraction of data to use when --testing is set (default: 0.10)")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=1, type=int, help="Set seed for reproducability")

    return parser

# gnn parser
def gnn_parser():

    parser = argparse.ArgumentParser(description="args for gnn models")
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--ports", action='store_true')
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")

    return parser



def add_banks_to_manager(parsers, banks, manager, df, scaler_encoders, tuned_hp = None, bank_type='train', superset_merge=True):
    # this part here is only used for individual banks settings
    if tuned_hp is not None:
        best_tuned_hp = max(tuned_hp.values(), key=lambda x: x['f1_score'])['hyperparameters']
        tuned_hp = {bank_id: entry['hyperparameters'] for bank_id, entry in tuned_hp.items()}

    for bank in banks:
        # For non-superset scenarios (e.g. FedAvg): reuse existing party objects for overlapping banks
        existing_party = None
        if not superset_merge and bank_type != 'train':
            existing_party = manager.parties.get(bank)
            if existing_party is None:
                existing_party = manager.vali_parties.get(bank)

        if existing_party is not None:
            manager.add_party(existing_party, bank_type=bank_type)
        else:
            manager._add_party(bank, df, parsers, copy.deepcopy(scaler_encoders), bank_type=bank_type)

        if tuned_hp is not None:
            tuned_hp[bank] = best_tuned_hp

    manager._num_parties = len(manager.parties)

    if superset_merge:
        if bank_type == 'vali':
            manager.vali_parties = manager.parties | manager.vali_parties
            # Sync parties to point to the same objects as vali_parties
            for bank_id in manager.parties:
                manager.parties[bank_id] = manager.vali_parties[bank_id]
        elif bank_type == 'test':
            manager.test_parties = manager.vali_parties | manager.test_parties
            # Sync parties and vali_parties to point to the same objects as test_parties
            for bank_id in manager.parties:
                manager.parties[bank_id] = manager.test_parties[bank_id]
            for bank_id in manager.vali_parties:
                manager.vali_parties[bank_id] = manager.test_parties[bank_id]

    return tuned_hp


