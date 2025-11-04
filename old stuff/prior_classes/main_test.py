import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_functions as data_funcs
import tuning as tune
import train_models as tr_models
import inference_saving.save_load_models as slm
from relevant_banks import get_relevant_banks


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")

    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=0, type=int, help="Set seed for reproducability")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--specific_banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    
    return parser


# set logging
utils.logger_setup()

# get arguments
parser = get_parser()
args = parser.parse_args()

# set seed
utils.set_seed(args.seed, True)

# load data
logging.info("load_data")
# also remember to change x_0 etc.
df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
#df = pd.read_csv('/data/leuven/362/vsc36278/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
raw_data = dp.get_data(df, split_perc = configs.split_perc)
logging.info("Obtained data")


if args.scenario == 'individual_banks':
    fr_banks, sr_banks = get_relevant_banks(args)

if args.specific_banks:
    args.scenario = 'individual_banks'
    fr_banks = args.specific_banks
    sr_banks = []


args.scenario = 'individual_banks'
args.model = 'lightGBM'
bank = 5

bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)

train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
tuned_hyperparameters = tune.tuning(args, train_data, vali_data, bank_indices)

trained_model_f1 = tr_models.train_model(args, train_data, vali_data, test_data, tuned_hyperparameters)
















