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
import trainer_utils as tu

from abc import ABC, abstractmethod


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


def get_model(args):
    model_name = args.model

    if model_name not in models_regi:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = models_regi[model_name]
    return model_class(args)


import process_data_type as pdt
import data_functions as data_func

class Model(ABC):

    def __init__(self, args):
        self.args = args
        self.model_type = tu.model_types.get(args.model)

    def get_indices(self, data, bank = None):
        return data_funcs.get_indices_bdt(data, args, bank = bank)

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def tuning(self):
        pass

    @abstractmethod
    def train(self):
        pass


class light_xgb(Model):

    def __init__(self, args):
        super().__init__(args)
    
    def get_indices(self, data, bank = None):
        return super().get_indices(data, bank)

    def get_data(self):
        pass
        #if self.args.scenario == 'individual_banks':
        #    train_data, vali_data, test_data = pdt.update_data(df['test_data']['df'], bank_indices, args)
        #else:
        #    return 0

    def tuning(self):
        pass

    def train(self):
        return super().train()


models_regi = {

    #'GINe': GINe,
    #'xgboost': xgboost,
    'light_xgb': light_xgb

}

# get arguments
parser = get_parser()
args = parser.parse_args()
args.model = 'light_xgb'

model = get_model(args)

model.get_indices(raw_data, bank=5)




df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
raw_data = dp.get_data(df, split_perc = configs.split_perc)


args.model = 'light_xgb'
model = get_model(args)

model.model_type
model.get_indices(raw_data, 5)




















