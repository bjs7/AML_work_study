# packages
import pandas as pd
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
from data.get_indices_type_data import get_indices_bdt
import copy
import data.data_functions as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
import models.gnn_models
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos
import data.feature_engi as fe
from results.save_results import save_results
from data.relevant_banks import get_relevant_banks

# setup -----------------------------------------------

utils.logger_setup()
parsers = utils.parser_all()
parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)
# -------------

parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
parsers['data_parser'].add_ids = False

#parsers['fl_parser'].fl_algo = 'FedVert'
#parsers['fl_parser'].fl_algo = 'full_info'
#parsers['data_parser'].scenario = 'full_info'

parsers['fl_parser'].fl_algo = 'FedAvg'
parsers['data_parser'].batching = True

parsers['fl_parser'].client_fraction = 0.25
parsers['fl_parser'].num_local_epochs = 10


parsers['fl_parser'].fl_algo = 'individual'
parsers['data_parser'].scenario = 'individual_banks'

# Get data ---------------------------------------------------------------------------------------
df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

if parsers['data_parser'].testing:
    df = df.iloc[:round(df.shape[0] * 0.05),:]

#data_paser = parsers['data_parser']

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)

# ------------------------------------------------------------------------------------------------
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))

# -------------------------------------------------------------------------------------------------
# Setup for manager and parties ------------------------------------------------------

# Get the manager
manager = Manager.get_algo_class(parsers)

self = manager
laundering_values = laundering_values_vali

# dynamic for both full and individual
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)
hyperparameters = tuned_hp


testindiceslen = []

for bank_id, party in self.parties.items():
    if len(party.indices['test_indices']) < 5:
        testindiceslen.append(bank_id)



self = self.parties[0]

len(self.parties[0].indices['test_indices'])


sum(self.parties[0].data['train_data']['df'].y)
sum(self.parties[0].data['vali_data']['df'].y)
sum(self.parties[0].data['test_data']['df'].y)





self.parties[0].data['train_data']['df']

#party = self.parties[0]
party = self.parties[None]
self = party



self = manager
#laundering_values = laundering_values_vali
laundering_values = laundering_values_test

self.args['data_parser'].testing_seeds = 1

results = self.train(tuned_hp, laundering_values_vali, laundering_values_test)

save_results(results, hyperparameters, manager)


self.edge_feat_start
self.parties[None].edge_feat_start

self.parties[None].data['train_data']['df'].edge_attr
self.parties[None].procs_data['train_data']['df'].edge_attr

import torch
self.parties[None].data['train_data']['df'].edge_attr[:,0] == self.parties[None].procs_data['train_data']['df'].edge_attr[:,0]
torch.all(self.parties[None].data['train_data']['df'].edge_attr[:,0] == self.parties[None].procs_data['train_data']['df'].edge_attr[:,0])
self.parties[None].procs_data['train_data']['df'].edge_attr[:,0]




# --------------------------


if data_parser.eval_mode == 'comparable':
    comparable_indices = load_comparable_indices(...)  # from JSON
    indices = [
        np.array([i for i in indices[0] if i in comparable_indices['train']]),
        np.array([i for i in indices[1] if i in comparable_indices['vali']]),
        np.array([i for i in indices[2] if i in comparable_indices['test']])
    ]



banks = get_relevant_banks(parsers)
train_indices, vali_indices, test_indices = [], [], []

for bank_id in banks:
    indices = get_indices_bdt(df, bank_id)
    train_indices += indices['train_indices']
    vali_indices += indices['vali_indices']
    test_indicies += indices['test_indicies']



pack_graph_data


# --------------------------

