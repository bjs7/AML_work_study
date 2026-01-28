# %% 

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

import matplotlib.pyplot as plt
import numpy as np



# %%


utils.logger_setup()
parsers = utils.parser_all()
#parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)
# -------------

# need to check that individual saves predictions/laundering values correct

parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
parsers['data_parser'].train_for_final = True


#parsers['fl_parser'].fl_algo = 'full_info'
#parsers['data_parser'].scenario = 'individual_banks' if parsers['fl_parser'].fl_algo != 'full_info' else 'full_info'

parsers['fl_parser'].fl_algo = 'individual'

# Get data ---------------------------------------------------------------------------------------
df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)

# ------------------------------------------------------------------------------------------------
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))


# %%
manager = Manager.get_algo_class(parsers)
self = manager
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)


# %%

num_nodes = []
num_edges = []
num_edges_train = []
num_edges_bank_id = {}
less_than_1k_train = []
less_than_1k_test = []

for bank_id, party in manager.parties.items():
    num_nodes.append(party.data['test_data']['df'].x.shape[0])
    num_edges.append(party.data['test_data']['df'].y.shape[0])
    num_edges_bank_id[bank_id] = party.data['test_data']['df'].y.shape[0]

    num_edges_train.append(party.data['train_data']['df'].y.shape[0])
    
    if party.data['test_data']['df'].y.shape[0] < 1000:
        less_than_1k_test.append(bank_id)

    if party.data['train_data']['df'].y.shape[0] < 1000:
        less_than_1k_train.append(bank_id)


# %%

top_30 = dict(sorted(num_edges_bank_id.items(), key=lambda x: x[1], reverse=True)[:30])
top_30.keys()


# %%



fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(num_nodes, bins=100, edgecolor='black')
#axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('num_nodes')

axes[1].hist(num_edges, bins=100, edgecolor='black')
#axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('num_edges')

plt.tight_layout()
plt.show()



# %%

np.where(num_edges < 1000)

whereis = [i for idx, i in enumerate(num_edges) if i < 1000]


llengths = []

for bank in less_than_1k_train:
    llengths.append(len(manager.parties[bank].data['train_data']['df'].y))


min(num_edges_train)


