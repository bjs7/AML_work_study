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
from relbanks_saving_analysis.save_results import save_results
from relbanks_saving_analysis.relevant_banks import get_relevant_banks


#from examples.checkpoint_workflow import save_checkpoint, load_checkpoint

# add settings, like type of algo etc. to sbatch info to output file of jobs?

# setup -----------------------------------------------

utils.logger_setup()
parsers = utils.parser_all()
parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)
# -------------

# need to check that individual saves predictions/laundering values correct

parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
parsers['data_parser'].train_for_final = True

parsers['fl_parser'].fl_algo = 'FedVert'
#parsers['fl_parser'].model = 'GINe'

# Get data ---------------------------------------------------------------------------------------
df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

if parsers['data_parser'].testing:
    df = pd.concat([df.iloc[0:50000,:], df.iloc[3000000:3050000,:], df.iloc[5000000:5050000,:]]).reset_index()

#unique_banks = sorted(df[['From Bank', 'To Bank']].stack().unique())

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



self = manager
#laundering_values = laundering_values_vali
laundering_values = laundering_values_test







# where is this used, and where is it set?
self.label_data



#results = manager.train(tuned_hp, laundering_values_test)


# save results
#save_results(results, tuned_hp, manager)





# ------------------------

# be sure to check that all values are as they supposed to.
# Or like that everything works as it is supposed to, check for errors etc.
# optimize, fl_get_data function and the two functions that it uses?
# 39735 is max number of nodes, with 271928 edges, 371380 edges, and 452751 edges, bank 68

# check that up_data / update_nodes is not used after feature engineering
