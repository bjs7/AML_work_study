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


# setup -----------------------------------------------

# have logging for most, but still need for federated manager

# caching in managers?

# ALSO NEED TO ADD MORE LOGGING AND SAVE THE LOGGING IN THE EXPERIEMENTS FOLDER!

# smart way for how to handle / sort folders on hpc and how to set "algos" etc. in the .sh file

# be sure that gpu is used on hpc

# in tuning, most of the metrics calcuations can be skipped
# conditino that only adds max_prob, avg_prob etc. if not full info?



utils.logger_setup()
parsers = utils.parser_all()
parsers['data_parser'].testing = True
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

if parsers['fl_parser'].fl_algo == 'full_info':
    df = pd.concat([df.iloc[0:50000,:], df.iloc[3000000:3050000,:], df.iloc[5000000:5050000,:]])

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)

# ------------------------------------------------------------------------------------------------
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))

# -------------------------------------------------------------------------------------------------
# Setup for manager and parties ------------------------------------------------------

# Get the manager
manager = Manager.get_algo_class(parsers)

# dynamic for both full and individual
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)


self = manager
#laundering_values = laundering_values_vali
laundering_values = laundering_values_test
hyperparameters = tuned_hp


results = manager.train(tuned_hp, laundering_values_test)


# save results
save_results(results, tuned_hp, manager)





# ------------------------

# be sure to check that all values are as they supposed to.
# Or like that everything works as it is supposed to, check for errors etc.
# optimize, fl_get_data function and the two functions that it uses?
# 39735 is max number of nodes, with 271928 edges, 371380 edges, and 452751 edges, bank 68

# check that up_data / update_nodes is not used after feature engineering
