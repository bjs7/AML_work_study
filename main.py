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
#from relbanks_saving_analysis.relevant_banks import get_relevant_banks

# remove files once no longer needed?
# check om der skal der skal samples 2 droprates?
# residuals i gnn model?

# still missing to put a lot of the stuff / data on CUDA

# setup -----------------------------------------------

parsers = utils.parser_all()
fl_parser, data_parser, gnn_parser = parsers['fl_parser'], parsers['data_parser'], parsers['gnn_parser']
parsers['data_parser'].testing = True

utils.logger_setup()
utils.set_seed(data_parser.seed, True)


# -------------

# need to check that individual saves predictions/laundering values correct

parsers['fl_parser'].fl_algo = 'individual'

# -------------

# Get data ---------------------------------------------------------------------------------------
# is it actually necessary to carry bank indices with the graph data? Or can one just use only graph data
# for gnn models? Think I can, or only have the indices with one, reset of columns can be dropped
df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{data_parser.size}' + f'_{data_parser.ir}' + '.csv')
raw_data = get_data(df, data_parser, split_perc = split_perc)
scaler_encoders = dfn.extract_enc_cats(df)


# check get_relevant_banks functions etc. to make sure everything works as intended
fr_banks, sr_banks = utils.get_relevant_banks(data_parser)
# bank 7 does not have any 1's in validation set, therefore might wanna remove it? But
# then would also be need for first calculations? Like full/individual calculations
relevant_banks = fr_banks #+ sr_banks
relevant_banks = relevant_banks[0:4] + relevant_banks[5:10]


# ------------------------------------------------------------------------------------------------
# laundering values ------------------------------ 
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(data_parser, copy.deepcopy(raw_data))

# -------------------------------------------------------------------------------------------------
# Setup for manager and parties ------------------------------------------------------

# fl setup --------------------------

# Get the manager
manager = Manager.get_algo_class(parsers)
smallest_bank = None
smallest_dim = 1e10



# one can split the GNNMixingParty up, or keep it, but part of it into
# seperate parts, one for FL and one for individual

# there are several functions that could be simplified. Check from top to bottom to find them

# still missing to include more of the features, like sent/received, currency and amount.
# though need to figure out how to deal with it, when applying feature engineering


# optimize, fl_get_data function and the two functions that it uses?

# 39735 is max number of nodes, with 271928 edges, 371380 edges, and 452751 edges, bank 68
# Get the parties, and their data
for bank in relevant_banks:

    bank_indices = get_indices_bdt(raw_data, bank=bank)
    train_data, vali_data, test_data = dfn.fl_get_data(parsers, raw_data, bank_indices)
    tmp_data = {'train_data': train_data, 'vali_data': vali_data, 'test_data': test_data}
    Party.get_algo_class(parsers = parsers, bank_id=bank, data=tmp_data, 
                         indices=bank_indices, manager=manager, 
                         scaler_encoders=scaler_encoders)


manager._num_parties = len(manager.parties)

#smallest_bank, smallest_dim = fl_utils.get_smallest_bank(parsers, train_data, bank, smallest_bank, smallest_dim)
#manager._smallest_bank = smallest_bank

# check that up_data / update_nodes is not used after feature engineering
# Need to check how to "extract" the right indices from GNN when batching. 
# Does having target edges make sense?

# -------------------------------------------------------------------------------------------------

# after tuning, there should be 'cleaned out' in the different variables, dataframe, etc.
# in order to save space
# and just to make sure the stuff from tuning that shouldn't be carried over, isn't carried over
# reset global_w etc.

# design for no FL
manager.set_mode('tuning')
self = manager
manager.mode

laundering_values = laundering_values_vali
tuned_hp = manager.tuning(laundering_values_vali)

hyperparameters = tuned_hp


#manager.__mro__


# train
laundering_values = laundering_values_test
self.set_mode('training')
results = self.train(tuned_hp, laundering_values_test)


# once training is done for individual, one needs to combine values and run inference


# -------------------------------------------------------------------------------------------------

#save_direc = config.save_direc_training
#save_direc = os.path.join(config.save_direc_training, manager.args['fl_parser'].model)


# might not need to attached split_perc to parser?

# need to get the right save direction. Need to find out if it should be saved pth or something else
# and need to check that I can load the parameters
# I am not saving the whole



