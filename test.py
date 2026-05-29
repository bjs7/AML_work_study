# packages
import pandas as pd
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
#from data.get_indices_type_data import get_indices_bdt
import copy
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
#from results.save_results import save_results

# setup -----------------------------------------------

utils.logger_setup()
parsers = utils.parser_all()
#parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)

parsers['fl_parser'].model = 'GINe'
parsers['fl_parser'].fl_algo = 'FedGraphSimple'
parsers['data_parser'].eval_mode = 'system'
parsers['data_parser'].ibm_hp = True



df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/data/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

if parsers['data_parser'].testing:
    df = df.iloc[:round(df.shape[0] * 0.05),:]

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc = split_perc)


laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(
    parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})




manager = Manager.get_algo_class(parsers)
self = manager

laundering_values = laundering_values_vali
mode = 'training'

train_banks = train_banks[0:10]



manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)



manager.mode
manager.parties[0].prep_data

for bank_id, party in manager.parties.items(): #self.iter_parties(include_test): #TODO Needs to only be self.parties when tuning?
    party.prep_data()

