# packages
import pandas as pd
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
from data.get_indices_type_data import get_indices_bdt
import logging
import copy
import data.data_functions as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
import models.gnn_models
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos
import data.feature_engi as fe
from relbanks_saving_analysis.save_results import save_results


def main():

    # Setup logger ----------------------------------------------------------------------------------------

    utils.logger_setup()

    # Get parsers and data ----------------------------------------------------------------------------------------

    # Parsers data -------
    parsers, df, scaler_encoders = utils.setup_get_data()

    logging.info('Parsers, data and scalers loaded')
    logging.info(f"train size: {df['regular_data']['train_data']['x'].shape[0]}, validation size: {df['regular_data']['vali_data']['x'].shape[0]}, test size: {df['regular_data']['test_data']['x'].shape[0]}")

    # df['regular_data']['train_data']['x'].shape[0]

    
    # Laudering values -------
    laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))


    

    # Setup manager and parties ----------------------------------------------------------------------------------------

    # Manager
    manager = Manager.get_algo_class(parsers)

    # Setup parties and tune
    tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)

    # Train the model ----------------------------------------------------------------------------------------

    #results = manager.train(tuned_hp, laundering_values_test)

    # Save restuls ----------------------------------------------------------------------------------------

    #save_results(results, tuned_hp, manager)

    #print(parsers['data_parser'].testing)
    #print(parsers['data_parser'].ibm_fe)


    




if __name__ == '__main__':
    main()

