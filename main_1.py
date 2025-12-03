import time
import logging
#from utils import create_parser, set_seed, logger_setup
import utils
from training_ import train_gnn
#from inference import infer_gnn
import json
import pandas as pd
from configs import configs

import data_loading as dl
import data.raw_data_processing as rdp


def main():

    utils.logger_setup()

    # Get parsers and data ----------------------------------------------------------------------------------------

    # Parsers data -------
    parsers = utils.parser_all()
    utils.set_seed(parsers['data_parser'].seed, True)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
    parsers['gnn_parser'].model = 'gin'
    parsers['data_parser'].ibm_fe = True
    
    #tr_data, val_data, te_data, tr_inds, val_inds, te_inds = dl.get_data(df, parsers['gnn_parser'])

    graph_data, scaler_encoders  = rdp.get_data(df, parsers['data_parser'], split_perc = [0.6, 0.2])
    graph_data = graph_data['graph_data']

    tr_data, tr_inds = graph_data['train_data']['df'], graph_data['train_data']['pred_indices']
    val_data, val_inds = graph_data['vali_data']['df'], graph_data['vali_data']['pred_indices']
    te_data, te_inds = graph_data['test_data']['df'], graph_data['test_data']['pred_indices']
    
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    #Training
    logging.info(f"Running Training")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, parsers)

if __name__ == "__main__":
    main()