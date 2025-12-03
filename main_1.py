import time
import logging
#from utils import create_parser, set_seed, logger_setup
import utils
from data_loading import get_data
from training import train_gnn
#from inference import infer_gnn
import json
import pandas as pd

def main():

    utils.logger_setup()

    # Get parsers and data ----------------------------------------------------------------------------------------

    # Parsers data -------
    parsers = utils.parser_all()
    utils.set_seed(parsers['data_parser'].seed, True)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    df_edges = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
    parsers['gnn_parser'].model = 'gin'
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(df_edges, parsers['gnn_parser'])
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    #Training
    logging.info(f"Running Training")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, parsers)

if __name__ == "__main__":
    main()