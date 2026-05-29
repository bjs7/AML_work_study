# packages
import utils
import logging
import copy
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
from result_io.save_results import save_results


def main():

    # Setup logger ----------------------------------------------------------------------------------------

    utils.logger_setup()

    # Get parsers and data ----------------------------------------------------------------------------------------

    # Parsers data -------
    parsers, df, scaler_encoders = utils.setup_get_data()

    logging.info('Parsers, data and scalers loaded')
    logging.info(f"train size: {df['regular_data']['train_data']['x'].shape[0]}, validation size: {df['regular_data']['vali_data']['x'].shape[0]}, test size: {df['regular_data']['test_data']['x'].shape[0]}")
    
    # Laudering values -------
    laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], {'regular_data': copy.deepcopy(df['regular_data'])})

    # Setup manager and parties ----------------------------------------------------------------------------------------

    # Manager
    manager = Manager.get_algo_class(parsers)

    # Setup parties and tune
    tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)

    # Tune-only mode: HPs have been saved, skip training
    if parsers['fl_parser'].tune and parsers['fl_parser'].model_type in ('booster', 'gnn'):
        logging.info("Tune-only mode complete. Hyperparameters saved. Run without --tune to train.")
        return

    # Train the model ----------------------------------------------------------------------------------------
    results = manager.train(tuned_hp, laundering_values_vali, laundering_values_test)

    # Save restuls ----------------------------------------------------------------------------------------

    save_results(results, tuned_hp, manager)


if __name__ == '__main__':
    main()

