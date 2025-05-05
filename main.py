import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_functions as data_funcs
import tuning as tune
import train_models as tr_models
import save_load_models as slm
from relevant_banks import get_relevant_banks


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")

    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument('--seed', default=0, type=int, help="Set seed for reproducability")

    # Data configs
    parser.add_argument('--size', default='small', type=str, help="Select the dataset size")
    parser.add_argument('--ir', default='HI', type=str, help="Select the illicit ratio")
    parser.add_argument('--banks', default='only_launderings', type=str)
    parser.add_argument('--specific_banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    
    return parser

# for booster, in tuning, the fraction used, select it randomly?
def main():

    # Logging, seeding, data args passing --------------------------------------------------------------------------------------------------

    # set logging
    utils.logger_setup()
    
    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    # set seed
    utils.set_seed(args.seed, True)

    # load data
    logging.info("load_data")
    # also remember to change x_0 etc.
    #df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    df = pd.read_csv('/data/leuven/362/vsc36278/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    raw_data = dp.get_data(df, split_perc = configs.split_perc)
    logging.info("Obtained data")


    if args.scenario == 'individual_banks':
        fr_banks, sr_banks = get_relevant_banks(args)
    
    if args.specific_banks:
        args.scenario = 'individual_banks'
        fr_banks = args.specific_banks
        sr_banks = []


    # Individual banks -------------------------------------------------------------------------------------------------------------
    
    if args.scenario == 'individual_banks':

        best_f1 = -1

        log_every = (round(len(fr_banks) / 10))
        #(round(len(fr_banks) / round(len(fr_banks) * 0.1)))
        
        for index, bank in enumerate(fr_banks[0:5]):
            
            if index % log_every == 0: logging.info(f'Starting training on bank {bank}, index: {index}')

            #bank = 6
            #bank = 5
            #bank = 3
            #args.scenario = 'individual_banks'
            #args.model = 'xgboost'
            #args.model = 'GINe'

            # first get bank indices
            if index % 1000 == 0: logging.info(f'Get indices for bank {bank}, index: {index}')
            bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)

            # next the data for training, validatio and testing, no feature engineering applied
            if index % 1000 == 0: logging.info(f'Getting data for bank {bank}')
            train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
            
            # tune
            if index % 1000 == 0: logging.info(f'Tuning bank {bank}, index: {index}')
            tuned_hyperparameters = tune.tuning(args, train_data, vali_data, bank_indices)

            # train based on tuned hyperparameters
            if index % 1000 == 0: logging.info(f'Traning bank {bank}, index: {index}')
            trained_model_f1 = tr_models.train_model(args, train_data, vali_data, test_data, tuned_hyperparameters)

            # save the model
            if index % 1000 == 0: logging.info(f'Saving bank {bank}, index: {index}')

            q = 0
            folder_name = f'bank_{bank}'
            for seed in trained_model_f1.keys():
                q += 1
                if trained_model_f1[seed]['f1'] > best_f1:
                    best_hyperparameters = tuned_hyperparameters
                    best_f1 = trained_model_f1[seed]['f1']
                
                slm.save_model(trained_model_f1[seed]['model'], tuned_hyperparameters, args, folder_name, file_name = seed, q = q)   


        # Training of banks with no validation data ---------------------------------------------------------------------------------------
        logging.info(f'Starting training on banks that had no training or validation data')
        for index, bank in enumerate(sr_banks):

            if index % 1000 == 0: logging.info(f'Starting training on bank {bank}, index: {index}')

            # first get bank indices
            if index % 1000 == 0: logging.info(f'Get indices for bank {bank}, index: {index}')
            bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)

            # next the data for training, validatio and testing, no feature engineering applied
            if index % 1000 == 0: logging.info(f'Getting data for bank {bank}, index: {index}')
            train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)

            # tuning here i skipped and go straight to training
            if index % 1000 == 0: logging.info(f'Training bank {bank}, index: {index}')
            trained_model_f1 = tr_models.train_model(args, train_data, vali_data, test_data, best_hyperparameters)

            if index % 1000 == 0: logging.info(f'Saving bank {bank}, index: {index}')
            q = 0
            folder_name = f'bank_{bank}'
            for seed in trained_model_f1.keys():
                q += 1
                slm.save_model(trained_model_f1[seed]['model'], best_hyperparameters, args, folder_name, file_name = seed, q = q)


    # Full information -------------------------------------------------------------------------------------------------------------------
             
    else:

        logging.info('Starting on full information scenario')
        # get data for training, validatio and testing, no feature engineering applied
        logging.info('Get graph data')
        train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=None)

        # tune
        logging.info('Tuning the model')
        tuned_hyperparameters = tune.tuning(args, train_data, vali_data, None)

        # train based on tuned hyperparameters
        logging.info('Training the model')
        trained_model_f1 = tr_models.train_model(args, vali_data, test_data, tuned_hyperparameters)

        # save the model
        logging.info('Saving the model')
        q = 0
        folder_name = 'full_info'
        for seed in trained_model_f1.keys():
            q += 1
            slm.save_model(trained_model_f1[seed]['model'], tuned_hyperparameters, args, folder_name, file_name = seed, q = q)


if __name__ == '__main__':
    main()



