import utils
import logging
import argparse
import pandas as pd
import data_processing as dp
import configs
import data_functions as data_funcs
import tuning as tune
import train_gnn as tr_gnn
import save_load_models as slm


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GINe', type=str, help='Select the type of model to train')
    parser.add_argument('--scenario', default='full_info', type=str, help='Select the scenario to study')
    parser.add_argument('--banks', default=[], type=utils.parse_banks, help='Used if specific banks are to be studied')
    parser.add_argument('--model_configs', default=None, type=str, help='should the hyperparameters be tuned, else provide some')
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")

    parser.add_argument('--graph_tuning_x_0', default = 2, type=int, help='Amount of models to train on for tuning GNN')
    parser.add_argument('--seed', default=0, type=int, help="Select the random seed for reproducability")
    #parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    
    return parser


def main():

    # set logging
    utils.logger_setup()
    
    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    # set seed
    utils.set_seed(args.seed)

    # load data
    logging.info("load_data")
    df = pd.read_csv("/home/nam_07/AML_work_study/formatted_transactions.csv")
    #df = pd.read_csv("/data/leuven/362/vsc36278/AML_work_study/formatted_transactions.csv")
    raw_data = dp.get_data(df, split_perc = configs.split_perc)
    
    if args.scenario == 'individual_banks':
        Banks = list((df.loc[:, 'From Bank'])) + list((df.loc[:, 'To Bank']))
        unique_banks = list(set(Banks))
        args.banks = unique_banks

    if args.banks:
        args.scenario = 'individual_banks'
    
    if args.scenario == 'individual_banks':

        no_train_data = []
        no_vali_data = []
        no_test_data = []
        best_f1 = -1
        
        # args.banks
        #range(10)
        for bank in args.banks:

            #bank = 5
            #args.scenario = 'individual_banks'

            # first get bank indices
            bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)

            if not bank_indices['train_data_indices'] or not bank_indices['vali_data_indices'] or not bank_indices['test_data_indices']:
                if not bank_indices['train_data_indices']:
                    no_train_data.append(bank)
                if not bank_indices['vali_data_indices']:
                    no_vali_data.append(bank)
                if not bank_indices['test_data_indices']:
                    no_test_data.append(bank)
                continue

            # next the data for training, validatio and testing, no feature engineering applied
            train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)
            
            # tune
            tuned_hyperparameters = tune.tuning(args, train_data, vali_data, bank_indices)

            # train based on tuned hyperparameters
            trained_model_f1 = tr_gnn.train_gnn(args, vali_data, test_data, tuned_hyperparameters)

            # save the model
            folder_name = f'bank_{bank}' if bank else args.scenario
            for seed in trained_model_f1.keys():

                if trained_model_f1[seed]['f1'] > best_f1:
                    best_hyperparameters = tuned_hyperparameters
                    best_f1 = trained_model_f1[seed]['f1']

                slm.save_model(trained_model_f1[seed]['model'], tuned_hyperparameters, args, folder_name, file_name = seed)

        # banks that still needs to be trained
        banks_to_train = list(set(no_vali_data) - set(no_test_data))

        for bank in banks_to_train:

            # first get bank indices
            bank_indices = data_funcs.get_indices_bdt(raw_data, args, bank = bank)

            # next the data for training, validatio and testing, no feature engineering applied
            train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=bank_indices)

            # tuning here i skipped and go straight to training
            trained_model_f1 = tr_gnn.train_gnn(args, vali_data, test_data, best_hyperparameters)

            folder_name = f'bank_{bank}' if bank else args.scenario
            for seed in trained_model_f1.keys():
                slm.save_model(trained_model_f1[seed]['model'], best_hyperparameters, args, folder_name, file_name = seed)

             
    else:

        # get data for training, validatio and testing, no feature engineering applied
        train_data, vali_data, test_data = data_funcs.get_graph_data(raw_data, args, bank_indices=None)

        # tune
        tuned_hyperparameters = tune.tuning(args, train_data, vali_data, None)

        # train based on tuned hyperparameters
        trained_model_f1 = tr_gnn.train_gnn(args, vali_data, test_data, tuned_hyperparameters)

        # save the model
        folder_name = f'bank_{bank}' if bank else args.scenario
        for seed in trained_model_f1.keys():
            slm.save_model(trained_model_f1[seed]['model'], tuned_hyperparameters, args, folder_name, file_name = seed)


if __name__ == '__main__':
    main()



