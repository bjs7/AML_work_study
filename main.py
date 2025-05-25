import utils
import logging
import argparse
import pandas as pd
from models.base import Model
from models import booster, gnn
from data.raw_data_processing import get_data
from configs.configs import split_perc
import inference_saving.save_load_models as slm
from data.relevant_banks import get_relevant_banks


# for booster, in tuning, the fraction used, select it randomly?
def main():

    # Logging, seeding, data args passing --------------------------------------------------------------------------------------------------

    utils.logger_setup()
    parser = utils.get_parser()
    args = parser.parse_args()

    utils.set_seed(args.seed, True)
    model = Model.from_model_type(args)

    logging.info("load_data")
    # also remember to change x_0 etc.
    df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    #df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    #df = pd.read_csv('/data/leuven/362/vsc36278/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    raw_data = get_data(df, model.args, split_perc = split_perc)
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
        
        for index, bank in enumerate(fr_banks):
            

            if index % log_every == 0: logging.info(f'Starting training on bank {bank}, index: {index}')
            model = Model.from_model_type(args)


            if index % 1000 == 0: logging.info(f'Get indices for bank {bank}, index: {index}')
            bank_indices = model.get_indices(raw_data, bank=bank)


            if index % 1000 == 0: logging.info(f'Getting data for bank {bank}')
            train_data, vali_data, test_data = model.get_data(raw_data, bank_indices)
            

            if index % 1000 == 0: logging.info(f'Tuning bank {bank}, index: {index}')
            tuned_hyperparameters = model.tuning(train_data, vali_data)


            if index % 1000 == 0: logging.info(f'Traning bank {bank}, index: {index}')
            trained_model_f1 = model.train(train_data, vali_data, test_data, tuned_hyperparameters)


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
            model = Model.from_model_type(args)


            if index % 1000 == 0: logging.info(f'Get indices for bank {bank}, index: {index}')
            bank_indices = model.get_indices(raw_data, bank=bank)


            if index % 1000 == 0: logging.info(f'Getting data for bank {bank}, index: {index}')
            train_data, vali_data, test_data = model.get_data(raw_data, bank_indices)

            
            if index % 1000 == 0: logging.info(f'Training bank {bank}, index: {index}')
            trained_model_f1 = model.train(train_data, vali_data, test_data, tuned_hyperparameters)


            if index % 1000 == 0: logging.info(f'Saving bank {bank}, index: {index}')
            q = 0
            folder_name = f'bank_{bank}'
            for seed in trained_model_f1.keys():
                q += 1
                slm.save_model(trained_model_f1[seed]['model'], best_hyperparameters, args, folder_name, file_name = seed, q = q)


    # Full information -------------------------------------------------------------------------------------------------------------------
             
    else:

        logging.info('Starting on full information scenario')
        model = Model.from_model_type(args)


        logging.info('Get data')
        train_data, vali_data, test_data = model.get_data(raw_data, bank_indices=None)


        logging.info('Tuning the model')
        tuned_hyperparameters = model.tuning(train_data, vali_data)


        logging.info('Training the model')
        trained_model_f1 = model.train(train_data, vali_data, test_data, tuned_hyperparameters)


        logging.info('Saving the model')
        q = 0
        folder_name = 'full_info'
        for seed in trained_model_f1.keys():
            q += 1
            slm.save_model(trained_model_f1[seed]['model'], tuned_hyperparameters, args, folder_name, file_name = seed, q = q)


if __name__ == '__main__':
    main()



