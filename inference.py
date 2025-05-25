
import utils
import logging
import pandas as pd
import numpy as np
import copy
from data.relevant_banks import get_relevant_banks
import pandas as pd
from data.raw_data_processing import get_data
from models import booster, gnn
from models.base import Model, InferenceModel
from models.booster import BoosterInf
from models.gnn import GNNInf
from configs.configs import split_perc
from utils import get_parser
from sklearn.metrics import f1_score
from pathlib import Path
import json


def main():


    # --------------------------------------------------------------------------------------------------------------------------
    # setup and load data ------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    utils.logger_setup()
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    logging.info("load_data")
    model = Model.from_model_type(args)
    
    df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    #df = pd.read_csv('/home/nam_07/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    #df = pd.read_csv('/data/leuven/362/vsc36278/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
    raw_data = get_data(df, model.args, split_perc = split_perc)
    raw_data_copy = copy.deepcopy(raw_data)

    logging.info("get dataframe with true values")
    save_model_arg = args.model
    args.model = 'xgboost'
    args.scenario = 'full_info'
    infer = InferenceModel.from_model_type(args)
    bank_indices = infer.get_indices(raw_data_copy, bank=None)
    train_data, vali_data, test_data = infer.get_data(raw_data_copy, bank_indices=None)
    laundering_values = pd.DataFrame( {'indices': bank_indices['test_indices'], 
                                    'true y': test_data['x']['Is Laundering'],
                                    'predicted_full_info': test_data['x']['Is Laundering'].shape[0] * [None],
                                    'predicted_individual_banks': test_data['x']['Is Laundering'].shape[0] * [0]
                                    })

    args.model = save_model_arg



    # -------------------------------------------------------------------------------------------------------------------------------------
    # full info ---------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------


    logging.info("get predictions from full info")

    args.scenario = 'individual_banks'
    bank = None
    bank = 4

    # get inferencer
    infer = InferenceModel.from_model_type(args)
    tmp_folder, model_parameters = infer.get_folder_params(bank)
    test_data, test_indices = infer.get_test_indices_data(raw_data, bank)

    model = infer.get_model(test_data, model_parameters, tmp_folder)
    predictions, f1_values = infer.get_predictions(model, test_data, model_parameters, tmp_folder)
    laundering_values['predicted_full_info'] = 1
    #laundering_values['predicted_full_info'] = predictions


    # -------------------------------------------------------------------------------------------------------------------------------------
    # individual banks --------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------


    logging.info("get predictions from individual bank")
    args.scenario = 'individual_banks'
    fr_banks, sr_banks = get_relevant_banks(args)
    banks = fr_banks + sr_banks

    for bank in [0,2,4,5,7]: # banks

        infer = InferenceModel.from_model_type(args)
        tmp_folder, model_parameters = infer.get_folder_params(bank)
        test_data, test_indices = infer.get_test_indices_data(raw_data, bank)

        model = infer.get_model(test_data, model_parameters, tmp_folder)
        predictions, f1_values = infer.get_predictions(model, test_data, model_parameters, tmp_folder, f1_values)
        tmp_preidctions = pd.DataFrame({'original_indices': test_indices, 'predictions': predictions})

        if sum(tmp_preidctions['predictions'] == 1) == 0:
            continue

        indices_to_update = tmp_preidctions['original_indices'][np.where(tmp_preidctions['predictions'] == 1)[0]]
        get_list = np.where(np.isin(laundering_values['indices'], indices_to_update))[0]
        laundering_values.loc[get_list,'predicted_individual_banks'] = 1


    logging.info("save the results")
    save_direc = infer.main_folder
    folder_path = Path(save_direc)
    folder_path.mkdir(parents=True, exist_ok=True)

    results = {
        'laundering_values': laundering_values.to_dict(orient='split'),
        'f1_values': f1_values.to_dict(orient='split')
    }

    file_path = folder_path / 'inference_results.json'
    with open(file_path, 'w') as f:
        json.dump(results, f)
    logging.info(f"Saved results to {file_path}")



if __name__ == '__main__':
    main()




    #logging.info("get f1 values")
    #f1_score_full_info = f1_score(laundering_values['true y'], laundering_values['predicted_full_info'], average='binary')
    #f1_score_individual_banks = f1_score(laundering_values['true y'], laundering_values['predicted_individual_banks'], average='binary')
    #results = {'f1_score_full_info': f1_score_full_info, 'f1_score_individual_banks': f1_score_individual_banks, 'f1_values': f1_values.to_dict()}



