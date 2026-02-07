from data.get_indices_type_data import get_indices_bdt, get_booster_data, get_graph_data
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def extract_enc_cats(df):

    # for currencies
    sent_received_list = df['Sent Currency'].tolist() + df['Received Currency'].tolist()
    currency_categories = [str(ele) for ele in set(sent_received_list)]
    encoder_cur = OneHotEncoder(categories=[currency_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_cur.fit(np.array(sent_received_list).reshape(-1, 1))

    # for payment format
    payment_categories = [str(ele) for ele in set(df['Payment Format'])]
    encoder_pay = OneHotEncoder(categories=[payment_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_pay.fit(np.array(df['Payment Format'].tolist()).reshape(-1, 1))

    return {'encoder_currency': encoder_cur, 'encoder_payment_format': encoder_pay}

def prep_laundering_dfs(data_parser, data_copy):
    """Prepare validation and test laundering value dataframes.

    Temporarily sets data_parser.scenario to 'full_info' to get booster data,
    then restores the original scenario.
    """
    # Temporarily change scenario to 'full_info'
    original_scenario = data_parser.scenario
    data_parser.scenario = 'full_info'

    bank_indices = get_indices_bdt(data_copy, None)
    train_data, vali_data, test_data = get_booster_data(data_parser, data_copy['regular_data'], bank_indices=None)

    # Create validation laundering values dataframe
    vali_y = vali_data['x']['Is Laundering']
    vali_dict = {
        'indices': bank_indices['vali_indices'],
        'true_y': vali_y,
        'pred_probabilities': np.zeros(len(vali_y)),
        'pred_label': np.zeros(len(vali_y)),
        'num_prob': np.zeros(len(vali_y)),
        'avg_prob': np.zeros(len(vali_y)),
        'max_prob': np.zeros(len(vali_y))
    }
    if 'Pattern' in vali_data['x'].columns:
        vali_dict['Pattern'] = vali_data['x']['Pattern'].values
    laundering_values_vali = pd.DataFrame(vali_dict)

    # Create test laundering values dataframe
    test_y = test_data['x']['Is Laundering']
    test_dict = {
        'indices': bank_indices['test_indices'],
        'true_y': test_y,
        'pred_probabilities': np.zeros(len(test_y)),
        'pred_label': np.zeros(len(test_y)),
        'num_prob': np.zeros(len(test_y)),
        'avg_prob': np.zeros(len(test_y)),
        'max_prob': np.zeros(len(test_y))
    }
    if 'Pattern' in test_data['x'].columns:
        test_dict['Pattern'] = test_data['x']['Pattern'].values
    laundering_values_test = pd.DataFrame(test_dict)

    # Restore original scenario
    data_parser.scenario = original_scenario

    return laundering_values_vali, laundering_values_test



def fl_get_data(parsers, df, bank_indices = None):
    """Get data for federated learning based on data type.

    Args:
        parsers: Dictionary containing parser configurations
        raw_data: Raw data dictionary with 'graph_data' and 'regular_data' keys
        bank_indices: Bank-specific indices for data splitting

    Returns:
        Processed data (graph or booster format) based on data_type
    """
    data_parser = parsers['data_parser']
    data_type = data_parser.data_type

    if data_type == 'graph_data':
        return get_graph_data(parsers, df[data_type], bank_indices)
    elif data_type == 'regular_data':
        return get_booster_data(data_parser, df[data_type], bank_indices)


