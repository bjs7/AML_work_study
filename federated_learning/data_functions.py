import ast
import argparse
import logging
import os
import sys
import torch
import random
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

from data.get_indices_type_data import get_indices_bdt
from data.get_indices_type_data import get_booster_data

from data.get_indices_type_data import get_booster_data, get_graph_data


def extract_enc_cats(df):

    # for currencies
    sent_received_list = list(df['Sent Currency']) + list(df['Received Currency'])
    currency_categories = [str(ele) for ele in set(sent_received_list)]
    encoder_cur = OneHotEncoder(categories=[currency_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_cur.fit(np.array([sent_received_list]).T)

    # for payment format
    payment_categories = [str(ele) for ele in set(df['Payment Format'])]
    encoder_pay = OneHotEncoder(categories=[payment_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_pay.fit(np.array([df.loc[:,'Payment Format']]).T)

    return {'encoder_currency': encoder_cur, 'encoder_payment_format': encoder_pay}


# the get_booster_data function used here, check if it is used other places, because if not, 
# then it can potentially be modified
def prep_laundering_dfs(data_parser, data_copy):

    data_parser_holder = data_parser.scenario
    data_parser.scenario = 'full_info'
    
    #raw_data_copy = copy.deepcopy(raw_data)
    bank_indices = get_indices_bdt(data_copy, None)
    train_data, vali_data, test_data = get_booster_data(data_parser, data_copy['regular_data'], bank_indices=None)

    laundering_values_vali = pd.DataFrame( {'indices': bank_indices['vali_indices'], 
                                'true_y': vali_data['x']['Is Laundering'], 
                                'predictions_fl': vali_data['x']['Is Laundering'].shape[0] * [0]}) #None

    laundering_values_test = pd.DataFrame( {'indices': bank_indices['test_indices'], 
                                    'true_y': test_data['x']['Is Laundering'], 
                                    'predictions_fl': test_data['x']['Is Laundering'].shape[0] * [0]}) #None

    data_parser.scenario = data_parser_holder

    return laundering_values_vali, laundering_values_test



def fl_get_data(parsers, raw_data, bank_indices):

    if parsers['data_parser'].data_type == 'graph_data':
        return get_graph_data(parsers, raw_data[parsers['data_parser'].data_type], bank_indices)
    elif parsers['data_parser'].data_type == 'regular_data':
        return get_booster_data(parsers['data_parser'], raw_data[parsers['data_parser'].data_type], bank_indices)




"""

    # for sent currency
    sent_currency_categories = [str(ele) for ele in set(df['Sent Currency'])]
    encoder_sent_cur = OneHotEncoder(categories=[sent_currency_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_sent_cur.fit(np.array([df.loc[:,'Sent Currency']]).T)


    # for received currency
    received_currency_categories = [str(ele) for ele in set(df['Received Currency'])]
    encoder_received_cur = OneHotEncoder(categories=[received_currency_categories],
                            sparse_output=False, handle_unknown='ignore')
    encoder_received_cur.fit(np.array([df.loc[:,'Received Currency']]).T)

"""


