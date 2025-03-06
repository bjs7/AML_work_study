import math
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.dates import get_epoch

df = pd.read_csv('C:\\Users\\u0168001\\OneDrive - KU Leuven\\Desktop\\Courses\\AML_work_study\\Multi-GNN\\formatted_transactions.csv')



lBanks = list((df.loc[:, 'From Bank'])) + list((df.loc[:, 'To Bank']))
unique_banks = list(set(lBanks))
unique_banks = unique_banks[0:100]
number_of_banks = len(unique_banks)

####################################################################################
# Need to revist processing of data. Like what does a bank actually have access to?#
####################################################################################

#number_of_banks = len(list(set(df.loc[:, 'From Bank'])))
#len(list(set(df.loc[:, 'To Bank'])))

# can the banks only see amount received/send?

# method for dealing with banks not having certain observations.
# create column where each observation holds a touple of the banks with access to the data
# use this in the input of the function, if bank value in the touple process through model,
# if not return 0

# how to split for train, test and validation? Do that first, and then split on bank
# or split on bank and then on time
# potentially do both?

# overveje evt. om der først skal sorteres på tid, også køres formateringen af dataen, altså i format_kaggle scriptet
# overveje om dataframes skal have der index reset

# should from_bank and to_bank be included as features? Probably not

"""banks write a SARS report and submit to the people above


3 frictions - banks only have info on their customers
banks can't share network
banks don't know the labels

first two er horizonal
last one is vertical

IBM has 8 motifs

"""





def data_split_time_bank(df, unique_banks, train_perc = 0.8, test_perc = 0.1):

    bank_splits = {}
    epochs_per_model = {}

    # skal lige have dobblte checket den her udregning af test_perc etc.
    test_perc = round(test_perc, 10) / round(1 - train_perc, 10)
    vali_perc = round(1 - test_perc, 10)

    num_training = math.ceil(df.shape[0] * train_perc)
    num_training = len(np.where(df.loc[:, 'Timestamp'] <= df.loc[num_training, 'Timestamp'])[0])
    data_train = df.iloc[:num_training, :]

    left_to_test_vali = df.shape[0] - data_train.shape[0]
    num_testing = math.ceil(left_to_test_vali * test_perc)
    num_testing = len(np.where(df.loc[:, 'Timestamp'] <= df.loc[(num_training + num_testing), 'Timestamp'])[0])

    data_test = df.iloc[num_training:num_testing, :]
    data_val = df.iloc[num_testing:] if vali_perc > 0 else None

    for bank in unique_banks:
        X_train, y_train = get_bank_data(data_train, bank)
        X_test, y_test = get_bank_data(data_test, bank)
        X_val, y_val = get_bank_data(data_val, bank) if vali_perc > 0 else None

        bank_splits[bank] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_val': X_val, 'y_val': y_val
        }

        epochs_per_model[bank] = get_epochs(X_train.shape[0])

    return bank_splits, epochs_per_model



def get_epochs(data_size):
    x = (
        250 if data_size > 5000 else
        50 if data_size > 2500 else
        25 if data_size > 1000 else
        10 if data_size > 250 else
        5
    )
    return x



#, X_and_y = False
def get_bank_data(data_frame, bank_number, include_tobank = True):

    listwith_tobank = []
    listwith_frombank = [int(toint) for toint in list(np.where(data_frame.loc[:, 'From Bank'] == bank_number)[0])]
    if include_tobank:
        listwith_tobank = [int(toint) for toint in list(np.where(data_frame.loc[:, 'To Bank'] == bank_number)[0])]
    indices_to_get = list(set(listwith_frombank + listwith_tobank))

    bankdata = data_frame.iloc[indices_to_get, :].sort_values('Timestamp')

    X_bankdata = bankdata.drop(columns = ['From Bank', 'To Bank', 'Is Laundering'])
    y_bankdata = bankdata['Is Laundering']

    return X_bankdata, y_bankdata




def data_split_bank_time(df, unique_banks, train_perc = 0.8, test_perc = 0.1):

    test_perc = round(test_perc, 10) / round(1 - train_perc, 10)
    vali_perc = round(1 - test_perc, 10)

    bank_splits = {}
    epochs_per_model = {}

    for index, bank in enumerate(unique_banks):

        X, y = get_bank_data(df, bank)

        num_training = math.ceil(X.shape[0] * train_perc)
        num_training = len(np.where(X.loc[:, 'Timestamp'] <= X.loc[:, 'Timestamp'].iloc[num_training])[0])

        X_train = X.iloc[:num_training,:]
        y_train = y.iloc[:num_training,]

        left_to_test_vali = X.shape[0] - X_train.shape[0]
        num_testing = math.ceil(left_to_test_vali * test_perc)

        if len(X.loc[:, 'Timestamp']) == (num_training + num_testing):
            iloc_value = num_training + num_testing - 1
        else:
            iloc_value = num_training + num_testing

        num_testing = len(np.where(X.loc[:, 'Timestamp'] <= X.loc[:, 'Timestamp'].iloc[iloc_value])[0])

        X_test = X.iloc[num_training:num_testing, :]
        y_test = y.iloc[num_training:num_testing, ]

        X_val = X.iloc[num_testing:] if vali_perc > 0 else None
        y_val = y.iloc[num_testing:] if vali_perc > 0 else None

        bank_splits[bank] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_val': X_val, 'y_val': y_val
        }

        epochs_per_model[bank] = get_epochs(X_train.shape[0])

    return bank_splits, epochs_per_model







dict_clientdata = {}
dict_X = {}
dict_y = {}
banks_lengths = []
epochs_per_model = {}

for index, j in enumerate(unique_banks):

    listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == j)[0])]
    listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == j)[0])]
    indices_to_get = list(set(listwith_frombank + listwith_tobank))
    # indices_to_get = list(set(listwith_frombank))

    #dict_clientdata[f'client_{j}_data'] = df.iloc[np.where(df.loc[:, 'From Bank'] == j)[0], :]
    dict_clientdata[f'bank_{j}_data'] = df.iloc[indices_to_get, :]
    dict_X[f'bank_{j}_X'] = dict_clientdata[f'bank_{j}_data'].drop(columns='Is Laundering')
    dict_y[f'bank_{j}_y'] = dict_clientdata[f'bank_{j}_data'].loc[:, 'Is Laundering']

    bank_data_length = dict_clientdata[f'bank_{j}_data'].shape[0]
    banks_lengths.append(bank_data_length)

    epochs_per_model[f'bank_{j}_epochs'] = (
        250 if bank_data_length > 5000 else
        50 if bank_data_length > 2500 else
        25 if bank_data_length > 1000 else
        10 if bank_data_length > 250 else
        5
    )






for index, bank in enumerate(unique_banks):

    listwith_frombank = [int(toint) for toint in list(np.where(df.loc[:, 'From Bank'] == bank)[0])]
    listwith_tobank = [int(toint) for toint in list(np.where(df.loc[:, 'To Bank'] == bank)[0])]
    indices_to_get = list(set(listwith_frombank + listwith_tobank))
    indices_to_get = list(set(listwith_frombank))

    tmp_bankdata = df.iloc[indices_to_get, :].sort_values('Timestamp')

    num_training = math.ceil(tmp_bankdata.shape[0] * train_perc)
    num_testing = math.floor(tmp_bankdata.shape[0] * test_perc)
    num_validating = tmp_bankdata.shape[0] - num_training - num_testing

    X = tmp_bankdata.drop(columns = 'Is Laundering')
    y = tmp_bankdata['Is Laundering']

    X_train = X.iloc[:num_training,:]
    y_train = y.iloc[:num_training,]

    X_test = X.iloc[num_training:(num_training + num_testing), :]
    y_test = y.iloc[num_training:(num_training + num_testing), ]

    X_val = X.iloc[num_training + num_testing:] if num_validating > 0 else None
    y_val = y.iloc[num_training + num_testing:] if num_validating > 0 else None

    bank_splits[bank] = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'X_val': X_val, 'y_val': y_val
    }

    epochs_per_model[bank] = get_epochs(X_train.shape[0])


















for j in range(number_of_banks):
    dict_clientdata[f'client_{j+1}_data'] = df.iloc[np.where(df.loc[:,'From Bank'] == j)[0], :]
    dict_X[f'client_{j+1}_X'] = dict_clientdata[f'client_{j+1}_data'].drop(columns = 'Is Laundering')
    dict_y[f'client_{j+1}_y'] = dict_clientdata[f'client_{j+1}_data'].loc[:,'Is Laundering']







df1 = pd.read_csv('C:\\Users\\u0168001\\OneDrive - KU Leuven\\Desktop\\Courses\\AML_work_study\\ibm-transactions-for-anti-money-laundering-aml\\versions\\7\\HI-Small_Trans.csv')

len(list(set(df1.loc[:,'From Bank'])))
len(list(set(df1.loc[:,'To Bank'])))
len(list(set(list(set(df1.loc[:,'From Bank'])) + list(set(df1.loc[:,'To Bank'])))))