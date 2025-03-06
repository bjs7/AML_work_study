import math
import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\u0168001\\OneDrive - KU Leuven\\Desktop\\Courses\\AML_work_study\\Multi-GNN\\formatted_transactions.csv')
lBanks = list((df.loc[:, 'From Bank'])) + list((df.loc[:, 'To Bank']))
unique_banks = list(set(lBanks))
unique_banks = unique_banks[0:100]
number_of_banks = len(unique_banks)

# Get Data -------------------------------------------------------------------------------------------------------------

#data =  get_data(df, split_perc = [0.6, 0.2])

data =  get_data(df, split_perc = [0.75, 0.25])
data['graph_data'].keys()

# set model and parameters ---------------------------------------------------------------------------------------------

model = xgb.Booster

# define training parameters
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6, 'learning_rate': 0.1}
num_rounds = 100


# set model and parameters ---------------------------------------------------------------------------------------------

model = GINe

train_model(model, data, [1])








# train model

# maybe currency and payment format should not be standardized? But should have one-hot-encoding?
#train per bank
test123 = train_model(model, data, [2,3], params = params, num_rounds = num_rounds)

# train on all data
test1234 = train_model(model, data, params = params, num_rounds = num_rounds)








import data_processing as dp
import data_utils

# define model

model = xgb.Booster

# define training parameters
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 3, 'learning_rate': 0.1}
num_rounds = 100


# Get data
unique_banks = unique_banks[0:10]

#data_features = ['from_id', 'to_id', 'Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
data_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
test123 = dp.get_data(df, data_features, unique_banks = [], split_perc = [0.6, 0.2], graph_data = True)
test123[99]


# train model



train_model(model, banks, banks_splits, params = params, num_rounds = num_rounds)


trainer_class = trainer_functions.get(model.__name__)
trained_models = {}

data = get_data(df, data_features, unique_banks = [], split_perc = [0.6, 0.2], graph_data = True)

train_model(model, data, entity = 1)


test123[0].keys


trainer_class(model(), banks_splits[bank]['X_train'], banks_splits[bank]['y_train'], **kwargs)



