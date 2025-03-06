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

