# %%

import os
import utils
import logging
import argparse
import pandas as pd
import configs
import inference_saving.save_load_models as slm
import numpy as np
import copy
from relevant_banks import get_relevant_banks
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
from pathlib import Path
import pandas as pd
from IPython.display import display



# IMPORTTANT HAVE F1 SCORE FOR FULL, BOTH FOR THE INDICES WHERE ONLY INDIVIDUAL BANKS HAVE DATA
# BUT ALSO FOR ONE WHERE ALL INFO IS USED, BECAUSE FULL INFO DOES HAVE ACCESS TO THIS!


# %%
# get arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
#args.model = 'xgboost'


# %%
infer = InferenceModel.from_model_type(args)


# %%
# Load the JSON file
file_path = Path(infer.main_folder) / 'inference_results.json'
with open(file_path, 'r') as f:
    results = json.load(f)


# %% 
# Convert back to DataFrames
laundering_values = pd.DataFrame(**results['laundering_values'])
f1_values = pd.DataFrame(**results['f1_values'])


# %%

f1_score_full_info = f1_score(laundering_values['true y'], laundering_values['predicted_full_info'], average='binary')
f1_score_individual_banks = f1_score(laundering_values['true y'], laundering_values['predicted_individual_banks'], average='binary')
avg_f1_indi = np.average(f1_values.iloc[:,1])


print(f"F1 Score (Full Info): {f1_score_full_info:.4f}")
print(f"F1 Score (Individual Banks): {f1_score_individual_banks:.4f}")
print(f"Average F1 Score (Individual Banks): {avg_f1_indi:.4f}")


# %% 
y_one_indices = np.where(laundering_values['true y'] == 1)[0]
num_full_found = sum(laundering_values['predicted_full_info'][y_one_indices] == 1)
num_indi_found = sum(laundering_values['predicted_individual_banks'][y_one_indices] == 1)

sum(laundering_values['predicted_full_info'])
sum(laundering_values['predicted_individual_banks'])




percentage_full = num_full_found / len(y_one_indices)
percentage_indi = num_indi_found / len(y_one_indices)


print(f'Number of illicit transactions found full info: {num_full_found}, percentage: {percentage_full:.4f}')
print(f'Number of illicit transactions found individual banks: {num_indi_found}, percentage: {percentage_indi:.4f}')




# %%
# Print results in a formatted way
print("\nF1 Values DataFrame:")
try:
    display(f1_values)  # For Jupyter notebook or VS Code with IPython support
except NameError:
    print(f1_values)  # Fallback for plain Python console









"""

save_direc = infer.main_folder  # Assuming infer.main_folder is defined
folder_path = Path(save_direc)
file_path = folder_path / 'inference_results.json'
with open(file_path, 'r') as f:
    results = json.load(f)

"""
