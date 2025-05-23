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
from pathlib import Path
import pandas as pd
from IPython.display import display


# %%
# get arguments
parser = get_parser()
args, unknown = parser.parse_known_args()
infer = InferenceModel.from_model_type(args)

# %%
save_direc = infer.main_folder  # Assuming infer.main_folder is defined
folder_path = Path(save_direc)
file_path = folder_path / 'inference_results.json'
with open(file_path, 'r') as f:
    results = json.load(f)

# %%
# Extract individual results
f1_score_full_info = results['f1_score_full_info']
f1_score_individual_banks = results['f1_score_individual_banks']
f1_values = pd.DataFrame(results['f1_values'])  # Convert back to DataFrame

# %%
# Print results in a formatted way
print("=== Inference Results ===")
print(f"F1 Score (Full Info): {f1_score_full_info:.4f}")
print(f"F1 Score (Individual Banks): {f1_score_individual_banks:.4f}")
print("\nF1 Values DataFrame:")
try:
    display(f1_values)  # For Jupyter notebook or VS Code with IPython support
except NameError:
    print(f1_values)  # Fallback for plain Python console



# %%
