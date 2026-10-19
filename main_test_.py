# %% 
# packages
import utils
import logging
import copy
import data.fl_data_helpers as dfn
from federated_learning.fl_base import Manager
import federated_learning.fl_algos  # noqa: F401 — side-effect import, populates FL algorithm registry
import models.gnn_models  # noqa: F401 — side-effect import, registers GNN architectures
import models.regression_models  # noqa: F401 — side-effect import, registers regression models
from result_io.save_results import save_results



import argparse
import logging
import os
import sys
import numpy as np
import torch
import random
import pandas as pd
import copy

from data.raw_data_processing import get_data
from configs.configs import split_perc
from configs.paths import get_data_path
from federated_learning.fl_base import Manager


utils.logger_setup()
parsers = utils.parser_all()
#parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)

parsers['fl_parser'].model = 'GINe'
parsers['fl_parser'].fl_algo = 'FedGraphSimple'
parsers['data_parser'].eval_mode = 'system'
parsers['data_parser'].ibm_hp = True
data_parser = parsers['data_parser']


test123 = pd.read_csv(f"{get_data_path()}/AML_work_study/data/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

df = pd.read_csv(f"{get_data_path()}/AML_work_study/data/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc=split_perc)

# %%
