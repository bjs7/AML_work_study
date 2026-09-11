# %%

"""
Data heterogeneity analysis — how unevenly are the FL parties' data distributed?

Mirrors the four subsections of the "Detection of Heterogeneity - System
Evaluation" section in data_heterogeneity_holder.tex (writing repo):
  1. Quantity Skew                                -> heterogeneity/quantity_skew.py
  2. Label Distribution Skew                       -> heterogeneity/label_skew.py
  3. Laundering-pattern & Feature Covariate Shift  -> heterogeneity/pattern_covariate_shift.py
  4. Inter-bank vs. Intra-bank                     -> heterogeneity/inter_intra_bank.py

Each module's run() produces both the figures the original (pre-refactor)
script drew, and a small summary table (.csv + .tex) with the numbers the
corresponding subsection's prose needs to cite — the original script only
produced figures, so there was previously no way to put concrete numbers in
the main text.

This orchestrator does the one-time (expensive) FL pipeline setup, builds the
shared stats_df/ii_df, then calls each subsection module in turn.
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/data/heterogeneity')

import copy
import pandas as pd

import utils
from configs.configs import split_perc
from data.raw_data_processing import get_data
from data.relevant_banks import load_relevant_banks
import data.fl_data_helpers as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, GNN_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
import models.gnn_models
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos

import stats as het_stats
import quantity_skew
import label_skew
import pattern_covariate_shift
import inter_intra_bank


# %% ========== Data Loading ==========

utils.logger_setup()
parsers = utils.parser_all()
utils.set_seed(parsers['data_parser'].seed, True)

parsers['data_parser'].ibm_hp = True
parsers['fl_parser'].fl_algo = 'FedAvg'

df = pd.read_csv(
    f"{utils.get_data_path()}/AML_work_study/formatted_transactions_"
    f"{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv"
)

df, scaler_encoders = get_data(df, parsers['data_parser'], split_perc=split_perc)
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))
manager = Manager.get_algo_class(parsers)
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali, analysis=True)

relevant_banks = load_relevant_banks(parsers['data_parser']).get(parsers['fl_parser'].fl_algo)
print(f"Train banks (relevant_banks): {len(relevant_banks['train_banks'])} | "
      f"Parties set up: {len(manager.parties)}")


# %% ========== Build shared stats (single source of truth — see heterogeneity/stats.py) ==========

DATA_STR = 'train_data'
parties = manager.parties

stats_df, pattern_cols, currency_cols, payment_cols = het_stats.compute_bank_stats(
    parties, df, scaler_encoders, data_str=DATA_STR
)

train_df = df['regular_data'][DATA_STR]['x']
ii_df = het_stats.compute_inter_intra_stats(train_df, set(parties.keys()))


# %% ========== 1. Quantity Skew ==========

quantity_skew.run(stats_df)


# %% ========== 2. Label Distribution Skew ==========

label_skew.run(stats_df)


# %% ========== 3. Laundering-pattern & Feature Covariate Shift ==========

pattern_covariate_shift.run(stats_df, pattern_cols, currency_cols, payment_cols, parties, data_str=DATA_STR)


# %% ========== 4. Inter-bank vs. Intra-bank ==========

inter_intra_bank.run(ii_df)

# %%
