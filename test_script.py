# Test script for feature_engineering in GNNMixinParty

import pandas as pd
import numpy as np
import copy
import torch
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
from data.get_indices_type_data import get_indices_bdt, get_graph_data
from data.relevant_banks import load_relevant_banks
import data.feature_engi as fe
import data.data_functions as dfn
from federated_learning.fl_base import Manager

# =============================================================================
# SETUP - Create real manager with parties
# =============================================================================

utils.logger_setup()
parsers = utils.parser_all()
parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
parsers['data_parser'].add_ids = False
parsers['data_parser'].batching = True
parsers['fl_parser'].fl_algo = 'FedAvg'
parsers['fl_parser'].client_fraction = 0.25
parsers['fl_parser'].num_local_epochs = 10
utils.set_seed(parsers['data_parser'].seed, True)

# Load data
print("Loading data...")
df_raw = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")
df, scaler_encoders = get_data(df_raw, parsers['data_parser'], split_perc=split_perc)

# Create laundering values
laundering_values_vali, laundering_values_test = dfn.prep_laundering_dfs(parsers['data_parser'], copy.deepcopy(df))

# Create manager and setup parties
print("Setting up manager and parties...")
manager = Manager.get_algo_class(parsers)
tuned_hp = manager.setup_parties(df, parsers, scaler_encoders, laundering_values_vali)

# Set mode for feature engineering
manager.set_mode('training')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_data_summary(data_dict, label):
    """Print summary of a data dict"""
    if data_dict is None or data_dict.get('df') is None:
        print(f"    {label}: None")
        return
    df = data_dict['df']
    n_edges = df.edge_index.shape[1] if hasattr(df, 'edge_index') else 0
    n_nodes = df.x.shape[0] if hasattr(df, 'x') else 0
    y_sum = df.y.sum().item() if hasattr(df, 'y') else 0
    print(f"    {label}: nodes={n_nodes}, edges={n_edges}, y_sum={y_sum}")


def print_party_data(party, label=""):
    """Print summary of party's data"""
    print(f"\n  {label} - Party data (self.data):")
    print_data_summary(party.data.get('train_data'), "train_data")
    print_data_summary(party.data.get('vali_data'), "vali_data")
    print_data_summary(party.data.get('test_data'), "test_data")


def print_party_indices(party, label=""):
    """Print party indices summary"""
    indices = party.indices
    print(f"  {label} - Indices: train={len(indices['train_indices'])}, vali={len(indices['vali_indices'])}, test={len(indices['test_indices'])}")


def test_feature_engineering(party, scenario_name):
    """Test feature_engineering on a party"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")

    print_party_indices(party, "Before")
    print_party_data(party, "Before")

    try:
        # This calls feature_engineering internally
        party.prep_data()

        print(f"\n  After prep_data() - Processed data (self.procs_data):")
        print_data_summary(party.procs_data.get('train_data'), "train_data")
        print_data_summary(party.procs_data.get('vali_data'), "vali_data")
        print_data_summary(party.procs_data.get('test_data'), "test_data")

        # Check edge_attr standardization
        for split in ['train_data', 'vali_data', 'test_data']:
            procs = party.procs_data.get(split)
            if procs and procs.get('df') is not None and procs['df'].edge_attr.shape[0] > 0:
                edge_attr = procs['df'].edge_attr
                start = party.edge_feat_start
                if edge_attr.shape[0] > 1:
                    mean = edge_attr[:, start:].mean(dim=0)
                    std = edge_attr[:, start:].std(dim=0)
                    print(f"    {split} edge_attr[{start}:] - mean~{mean[:3].tolist()}, std~{std[:3].tolist()}")

        print(f"\n  SUCCESS")
        return True

    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def modify_party_indices(party, train_n=None, vali_n=None, test_n=None):
    """Modify party indices to create test scenarios.

    Args:
        train_n: Number of train indices to keep (None = keep all, 0 = empty)
        vali_n: Number of vali indices to keep
        test_n: Number of test indices to keep
    """
    # Store original indices if not already stored
    if not hasattr(party, '_original_indices'):
        party._original_indices = copy.deepcopy(party.indices)

    # Restore original first
    party.indices = copy.deepcopy(party._original_indices)

    # Modify
    if train_n is not None:
        party.indices['train_indices'] = party.indices['train_indices'][:train_n]
    if vali_n is not None:
        party.indices['vali_indices'] = party.indices['vali_indices'][:vali_n]
    if test_n is not None:
        party.indices['test_indices'] = party.indices['test_indices'][:test_n]

    # Rebuild party.data based on new indices
    # This mimics what happens in party setup
    rebuild_party_data(party)


def rebuild_party_data(party):
    """Rebuild party.data based on current indices"""
    bank_indices = party.indices

    # Get graph data for this party's indices
    train_data, vali_data, test_data = get_graph_data(
        party.args,
        party.manager.df['graph_data'],
        bank_indices
    )

    party.data = {
        'train_data': train_data,
        'vali_data': vali_data,
        'test_data': test_data
    }


# =============================================================================
# GET A TEST PARTY
# =============================================================================

# Get first party from self.parties (train party - has data in all splits)
test_bank_id = list(manager.parties.keys())[0]
test_party = manager.parties[test_bank_id]

print(f"\nUsing party for bank {test_bank_id} as test subject")
print_party_indices(test_party, "Original")

# =============================================================================
# TEST SCENARIOS
# =============================================================================

# Scenario 1: Normal case - all data present
modify_party_indices(test_party, train_n=None, vali_n=None, test_n=None)
test_feature_engineering(test_party, f"Bank {test_bank_id}: Normal (all data)")

# Scenario 2: Only 1 data point in train
modify_party_indices(test_party, train_n=1, vali_n=10, test_n=10)
test_feature_engineering(test_party, f"Bank {test_bank_id}: 1 train, 10 vali, 10 test")

# Scenario 3: Only 1 data point in each split
modify_party_indices(test_party, train_n=1, vali_n=1, test_n=1)
test_feature_engineering(test_party, f"Bank {test_bank_id}: 1 in each split")

# Scenario 4: Empty train
modify_party_indices(test_party, train_n=0, vali_n=10, test_n=10)
test_feature_engineering(test_party, f"Bank {test_bank_id}: NO train (0), 10 vali, 10 test")

# Scenario 5: Empty vali
modify_party_indices(test_party, train_n=10, vali_n=0, test_n=10)
test_feature_engineering(test_party, f"Bank {test_bank_id}: 10 train, NO vali (0), 10 test")

# Scenario 6: Only test data
modify_party_indices(test_party, train_n=0, vali_n=0, test_n=10)
test_feature_engineering(test_party, f"Bank {test_bank_id}: ONLY test (0, 0, 10)")

# Scenario 7: Small amounts (2 each)
modify_party_indices(test_party, train_n=2, vali_n=2, test_n=2)
test_feature_engineering(test_party, f"Bank {test_bank_id}: 2 in each split")

# Scenario 8: Completely empty (edge case)
modify_party_indices(test_party, train_n=0, vali_n=0, test_n=0)
test_feature_engineering(test_party, f"Bank {test_bank_id}: Completely empty")

# =============================================================================
# RESTORE ORIGINAL
# =============================================================================

modify_party_indices(test_party, train_n=None, vali_n=None, test_n=None)
print(f"\n{'='*70}")
print("TESTS COMPLETE - Party restored to original state")
print(f"{'='*70}")
