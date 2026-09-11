# %%

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis')

from lib.analysis_functions import (
    assert_paths_exist,
    build_scenario_summary_table,
    build_delta_table,
)
from lib.scenarios import build_scenario_map

EVAL_MODE   = 'comparable'
LOCAL_TABLES = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables'
WRITING_DIR = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol'
TABLES_BASELINES    = f'{WRITING_DIR}/tables/baselines'
TABLES_SENSITIVITY  = f'{WRITING_DIR}/tables/sensitivity'
TABLES_HETEROGENEITY = f'{WRITING_DIR}/tables/heterogeneity'
FIGS_BASELINES      = f'{WRITING_DIR}/figs/baselines'
FIGS_SENSITIVITY    = f'{WRITING_DIR}/figs/sensitivity'
FIGS_HETEROGENEITY  = f'{WRITING_DIR}/figs/heterogeneity'
CSV_BASELINES    = f'{LOCAL_TABLES}/baselines'
CSV_SENSITIVITY  = f'{LOCAL_TABLES}/sensitivity'
CSV_HETEROGENEITY = f'{LOCAL_TABLES}/heterogeneity'

# %% ========== Scenario maps ==========

scenario_map_gnn = build_scenario_map(["S1", "S2", "S3", "S4", "F1", "P2", "V1"], eval_mode=EVAL_MODE)

# Boosters kept separate; loaded explicitly (include_boost=True) since path-checking
# and the dedicated booster table below still need them even though the headline
# baselines table excludes them by default (see INCLUDE_BOOST_DEFAULT in scenarios.py).
scenario_map_boosters = build_scenario_map(["B1", "B2", "B3", "B4"], eval_mode=EVAL_MODE, include_boost=True)

scenario_map_fedavg_sens = build_scenario_map(
    ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"], eval_mode=EVAL_MODE)
scenario_map_fedprox_mu = build_scenario_map(["P1", "P2", "P3"], eval_mode=EVAL_MODE)
scenario_map_heterogeneity = build_scenario_map(
    ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"], eval_mode=EVAL_MODE)

# GNN-only by default (boosters excluded — see INCLUDE_BOOST_DEFAULT in scenarios.py).
# Flip to `{**scenario_map_gnn, **scenario_map_boosters}` once B-series results are
# good enough to show in the headline table again.
scenario_map_all = scenario_map_gnn

# %% ========== Orderings ==========

# Layout C — GNN limits | Non-GNN | Federated GNN
order_C = {
    "gnn_limits": ["S1", "S2", "S3", "S4"],
    "non_gnn":    ["B1", "B2", "B3", "B4"],
    "fed_gnn":    ["F1", "P2", "V1"],
}
order_C_flat = [s for group in order_C.values() for s in group]

fedavg_order  = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
fedprox_order = ["P1", "P2", "P3"]
hetero_order  = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]

# %% ========== Check paths ==========

assert_paths_exist(scenario_map_all)
assert_paths_exist(scenario_map_boosters)

# %%
assert_paths_exist(scenario_map_fedavg_sens)
assert_paths_exist(scenario_map_fedprox_mu)
assert_paths_exist(scenario_map_heterogeneity)

# %% ========== Baseline summary table — Layout C (paradigm grouped) ==========
# Boosters (B1-B4) excluded from this headline table by default — see
# INCLUDE_BOOST_DEFAULT in scenarios.py. The "non_gnn" group in order_C
# will simply be dropped (empty group) until boosters are re-enabled.

baselines_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_BASELINES, order=order_C_flat, groups=order_C,
    out_name="baselines_grouped_comparable",
    csv_dir=CSV_BASELINES,
)

# %% ========== Sensitivity tables ==========

fedavg_df, _ = build_scenario_summary_table(
    scenario_map_fedavg_sens,
    out_dir=TABLES_SENSITIVITY, order=fedavg_order,
    out_name="fedavg_sensitivity_comparable",
    csv_dir=CSV_SENSITIVITY,
)

fedprox_df, _ = build_scenario_summary_table(
    {"F1": scenario_map_gnn["F1"], **scenario_map_fedprox_mu},
    out_dir=TABLES_SENSITIVITY, order=["F1"] + fedprox_order,
    out_name="fedprox_comparable",
    csv_dir=CSV_SENSITIVITY,
)

# %% ========== Heterogeneity table ==========

hetero_df, _ = build_scenario_summary_table(
    {"F1": scenario_map_gnn["F1"], **scenario_map_heterogeneity},
    out_dir=TABLES_HETEROGENEITY, order=["F1"] + hetero_order,
    out_name="heterogeneity_comparable",
    csv_dir=CSV_HETEROGENEITY,
)

# %%
