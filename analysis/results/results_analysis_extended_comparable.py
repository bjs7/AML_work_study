# %%
"""
Extended comparable analysis — GNN baselines + SplitFed + XGBoost models.

Scenario IDs:
  S1 — full_info GNN (R0, BN)               — replicate IBM setup
  S2 — full_info GNN (R1, LN)               — best full-info GNN
  S3 — individual GNN (R1, batching)
  S4 — individual GNN (R1, full batch)
  F1 — FedAvg (C=0.1, E=5)
  P2 — FedProx (mu=0.1)
  V1 — FedGraphSimple (SplitFed)
  B1 — XGBoost full-info (R0, IBM FE)       — analogous to S1
  B2 — XGBoost full-info (R1, std FE)       — analogous to S2
  B3 — XGBoost individual (std FE)
  B4 — SecureBoost (federated XGBoost via SplitFed)

Note: B1-B4 (boosters) are shown throughout this script explicitly — it exists
to compare them against the GNN models. This differs from the headline
baselines table (results_analysis_comparable.py), which excludes boosters by
default until their results are ready (see INCLUDE_BOOST_DEFAULT in scenarios.py).

Two table layouts:
  Layout A — by federation type:
    Oracle (upper bound) : S1, S2, B1, B2
    Individual (lower)   : S3, B3
    Horizontal FL        : F1, P2
    SplitFed              : V1, B4

  Layout B — by model type:
    GNN oracle + individual : S1, S2, S3
    XGBoost oracle + indiv  : B1, B2, B3
    Horizontal FL (GNN)     : F1, P2
    SplitFed (GNN)          : V1
    SplitFed (XGBoost)      : B4
"""

import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study/analysis/lib')

from analysis_functions import (
    assert_paths_exist,
    build_scenario_summary_table,
    build_delta_table,
)
from scenarios import build_scenario_map

EVAL_MODE    = 'comparable'
LOCAL_TABLES = '/home/nam_07/projects/AML_work_study/AML_work_study/analysis/tables'
WRITING_DIR  = '/home/nam_07/projects/AML_work_study/writing/Experimental-Protocol/tables'
TABLES_SUMMARY = f'{WRITING_DIR}/baselines'
TABLES_DELTAS  = f'{WRITING_DIR}/deltas'
CSV_SUMMARY    = f'{LOCAL_TABLES}/baselines'
CSV_DELTAS     = f'{LOCAL_TABLES}/deltas'

# %% ========== Scenario maps ==========

scenario_map_gnn = build_scenario_map(["S1", "S2", "S3", "S4", "F1", "P2", "V1"], eval_mode=EVAL_MODE)

# Boosters use no data flags (→ 'default') or ibm_fe only — not ibm_hp.
# B1: XGBoost full-info (R0) = IBM FE  — analogous to S1 (GNN R0)
# B2: XGBoost full-info (R1) = std FE  — analogous to S2 (GNN R1)
# include_boost=True: this script's whole purpose is comparing boosters to GNNs
# (unlike the headline baselines table, which excludes them by default).
scenario_map_boosters = build_scenario_map(["B1", "B2", "B3", "B4"], eval_mode=EVAL_MODE, include_boost=True)

scenario_map_all = {**scenario_map_gnn, **scenario_map_boosters}

# %% ========== Orderings for the two table layouts ==========

# Layout A — grouped by federation type
order_A = {
    "oracle":     ["S1", "S2", "B1", "B2"],
    "individual": ["S3", "B3"],
    "horiz_fl":   ["F1", "P2"],
    "vert_fl":    ["V1", "B4"],
}
order_A_flat = [s for group in order_A.values() for s in group]

# Layout B — grouped by model type
order_B = {
    "gnn_bounds": ["S1", "S2", "S3"],
    "xgb_bounds": ["B1", "B2", "B3"],
    "horiz_fl":   ["F1", "P2"],
    "vert_gnn":   ["V1"],
    "vert_xgb":   ["B4"],
}
order_B_flat = [s for group in order_B.values() for s in group]

# Layout C — GNN limits | Non-GNN | Federated GNN
order_C = {
    "gnn_limits":  ["S1", "S2", "S3", "S4"],
    "non_gnn":     ["B1", "B2", "B3", "B4"],
    "fed_gnn":     ["F1", "P2", "V1"],
}
order_C_flat = [s for group in order_C.values() for s in group]

# GNN-only order (S1 included for completeness)
order_gnn_only = ["S1", "S2", "S3", "S4", "F1", "P2", "V1"]

# %% ========== Check paths ==========

print("=== GNN scenarios ===")
assert_paths_exist(scenario_map_gnn)

print("\n=== Boosters ===")
assert_paths_exist(scenario_map_boosters)

# %% ========== GNN-only summary table ==========

gnn_df, _ = build_scenario_summary_table(
    scenario_map_gnn,
    out_dir=TABLES_SUMMARY, order=order_gnn_only,
    out_name="gnn_all_comparable",
    csv_dir=CSV_SUMMARY,
)

# %% ========== Booster summary table ==========

booster_df, _ = build_scenario_summary_table(
    scenario_map_boosters,
    out_dir=TABLES_SUMMARY, order=["B1", "B2", "B3", "B4"],
    out_name="boosters_comparable",
    csv_dir=CSV_SUMMARY,
)

# %% ========== Layout A: by federation type ==========
# Groups: Oracle | Individual | Horizontal FL | SplitFed
# Flat version (no \midrule) + grouped version (with \midrule between groups)

layout_A_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_A_flat,
    out_name="all_models_by_federation_comparable",
    csv_dir=CSV_SUMMARY,
)
layout_A_grouped_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_A_flat, groups=order_A,
    out_name="all_models_by_federation_grouped_comparable",
    csv_dir=CSV_SUMMARY,
)
delta_A = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_federation_vs_S2_comparable",
    order=order_A_flat, csv_dir=CSV_DELTAS,
)
delta_A_grouped = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_federation_grouped_vs_S2_comparable",
    order=order_A_flat, groups=order_A, csv_dir=CSV_DELTAS,
)

# %% ========== Layout B: by model type ==========
# Groups: GNN bounds | XGBoost bounds | Horizontal FL GNN | SplitFed GNN | SecureBoost

layout_B_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_B_flat,
    out_name="all_models_by_model_type_comparable",
    csv_dir=CSV_SUMMARY,
)
layout_B_grouped_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_B_flat, groups=order_B,
    out_name="all_models_by_model_type_grouped_comparable",
    csv_dir=CSV_SUMMARY,
)
delta_B = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_model_type_vs_S2_comparable",
    order=order_B_flat, csv_dir=CSV_DELTAS,
)
delta_B_grouped = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_model_type_grouped_vs_S2_comparable",
    order=order_B_flat, groups=order_B, csv_dir=CSV_DELTAS,
)

# %% ========== Layout C: GNN limits | Non-GNN | Federated GNN ==========

layout_C_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_C_flat,
    out_name="all_models_by_paradigm_comparable",
    csv_dir=CSV_SUMMARY,
)
layout_C_grouped_df, _ = build_scenario_summary_table(
    scenario_map_all,
    out_dir=TABLES_SUMMARY, order=order_C_flat, groups=order_C,
    out_name="all_models_by_paradigm_grouped_comparable",
    csv_dir=CSV_SUMMARY,
)
delta_C = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_paradigm_vs_S2_comparable",
    order=order_C_flat, csv_dir=CSV_DELTAS,
)
delta_C_grouped = build_delta_table(
    scenario_map_all, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="deltas_by_paradigm_grouped_vs_S2_comparable",
    order=order_C_flat, groups=order_C, csv_dir=CSV_DELTAS,
)

# %% ========== Delta table (GNN only) ==========

delta_gnn = build_delta_table(
    scenario_map_gnn, baseline_id="S2",
    out_dir=TABLES_DELTAS, out_name="gnn_deltas_vs_S2_comparable",
    order=order_gnn_only, csv_dir=CSV_DELTAS,
)

# %% ========== Pattern recall — handled in results_analysis_why_fullinfo_comparable.py ==========

# %%
