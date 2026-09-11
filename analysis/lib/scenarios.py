"""Shared scenario registry for analysis/pattern_analysis and analysis/results scripts.

Single source of truth for the S/F/P/V/B scenario IDs (kwargs to get_experiment_path
+ display names), replacing the near-identical `scenarios = {...}` dict that used to
be copy-pasted in every script. Also centralizes the "vertical" -> "SplitFed" display
name via result_io.load_results.ALGO_NAME_ALIASES, and the boost (B-series) opt-in
toggle (see INCLUDE_BOOST_DEFAULT below).

Usage:
    from scenarios import build_scenario_map, DEFAULT_SCENARIO_IDS
    scenarios = build_scenario_map(DEFAULT_SCENARIO_IDS, eval_mode='comparable')
"""

import sys
from pathlib import Path
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
sys.path.append(str(Path(__file__).resolve().parent))  # analysis/lib — needed regardless of how the caller set up sys.path

from analysis_functions import get_experiment_path
from result_io.load_results import ALGO_NAME_ALIASES

# new (public) name -> old (internal folder / fl_algo) name, e.g. {'SplitFed': 'FedGraphSimple', ...}
# invert it: internal name -> public display name, e.g. {'FedGraphSimple': 'SplitFed', ...}
_ALGO_DISPLAY = {old: new for new, old in ALGO_NAME_ALIASES.items()}


# ---- core (non-boost) scenario definitions: id -> get_experiment_path kwargs ----
SCENARIO_KWARGS = {
    "S1": dict(fl_algo="full_info", model="GINe",
               batching=True, batchnorm=True, ibm_fe=True, ibm_hp=True, emlps=True),
    "S2": dict(fl_algo="full_info", model="GINe",
               batching=True, ibm_hp=True, emlps=True),
    "S3": dict(fl_algo="individual", model="GINe",
               batching=True, ibm_hp=True, emlps=True),
    "S4": dict(fl_algo="individual", model="GINe",
               ibm_hp=True, emlps=True),
    "F1": dict(fl_algo="FedAvg", model="GINe",
               batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5),
    "P2": dict(fl_algo="FedProx", model="GINe",
               batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, mu=0.1),
    "V1": dict(fl_algo="FedGraphSimple", model="GINe",
               batching=True, ibm_hp=True, emlps=True),
}

SCENARIO_NAMES = {
    "S1": "Full-info GNN (R0, BN)",
    "S2": "Full-info GNN (R1, LN)",
    "S3": "Individual GNN (batching)",
    "S4": "Individual GNN (full batch)",
    "F1": "FedAvg (GNN)",
    "P2": "FedProx (mu=0.1)",
    "V1": f"FedGraphSimple ({_ALGO_DISPLAY.get('FedGraphSimple', 'FedGraphSimple')})",
}

# --- FedAvg sensitivity (F0, F2-F10; F1 is the shared baseline above) ---
_FEDAVG_SENS_KWARGS = {
    "F0": dict(fl_algo="FedAvg", model="GINe", ibm_hp=True, emlps=True,
               weighting="uniform", client_fraction=1.0, num_local_epochs=1),
    "F2": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               batch_size=4096, client_fraction=0.1, num_local_epochs=5),
    "F3": dict(fl_algo="FedAvg", model="GINe", ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5),
    "F4": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=1),
    "F5": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=10),
    "F6": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=25),
    "F7": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.25, num_local_epochs=5),
    "F8": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.50, num_local_epochs=5),
    "F9": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               weighting="uniform", client_fraction=0.1, num_local_epochs=5),
    "F10": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
                client_fraction=0.1, num_local_epochs=5, num_rounds=50),
}
_FEDAVG_SENS_NAMES = {
    "F0": "FedAvg (FedSGD-like)", "F2": "FedAvg (B=4096)", "F3": "FedAvg (full batch)",
    "F4": "FedAvg (E=1)", "F5": "FedAvg (E=10)", "F6": "FedAvg (E=25)",
    "F7": "FedAvg (C=0.25)", "F8": "FedAvg (C=0.50)", "F9": "FedAvg (uniform)",
    "F10": "FedAvg (T=50)",
}

# --- FedProx mu sensitivity (P1, P3; P2 is the shared baseline above) ---
_FEDPROX_MU_KWARGS = {
    "P1": dict(fl_algo="FedProx", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, mu=0.01),
    "P3": dict(fl_algo="FedProx", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, mu=1.0),
}
_FEDPROX_MU_NAMES = {"P1": "FedProx (mu=0.01)", "P3": "FedProx (mu=1.0)"}

# --- Heterogeneity experiments (H1-H8), all FedAvg with one factor varied ---
_HETEROGENEITY_KWARGS = {
    "H1": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, bank_filter="no_top10"),
    "H2": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, bank_filter="no_top1"),
    "H3": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, bank_filter="no_bottom10"),
    "H4": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, bank_filter="no_bottom5pct"),
    "H5": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, loss_ratio=1),
    "H6": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, loss_ratio=980),
    "H7": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, loss_ratio=50),
    "H8": dict(fl_algo="FedAvg", model="GINe", batching=True, ibm_hp=True, emlps=True,
               client_fraction=0.1, num_local_epochs=5, normalize_currency=True),
}
_HETEROGENEITY_NAMES = {
    "H1": "FedAvg (no top-10)", "H2": "FedAvg (no top-1)", "H3": "FedAvg (no bottom-10)",
    "H4": r"FedAvg (no bottom-5\%)", "H5": "FedAvg (loss [1,1])", "H6": "FedAvg (loss [1,980])",
    "H7": "FedAvg (loss [1,50])", "H8": "FedAvg (normalize currency)",
}

# Fold the sensitivity/heterogeneity variants into the main lookup tables so
# build_scenario_map resolves F0/F2-10, P1/P3 and H1-H8 the same way as the
# core scenarios above.
SCENARIO_KWARGS.update(_FEDAVG_SENS_KWARGS)
SCENARIO_KWARGS.update(_FEDPROX_MU_KWARGS)
SCENARIO_KWARGS.update(_HETEROGENEITY_KWARGS)
SCENARIO_NAMES.update(_FEDAVG_SENS_NAMES)
SCENARIO_NAMES.update(_FEDPROX_MU_NAMES)
SCENARIO_NAMES.update(_HETEROGENEITY_NAMES)

# ---- boost (B-series, XGBoost/SecureBoost) scenarios kept separate ----
# Results aren't good enough yet to show by default (see INCLUDE_BOOST_DEFAULT) but
# the definitions stay here, verbatim, so re-enabling them later is a one-line flip.
BOOST_SCENARIO_KWARGS = {
    "B1": dict(fl_algo="full_info", model="xgboost", ibm_fe=True),
    "B2": dict(fl_algo="full_info", model="xgboost"),
    "B3": dict(fl_algo="individual", model="xgboost"),
    "B4": dict(fl_algo="SecureBoost", model="xgboost"),
}
BOOST_SCENARIO_NAMES = {
    "B1": "XGBoost full-info (R0)",
    "B2": "XGBoost full-info (R1)",
    "B3": "XGBoost individual (std FE)",
    "B4": "SecureBoost",
}

# Flip to True once B-series results are good enough to show by default.
# Scripts whose whole purpose is displaying boosters should pass include_boost=True
# explicitly at the call site instead of relying on this default.
INCLUDE_BOOST_DEFAULT = False

DEFAULT_SCENARIO_IDS = ["S2", "F1", "P2", "V1"]


def build_scenario_map(ids, eval_mode, testing=False, include_boost=None):
    """Resolve {id: {'name': ..., 'path': ...}} for the given scenario ids + eval_mode.

    include_boost: None (default) -> use INCLUDE_BOOST_DEFAULT to silently drop any
    boost ids present in `ids`; True/False force-include/exclude explicitly.
    """
    inc_boost = INCLUDE_BOOST_DEFAULT if include_boost is None else include_boost
    all_kwargs = {**SCENARIO_KWARGS, **(BOOST_SCENARIO_KWARGS if inc_boost else {})}
    all_names = {**SCENARIO_NAMES, **(BOOST_SCENARIO_NAMES if inc_boost else {})}

    out = {}
    for sid in ids:
        if sid not in all_kwargs:
            continue
        out[sid] = {
            "name": all_names[sid],
            "path": get_experiment_path(eval_mode=eval_mode, testing=testing, **all_kwargs[sid]),
        }
    return out
