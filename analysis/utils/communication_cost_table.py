# %%
"""
Communication cost analysis — horizontal (FedAvg/FedProx) vs. SplitFed FL.

Computes theoretical bytes exchanged during training for each scenario × eval-mode
combination, based on:
  - Model architecture: W (parameters) derived from hyper_parameters.json of actual runs
  - relevant_banks JSON: K (party pool sizes for system vs. comparable)
  - Batch scripts: T (rounds/epochs), client_fraction, batch_size
  - Architecture: embed_dim for V1 embedding exchange

FedAvg/FedProx (F1, P2):
  Each round, K selected parties send model weights to manager + receive aggregated
  weights back. send_local_weights() uses gnn.named_parameters() — all GINe params
  including mlp_vert (unused in FedAvg inference but present in the model).
    per_exchange = W_FEDAVG × 4 bytes
    n_exchanges  = 2 × K × T          (upload + download, T rounds)
    total        = per_exchange × n_exchanges

V1 (SplitFed):
  Each batch, each party whose transactions appear in the batch sends per-edge
  embeddings to the manager (forward pass); manager sends gradients back (backward).
  Not every transaction is necessarily seen by two parties in the same batch — a
  transaction may only have one party active. The worst-case bound assumes both
  parties contribute for every edge in the batch.
    per_batch      = batch_size × 2 × embed_dim × 4 bytes   (both parties, worst case)
    n_exchanges    = 2 × n_batches × T                       (forward + backward)
    total          = per_batch × n_exchanges
  Which simplifies to:
    total = n_train_transactions × T × 2 × embed_dim × 4 × 2

Note on transaction counts: the 60/20/20 split cuts at end-of-day boundaries, so
obs_ava × 0.6 is an approximation; actual counts differ by at most a handful of batches.

Note: FedAvg uses batching=False → GINe BatchNorm has no running-stats buffers;
named_parameters() and state_dict() cover the same set of values.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import get_data_path
from models.gnn_models import GINe

# =============================================================================
# ====================== CONSTANTS FROM TRAINING SETUP ========================
# =============================================================================

BATCH_SIZE        = 8192
CLIENT_FRACTION   = 0.1
T_FEDAVG          = 100          # default --num_rounds
T_V1_COMPARABLE   = 50           # run_vertical_simple_comparable.sh
T_V1_SYSTEM       = 20           # run_vertical_simple_system.sh
BYTES_PER_FLOAT   = 4            # float32


# =============================================================================
# ==================== MODEL PARAMETER COUNT (W) ==============================
# =============================================================================
# Parameters counted by instantiating GINe from the actual experiment's
# hyper_parameters.json (available locally; model.pth lives only on HPC).
#
# Input dimensions from feature engineering (ibm_fe=False, ibm_hp=True):
#   num_features = 1 (single node-level feature)
#   edge_dim     = 41 (4 continuous + OHE of two currency columns + OHE of
#                  payment format; confirmed from edge_emb.weight shape in
#                  comparable FedGraphSimple/GINe__emlps/batching__ibm_hp run)
#
# Architecture (same for F1/P2/V1 — same hyper_parameters.json):
#   n_hidden=66, num_gnn_layers=2, edge_updates=True (GINe__emlps), batching=False
#
# FedAvg send_local_weights() calls gnn.named_parameters(), which covers all
# GINe params — including mlp_vert (manager head, unused for FedAvg inference
# but present in the instantiated model and therefore communicated).
#
# V1 parties exchange per-edge 198-dim embeddings, not weights.
# The manager's mlp_vert (21,177 params) stays server-side and is never sent.

EXPERIMENTS = Path(get_data_path()) / 'AML_work_study/experiments'

# Load hyper_parameters.json from the actual comparable experiment (architecture
# is identical for system runs — same hidden_embedding_size and num_gnn_layers).
_HP_F1 = json.load(open(
    EXPERIMENTS / 'small_HI/split_0.6_0.2/comparable'
    / 'FedAvg/proportional_C0.1_E5/GINe__emlps/batching__ibm_hp'
    / '20260306_223047/hyper_parameters.json'
))

# Instantiate with the same arguments used in _create_gnn_model (gnn_base.py):
# batching=False for FedAvg/FedProx (no BatchNorm running stats).
_model = GINe(
    num_features=1,
    num_gnn_layers=_HP_F1['num_gnn_layers'],
    n_hidden=_HP_F1['hidden_embedding_size'],
    edge_dim=41,
    edge_updates=True,   # GINe__emlps
    batching=False,
)
_state = _model.state_dict()

# W_FEDAVG: all params sent by send_local_weights() via named_parameters()
# = full GINe including mlp_vert (present in model, communicated even though
#   gradients never flow through it in FedAvg training).
W_FEDAVG    = sum(v.numel() for v in _state.values())
W_V1_PARTY  = sum(v.numel() for k, v in _state.items() if not k.startswith('mlp_vert'))
W_V1_MANAGER = sum(v.numel() for k, v in _state.items() if k.startswith('mlp_vert'))
test1 = sum(v.numel() for k, v in _state.items() if k.startswith('mlp.'))
W_V1_TOTAL   = W_FEDAVG   # same model class; total = party + manager

# embed_dim per party = mlp_vert first-layer input (396) / 2 parties
EMBED_DIM = _state['mlp_vert.0.weight'].shape[1] // 2   # 198

print(f"F1/P2  W (all params via named_parameters): {W_FEDAVG:,}")
print(f"V1     W total: {W_V1_TOTAL:,}  |  party (GNN): {W_V1_PARTY:,}  |  manager (mlp_vert): {W_V1_MANAGER:,}")
print(f"Embed dim per party (V1): {EMBED_DIM}")


# =============================================================================
# ==================== LOAD PARTY COUNTS FROM RELEVANT BANKS ==================
# =============================================================================

RB_PATH = (
    Path(get_data_path())
    / 'AML_work_study/experiments/relevant_banks/small_HI__split_0.6_0.2.json'
)
with open(RB_PATH) as f:
    rb = json.load(f)

# FedAvg: train_banks = pool from which K = client_fraction × pool are sampled each round
K_FEDAVG_SYSTEM_POOL     = len(rb['FedAvg']['train_banks'])     # 1,093
K_FEDAVG_COMPARABLE_POOL = len(rb['individual']['banks'])       # 630

K_FEDAVG_SYSTEM     = round(K_FEDAVG_SYSTEM_POOL     * CLIENT_FRACTION)  # 109
K_FEDAVG_COMPARABLE = round(K_FEDAVG_COMPARABLE_POOL * CLIENT_FRACTION)  # 63

# V1 (SplitFed): all banks with data in the train split participate per epoch.
# System: all banks appearing in any train-split transaction (FedGraph.train_banks).
# Comparable: the 630 comparable-eligible banks.
K_V1_SYSTEM     = len(rb['FedGraph']['train_banks'])   # 30,528
K_V1_COMPARABLE = len(rb['individual']['banks'])       # 630

print(f"\nFedAvg K pool  — system: {K_FEDAVG_SYSTEM_POOL:,}, comparable: {K_FEDAVG_COMPARABLE_POOL:,}")
print(f"FedAvg K/round — system: {K_FEDAVG_SYSTEM}, comparable: {K_FEDAVG_COMPARABLE} (fraction={CLIENT_FRACTION})")
print(f"V1 K (all)     — system: {K_V1_SYSTEM:,}, comparable: {K_V1_COMPARABLE}")


# =============================================================================
# ================== ESTIMATE TRAINING TRANSACTION COUNTS =====================
# =============================================================================

# 60% of the transactions associated with the relevant banks are in the train split.
TRAIN_FRAC = 0.6

# System: ~60% of FedAvg's obs_ava (≈ full dataset)
# Comparable: ~60% of individual's obs_ava
N_TRAIN_SYSTEM     = round(rb['FedAvg']['obs_ava'] * TRAIN_FRAC)
N_TRAIN_COMPARABLE = round(rb['individual']['obs_ava'] * TRAIN_FRAC)

N_BATCHES_SYSTEM     = math.ceil(N_TRAIN_SYSTEM     / BATCH_SIZE)
N_BATCHES_COMPARABLE = math.ceil(N_TRAIN_COMPARABLE / BATCH_SIZE)

print(f"\nTrain transactions — system: {N_TRAIN_SYSTEM:,}, comparable: {N_TRAIN_COMPARABLE:,}")
print(f"Batches/epoch      — system: {N_BATCHES_SYSTEM}, comparable: {N_BATCHES_COMPARABLE}")


# =============================================================================
# ========================= COMPUTE TOTAL BYTES ===============================
# =============================================================================

def fedavg_bytes(k, t, w=None, bpf=BYTES_PER_FLOAT):
    if w is None: w = W_FEDAVG
    """Total bytes = 2 (up+down) × K × W × 4 × T rounds."""
    return 2 * k * w * bpf * t

def v1_embed_bytes(n_batches, t, bs=BATCH_SIZE, ed=EMBED_DIM, bpf=BYTES_PER_FLOAT):
    """Total bytes = T × n_batches × batch_size × 2 (parties) × embed_dim × 4 × 2 (fwd+bwd)."""
    return t * n_batches * bs * 2 * ed * bpf * 2

def v1_weight_bytes(k_batch, n_batches, t, w=W_V1_PARTY, bpf=BYTES_PER_FLOAT):
    """GNN backbone weight exchange in SplitFed — per batch, each participating party uploads
    delta_w^GNN and downloads updated global w^GNN (W_V1_PARTY params each direction).
    PyTorch handles this implicitly; cost shown for real distributed setting.
    Total = 2 (up+down) × K_batch × W_V1_PARTY × 4 × n_batches × T."""
    return 2 * k_batch * w * bpf * n_batches * t

def v1_k_batch(k):
    """Expected unique parties per batch: K × (1 - exp(-2B/K)).
    Approximates birthday-problem coupon-collector for 2×BATCH_SIZE draws from K parties."""
    import math as _m
    return k * (1 - _m.exp(-2 * BATCH_SIZE / k))

def fmt_bytes(b):
    if b >= 1e12: return f"{b/1e12:.2f} TB"
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.2f} MB"
    return f"{b/1e3:.2f} KB"


rows = []

# --- FedAvg / FedProx ---
for eval_mode, k, n_batches in [
    ('Comparable', K_FEDAVG_COMPARABLE, N_BATCHES_COMPARABLE),
    ('System',     K_FEDAVG_SYSTEM,     N_BATCHES_SYSTEM),
]:
    per_ex  = W_FEDAVG * BYTES_PER_FLOAT
    n_exch  = 2 * k * T_FEDAVG
    total   = fedavg_bytes(k, T_FEDAVG)
    rows.append({
        'Scenario':    'FedAvg / FedProx (F1, P2)',
        'Eval':        eval_mode,
        'K':           k,
        'T':           T_FEDAVG,
        'Per-exchange': f'$W \\times 4$ = {fmt_bytes(per_ex)}',
        'Exchanges':    f'$2 \\cdot K \\cdot T = {n_exch:,}$',
        'Total':        fmt_bytes(total),
        '_total_bytes': total,
    })

# --- V1 (SplitFed) ---
for eval_mode, k, n_batches, t in [
    ('Comparable', K_V1_COMPARABLE, N_BATCHES_COMPARABLE, T_V1_COMPARABLE),
    ('System',     K_V1_SYSTEM,     N_BATCHES_SYSTEM,     T_V1_SYSTEM),
]:
    per_ex = BATCH_SIZE * 2 * EMBED_DIM * BYTES_PER_FLOAT   # both parties, one direction
    n_exch = 2 * n_batches * t                               # forward + backward
    total  = v1_embed_bytes(n_batches, t)
    rows.append({
        'Scenario':    'SplitFed (V1) — embeddings',
        'Eval':        eval_mode,
        'K':           k,
        'T':           t,
        'Per-exchange': f'$B \\times 2 \\times d \\times 4$ = {fmt_bytes(per_ex)}',
        'Exchanges':    f'$2 \\cdot n_{{\\text{{batch}}}} \\cdot T = {n_exch:,}$',
        'Total':        fmt_bytes(total),
        '_total_bytes': total,
    })

# --- V1 GNN weight exchange (real distributed setting) ---
# Per batch each participating party uploads delta_w^GNN and downloads updated w^GNN.
# K_batch estimated as expected unique parties per batch (birthday approximation).
for eval_mode, k, n_batches, t in [
    ('Comparable', K_V1_COMPARABLE, N_BATCHES_COMPARABLE, T_V1_COMPARABLE),
    ('System',     K_V1_SYSTEM,     N_BATCHES_SYSTEM,     T_V1_SYSTEM),
]:
    kb     = round(v1_k_batch(k))
    per_ex = W_V1_PARTY * BYTES_PER_FLOAT
    n_exch = 2 * kb * n_batches * t          # up + down, per batch, over all epochs
    total  = v1_weight_bytes(kb, n_batches, t)
    rows.append({
        'Scenario':    'SplitFed (V1) — GNN weights',
        'Eval':        eval_mode,
        'K':           kb,
        'T':           t,
        'Per-exchange': f'$W_{{\\text{{GNN}}}} \\times 4$ = {fmt_bytes(per_ex)}',
        'Exchanges':    f'$2 \\cdot \\hat{{K}}_{{\\text{{batch}}}} \\cdot n_{{\\text{{batch}}}} \\cdot T = {n_exch:,}$',
        'Total':        fmt_bytes(total),
        '_total_bytes': total,
    })

df = pd.DataFrame(rows)
print()
print(df[['Scenario', 'Eval', 'K', 'T', 'Per-exchange', 'Exchanges', 'Total']].to_string(index=False))


# =============================================================================
# ============================= LATEX TABLE ===================================
# =============================================================================
# Grouped by eval mode for SplitFed, with combined totals.
# FedAvg rows first, then SplitFed Comparable (embeds + weights + total),
# then SplitFed System (embeds + weights + total).

OUT_DIR = Path(__file__).resolve().parents[3] / 'writing/Experimental-Protocol/tables/data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _esc(s):
    return str(s).replace('%', r'\%')

def _k_str(val):
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return ''

v1_comp_total_bytes = (
    v1_embed_bytes(N_BATCHES_COMPARABLE, T_V1_COMPARABLE)
    + v1_weight_bytes(round(v1_k_batch(K_V1_COMPARABLE)), N_BATCHES_COMPARABLE, T_V1_COMPARABLE)
)
v1_sys_total_bytes = (
    v1_embed_bytes(N_BATCHES_SYSTEM, T_V1_SYSTEM)
    + v1_weight_bytes(round(v1_k_batch(K_V1_SYSTEM)), N_BATCHES_SYSTEM, T_V1_SYSTEM)
)

_TOTAL_ROW = lambda eval_mode, total_b: {
    'Scenario': 'SplitFed (V1) — Total',
    'Eval': eval_mode,
    'K': '',
    'T': '',
    'Per-exchange': '',
    'Exchanges': '',
    'Total': fmt_bytes(total_b),
    '_total_bytes': total_b,
}

table_rows = []
table_rows += [r.to_dict() for _, r in df[df['Scenario'] == 'FedAvg / FedProx (F1, P2)'].iterrows()]
table_rows += [r.to_dict() for _, r in df[df['Scenario'].str.startswith('SplitFed') & (df['Eval'] == 'Comparable')].iterrows()]
table_rows.append(_TOTAL_ROW('Comparable', v1_comp_total_bytes))
table_rows += [r.to_dict() for _, r in df[df['Scenario'].str.startswith('SplitFed') & (df['Eval'] == 'System')].iterrows()]
table_rows.append(_TOTAL_ROW('System', v1_sys_total_bytes))

body = []
prev_group = None
for row in table_rows:
    is_split = row['Scenario'].startswith('SplitFed')
    curr_group = ('SplitFed', row['Eval']) if is_split else ('FedAvg',)
    if prev_group is not None and curr_group != prev_group:
        body.append(r'\midrule')
    prev_group = curr_group
    total_str = (r'\textbf{' + row['Total'] + r'}') if row['Scenario'].endswith('Total') else row['Total']
    cells = [
        _esc(row['Scenario']),
        _esc(row['Eval']),
        _k_str(row['K']),
        str(row['T']),
        row['Per-exchange'],
        row['Exchanges'],
        total_str,
    ]
    body.append(' & '.join(cells) + r' \\')

lines = [
    r'\begin{tabular}{llrrrrl}',
    r'\toprule',
    r'Scenario & Eval & $K$ & $T$ & Per-exchange size & Exchanges & Total \\',
    r'\midrule',
    *body,
    r'\bottomrule',
    r'\end{tabular}',
]
TEX_PATH = OUT_DIR / 'communication_cost.tex'
TEX_PATH.write_text('\n'.join(lines) + '\n')
print(f'\nLaTeX table → {TEX_PATH}')

CSV_PATH = OUT_DIR / 'communication_cost.csv'
df[['Scenario', 'Eval', 'K', 'T', 'Per-exchange', 'Exchanges', 'Total']].to_csv(CSV_PATH, index=False)
print(f'CSV         → {CSV_PATH}')

# %%

# Summary printout
print('\n=== Summary ===')
print(f'W (FedAvg, all named_params) = {W_FEDAVG:,} params × 4 bytes = {fmt_bytes(W_FEDAVG * 4)}')
print(f'Embed dim (V1) = {EMBED_DIM}')
for row in table_rows:
    print(f"  {row['Scenario'][:25]:25s} {row['Eval']:12s}  total={row['Total']}")

# %%
