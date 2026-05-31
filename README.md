# Federated Learning for Anti-Money Laundering Detection

This repository studies the use of Federated Learning (FL) and Graph Neural Networks (GNNs) for anti-money laundering. Both horizontal and vertical FL frameworks are implemented and benchmarked against two baselines: a full-information model (upper bound) and individual per-bank models (lower bound).

---

## Status

This project is still work in progress. Setup and usage sections are included to illustrate the project's design and scope rather than as a guide for production use. The project is not intended as a production-ready tool.

## Repository Structure

```
configs/                    — training constants and path resolution
data/                       — data loading, splitting, and feature engineering
federated_learning/
  gnn/                      — GNN-based FL (horizontal, vertical, baselines)
    vertical/               — full vertical FL (per-layer embedding exchange)
    vertical_simple/        — simplified vertical FL (single embedding pass)
  booster/                  — tree-based FL (XGBoost, SecureBoost)
models/                     — GNN and booster model architectures
inference.py                — evaluation metrics (F1, PR-AUC, ROC-AUC)
result_io/                  — saving and loading experiment results
main.py                     — entry point
```

---

## Dataset

The project uses the [IBM Realistic Synthetic Financial Transactions dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml?select=HI-Medium_Trans.csv), formatted as transaction CSVs. Three sizes (`small`, `medium`, `large`) and two illicit ratios (`HI`, `LO`) are supported.

Raw CSVs should be placed at:
```
<data_root>/AML_work_study/formatted_transactions_<size>_<ir>.csv
```

---

## Setup

```bash
conda env create -f environments/environment_hpc_export.yml
conda activate multignn_hpc
bash environments/post_install_local.sh   # installs PyTorch + PyG
```

---

## Usage

**FedAvg (horizontal FL, GNN):**
```bash
python main.py --fl_algo FedAvg --model GINe --size small --ir HI \
               --ibm_fe --ibm_hp --batching --emlps
```

**FedGraphSimple (vertical FL, GNN):**
```bash
python main.py --fl_algo FedGraphSimple --model GINe --size small --ir HI \
               --ibm_fe --ibm_hp --batching --emlps
```

**Individual baseline (single-bank GNN):**
```bash
python main.py --fl_algo individual --model GINe --size small --ir HI \
               --ibm_fe --ibm_hp --batching --emlps
```

**Quick smoke test (reduced data and epochs):**
```bash
python main.py --fl_algo FedAvg --model GINe --size small --ir HI \
               --ibm_fe --ibm_hp --batching --emlps --testing
```

### Key arguments

| Argument | Description |
|---|---|
| `--fl_algo` | FL algorithm: `FedAvg`, `FedProx`, `FedGraph`, `FedGraphSimple`, `individual`, `full_info` |
| `--model` | GNN architecture: `GINe` |
| `--size` | Dataset size: `small`, `medium`, `large` |
| `--ir` | Illicit ratio: `HI`, `LO` |
| `--eval_mode` | Evaluation mode: `system` (all banks) or `comparable` (matched bank set) |
| `--batching` | Enable mini-batch training via `LinkNeighborLoader` |
| `--emlps` | Enable edge message passing in GINe |
| `--num_rounds` | Number of training rounds (default: 100) |
| `--client_fraction` | Fraction of parties sampled per round (FedAvg/FedProx) |
| `--mu` | Proximal term weight for FedProx |
| `--bank_filter` | Filter banks by edge count percentile (e.g. `no_bottom10`) |

---

## Evaluation

Models are evaluated and selected based on F1. Two evaluation modes are supported:

- **System-level**: aggregates predictions across all banks seeing a transaction; a laundering transaction is flagged if *any* bank detects it
- **Comparable**: restricts evaluation to a fixed subset of banks so results are directly comparable across FL algorithms

Results are saved per seed under `experiments/` and can be loaded via `result_io/load_results.py`.
Results include per-seed metrics (F1, precision, recall), model weights and predicted probabilities.

---
