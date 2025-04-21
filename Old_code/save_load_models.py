from pathlib import Path
from abc import ABC, abstractmethod
import json
import trainer_utils as tu
import os
import configs
import re
import pandas as pd
from torch_geometric.loader import LinkNeighborLoader

def save_model(model, filename, scaler=None):

    if isinstance(model, xgb.Booster):
        if scaler is not None:
            # Save XGBoost model along with scaler
            joblib.dump({"model": model, "scaler": scaler}, filename)
            print(f"XGBoost model and scaler saved to {filename}")
        else:
            # Save only the XGBoost model
            model.save_model(filename)
            print(f"XGBoost model saved to {filename}")

    elif isinstance(model, torch.nn.Module):
        # Save PyTorch Geometric GNN model
        torch.save(model.state_dict(), filename)
        print(f"PyTorch GNN model saved to {filename}")

    else:
        raise TypeError("Unsupported model type. Provide an XGBoost Booster or PyTorch nn.Module.")


def save_configs(args, save_direc):
    
    # prep to save arguments
    args_dict = {'arguments': vars(args), 'model_configs': tu.get_model_configs(args)}
    folder_path = Path(save_direc)
    file_path = folder_path / 'configurations.json'

    if not os.path.exists(file_path):
        print(file_path)
        # ensure folder exists and save the file
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(args_dict, indent=4))
    




