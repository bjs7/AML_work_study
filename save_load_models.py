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
