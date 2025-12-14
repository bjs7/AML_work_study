import os
import json
import pickle
import torch
from pathlib import Path

class ExperimentResults:
    """Container for loaded experiment results."""
    
    def __init__(self, experiment_path):
        self.path = experiment_path
        self.config = None
        self.hyperparameters = None
        self.tuning_configs = None
        self.aggregated_stats = None
        self.seed_results = {}
        
    def __repr__(self):
        return f"ExperimentResults(path='{self.path}', seeds={list(self.seed_results.keys())})"

def load_experiment(experiment_path):
    """Load all results from an experiment directory.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        ExperimentResults object with all loaded data
    
    Example:
        results = load_experiment('/path/to/experiments/.../GINe/default')
        print(results.config)
        print(results.aggregated_stats['f1']['mean'])
        best_seed_num = results.aggregated_stats['best_seed']
        best_metrics = results.seed_results[best_seed_num]['metrics']
    """
    exp_path = Path(experiment_path)
    results = ExperimentResults(experiment_path)
    
    # Load experiment config
    config_file = exp_path / 'experiment_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            results.config = json.load(f)
    
    # Load hyperparameters
    hp_file = exp_path / 'hyper_parameters.json'
    if hp_file.exists():
        with open(hp_file, 'r') as f:
            results.hyperparameters = json.load(f)
    
    # Load tuning configs
    tuning_file = exp_path / 'model_tuning_configs.json'
    if tuning_file.exists():
        with open(tuning_file, 'r') as f:
            results.tuning_configs = json.load(f)
    
    # Load aggregated stats
    agg_file = exp_path / 'aggregated_results.json'
    if agg_file.exists():
        with open(agg_file, 'r') as f:
            results.aggregated_stats = json.load(f)
    
    # Load seed results
    for seed_dir in exp_path.glob('seed_*'):
        if seed_dir.is_dir():
            seed_num = int(seed_dir.name.split('_')[1])
            seed_data = {}
            
            # Load metrics and laundering values
            metrics_file = seed_dir / 'metrics_laundering_values.pkl'
            if metrics_file.exists():
                with open(metrics_file, 'rb') as f:
                    data = pickle.load(f)
                    seed_data['metrics'] = data['metrics']
                    seed_data['laundering_values'] = data['laundering_values']
            
            # Load model (full_info and FL)
            model_file = seed_dir / 'model.pth'
            if model_file.exists():
                seed_data['model'] = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
            
            # Load models (individual)
            models_file = seed_dir / 'models_hyperparameters.pkl'
            if models_file.exists():
                with open(models_file, 'rb') as f:
                    data = pickle.load(f)
                    seed_data['models'] = data['models_hyperparameters']
            
            results.seed_results[seed_num] = seed_data
    
    return results

def find_experiments(base_path, **filters):
    """Find experiments matching filter criteria.
    
    Args:
        base_path: Root experiments directory
        **filters: Filter criteria (e.g., size='small', fl_algo='individual', ibm_fe=True)
        
    Returns:
        List of experiment paths matching filters
        
    Example:
        # Find all small dataset individual bank experiments
        exps = find_experiments('/path/to/experiments', size='small', fl_algo='individual')
        
        # Load first match
        results = load_experiment(exps[0])
    """
    matches = []
    base_path = Path(base_path)
    
    for exp_config in base_path.rglob('experiment_config.json'):
        with open(exp_config, 'r') as f:
            config = json.load(f)
        
        # Check if all filters match
        match = True
        for key, value in filters.items():
            # Navigate nested dict (e.g., 'data.size' or just 'size')
            if '.' in key:
                parts = key.split('.')
                config_value = config
                for part in parts:
                    config_value = config_value.get(part)
                    if config_value is None:
                        match = False
                        break
            else:
                # Try to find key in any nested dict
                config_value = None
                for section in config.values():
                    if isinstance(section, dict) and key in section:
                        config_value = section[key]
                        break
                
                if config_value is None:
                    match = False
                    break
            
            if config_value != value:
                match = False
                break
        
        if match:
            matches.append(str(exp_config.parent))
    
    return matches