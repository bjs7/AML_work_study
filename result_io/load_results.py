import json
import pickle
import torch
import io
import re
from pathlib import Path

# Custom unpickler to force all PyTorch objects to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

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

def _resolve_run_folder(exp_path, run_id=None):
    """Resolve the actual run folder inside an experiment path.

    If exp_path directly contains experiment files, returns exp_path.
    If it contains timestamp subfolders (YYYYMMDD_HHMMSS), returns the
    latest one (or the one matching run_id).
    """
    if not exp_path.exists():
        return exp_path  # Path doesn't exist yet, return as-is

    # Check if experiment files exist directly (backward compat)
    if (exp_path / 'experiment_config.json').exists():
        return exp_path

    # Look for timestamp subfolders
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
    run_folders = sorted(
        [d for d in exp_path.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)],
        key=lambda d: d.name
    )

    if not run_folders:
        return exp_path  # No runs found, return as-is (will just have empty results)

    if run_id is not None:
        match = [d for d in run_folders if d.name == run_id]
        if not match:
            raise FileNotFoundError(f"Run '{run_id}' not found in {exp_path}. "
                                    f"Available: {[d.name for d in run_folders]}")
        return match[0]

    # Default: latest run
    return run_folders[-1]


def load_experiment(experiment_path, run_id=None):
    """Load all results from an experiment directory.

    Args:
        experiment_path: Path to experiment folder (up to the data_flags level)
        run_id: Specific run to load (e.g. '20260208_143022'). None = latest.

    Returns:
        ExperimentResults object with all loaded data

    Example:
        results = load_experiment('/path/to/experiments/.../GINe/default')
        results = load_experiment('/path/to/...', run_id='20260208_143022')
    """
    exp_path = Path(experiment_path)
    exp_path = _resolve_run_folder(exp_path, run_id)
    results = ExperimentResults(str(exp_path))
    
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
                    data = CPU_Unpickler(f).load()
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
                    data = CPU_Unpickler(f).load()
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