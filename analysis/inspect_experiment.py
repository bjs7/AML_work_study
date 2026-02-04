# %%

import os
import json
import torch


# %%

# ============================================================================
# EXPERIMENT INSPECTION UTILITIES
# ============================================================================

# Known data flags that can appear in the directory path
DATA_FLAGS = ['batching', 'ibm_fe', 'ibm_hp', 'use_global_stats', 'train_for_final']

# FL algorithms that force LayerNorm regardless of batching flag
LAYERNORM_FORCED_ALGOS = ('FedGraph', 'FedAvg', 'FedProx')


def parse_path_settings(experiment_path):
    """Extract settings encoded in the directory path that may not be in experiment_config.json.

    The directory structure encodes:
        .../{size}_{ir}/split_{t}_{v}/{eval_mode}/{fl_algo}/{algo_subfolder}/{model}/{data_flags}/

    Returns dict with parsed path components.
    """
    parts = experiment_path.replace('\\', '/').rstrip('/').split('/')

    parsed = {}

    # Find data flags folder - it's the folder containing __ separated flags or 'default'
    for part in parts:
        flags_found = [f for f in DATA_FLAGS if f in part.split('__')]
        if flags_found or part == 'default':
            parsed['data_flags_from_path'] = flags_found
            for flag in DATA_FLAGS:
                parsed[f'path_{flag}'] = flag in flags_found
            break

    # Find size_ir pattern (e.g. small_HI)
    for part in parts:
        for size in ('small', 'medium', 'large'):
            for ir in ('HI', 'LO'):
                if part == f'{size}_{ir}':
                    parsed['path_size'] = size
                    parsed['path_ir'] = ir

    # Find split pattern
    for part in parts:
        if part.startswith('split_'):
            parsed['path_split'] = part

    # Find FL algo
    fl_algos = ('full_info', 'individual', 'FedAvg', 'FedProx', 'FedGraph')
    for part in parts:
        if part in fl_algos:
            parsed['path_fl_algo'] = part

    # Find model name
    model_names = ('GINe', 'GINe_2', 'GATe', 'xgboost', 'light_gbm', 'regression')
    for part in parts:
        base = part.split('__')[0]
        if base in model_names:
            parsed['path_model'] = part
            parsed['path_model_base'] = base
            gnn_flags = part.split('__')[1:]
            if gnn_flags:
                parsed['path_gnn_flags'] = gnn_flags

    # Find FedAvg/FedProx algo subfolder (e.g. proportional_C0.25_E10)
    for part in parts:
        if part.startswith(('proportional_', 'uniform_')):
            parsed['path_algo_subfolder'] = part

    return parsed


def derive_norm_type(fl_algo, batching):
    """Derive normalization type from fl_algo and batching flag.

    Logic from models/gnn.py _create_gnn_model():
        use_batchnorm = batching AND fl_algo NOT IN ('FedGraph', 'FedAvg', 'FedProx')
    """
    use_batchnorm = batching and fl_algo not in LAYERNORM_FORCED_ALGOS
    return 'BatchNorm' if use_batchnorm else 'LayerNorm'


def inspect_model_weights(experiment_path, seed=1):
    """Load and inspect saved model weights to infer architecture details.

    Returns dict with architecture info extracted from the state dict,
    or None if no model file exists.
    """
    model_path = os.path.join(experiment_path, f'seed_{seed}', 'model.pth')
    if not os.path.exists(model_path):
        return None

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    if not isinstance(state_dict, dict):
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        else:
            return {'type': type(state_dict).__name__, 'note': 'Could not extract state dict'}

    info = {
        'num_parameters': sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel')),
        'layers': {}
    }

    for name, param in state_dict.items():
        if hasattr(param, 'shape'):
            info['layers'][name] = list(param.shape)

    # Infer norm type from layer names
    norm_layers = [k for k in state_dict.keys() if 'batch_norm' in k or 'layer_norm' in k]
    if norm_layers:
        # Check if it's actually BatchNorm or LayerNorm by looking at parameter names
        has_running_mean = any('running_mean' in k for k in state_dict.keys())
        info['inferred_norm'] = 'BatchNorm' if has_running_mean else 'LayerNorm'

    return info


def inspect_experiment(experiment_path, show_model_weights=False, seed=1):
    """Inspect and display all settings/configuration for a saved experiment.

    Args:
        experiment_path: Path to experiment directory (the one containing experiment_config.json)
        show_model_weights: If True, load model.pth and show architecture details
        seed: Which seed's model to inspect (default: 1)
    """
    path_settings = parse_path_settings(experiment_path)

    # Load config files
    config = None
    config_path = os.path.join(experiment_path, 'experiment_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    hyperparams = None
    hp_path = os.path.join(experiment_path, 'hyper_parameters.json')
    if os.path.exists(hp_path):
        with open(hp_path, 'r') as f:
            hyperparams = json.load(f)

    tuning_configs = None
    tc_path = os.path.join(experiment_path, 'model_tuning_configs.json')
    if os.path.exists(tc_path):
        with open(tc_path, 'r') as f:
            tuning_configs = json.load(f)

    aggregated = None
    agg_path = os.path.join(experiment_path, 'aggregated_results.json')
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            aggregated = json.load(f)

    # Determine key settings, preferring config file, falling back to path
    fl_algo = (config or {}).get('fl', {}).get('fl_algo', path_settings.get('path_fl_algo', '?'))
    batching = path_settings.get('path_batching', False)
    ibm_hp = path_settings.get('path_ibm_hp', False)
    ibm_fe = (config or {}).get('data', {}).get('ibm_fe', path_settings.get('path_ibm_fe', False))
    train_for_final = (config or {}).get('data', {}).get('train_for_final', path_settings.get('path_train_for_final', False))
    model_name = (config or {}).get('model', {}).get('model_name', path_settings.get('path_model_base', '?'))
    model_type = (config or {}).get('model', {}).get('model_type', '?')

    norm_type = derive_norm_type(fl_algo, batching) if model_type == 'gnn' else 'N/A'

    # Count available seeds
    seed_dirs = [d for d in os.listdir(experiment_path) if d.startswith('seed_') and os.path.isdir(os.path.join(experiment_path, d))]
    has_model_file = any(os.path.exists(os.path.join(experiment_path, sd, 'model.pth')) for sd in seed_dirs)

    # --- Display ---

    print("=" * 70)
    print("EXPERIMENT INSPECTION")
    print("=" * 70)
    print(f"Path: {experiment_path}")
    if config and 'timestamp' in config:
        print(f"Run timestamp: {config['timestamp']}")
    print()

    # Data settings
    print("-" * 40)
    print("DATA SETTINGS")
    print("-" * 40)
    data_cfg = (config or {}).get('data', {})
    print(f"  Dataset size:     {data_cfg.get('size', path_settings.get('path_size', '?'))}")
    print(f"  Illicit ratio:    {data_cfg.get('ir', path_settings.get('path_ir', '?'))}")
    print(f"  Split:            {data_cfg.get('split', path_settings.get('path_split', '?'))}")
    print(f"  Eval mode:        {data_cfg.get('eval_mode', '?')}")
    print(f"  Testing:          {data_cfg.get('testing', '?')}")
    print(f"  IBM features:     {ibm_fe}")
    print(f"  IBM hyperparams:  {ibm_hp}")
    print(f"  Batching:         {batching}")
    print(f"  Train for final:  {train_for_final}")
    if 'data_flags_from_path' in path_settings:
        print(f"  [from path]:      {path_settings['data_flags_from_path']}")
    print()

    # FL settings
    print("-" * 40)
    print("FEDERATED LEARNING SETTINGS")
    print("-" * 40)
    fl_cfg = (config or {}).get('fl', {})
    print(f"  FL algorithm:     {fl_algo}")
    print(f"  Scenario:         {fl_cfg.get('scenario', '?')}")
    if fl_algo in ('FedAvg', 'FedProx'):
        print(f"  Weighting:        {fl_cfg.get('weighting', path_settings.get('path_algo_subfolder', '?'))}")
        print(f"  Client fraction:  {fl_cfg.get('client_fraction', '?')}")
        print(f"  Local epochs:     {fl_cfg.get('num_local_epochs', '?')}")
        if fl_algo == 'FedProx':
            print(f"  Mu (proximal):    {fl_cfg.get('mu', '?')}")
        if 'path_algo_subfolder' in path_settings:
            print(f"  [algo subfolder]: {path_settings['path_algo_subfolder']}")
    elif fl_algo == 'FedGraph':
        print(f"  Aggregation:      {fl_cfg.get('aggregation', '?')}")
    print()

    # Model settings
    print("-" * 40)
    print("MODEL SETTINGS")
    print("-" * 40)
    print(f"  Model type:       {model_type}")
    print(f"  Model name:       {model_name}")

    gnn_settings = (config or {}).get('model', {}).get('gnn_settings', {})
    if gnn_settings:
        print(f"  Edge MLPs:        {gnn_settings.get('emlps', False)}")
        print(f"  Ports:            {gnn_settings.get('ports', False)}")
        print(f"  Time deltas:      {gnn_settings.get('tds', False)}")
        print(f"  Reverse MP:       {gnn_settings.get('reverse_mp', False)}")
    elif 'path_gnn_flags' in path_settings:
        print(f"  GNN flags [path]: {path_settings['path_gnn_flags']}")

    # Derived model settings
    print()
    print(f"  >>> Normalization: {norm_type} <<<")
    if model_type == 'gnn':
        if batching and fl_algo in LAYERNORM_FORCED_ALGOS:
            print(f"      (batching=True, but {fl_algo} forces LayerNorm)")
        elif batching:
            print(f"      (batching=True, full_info/individual → BatchNorm)")
        else:
            print(f"      (batching=False → LayerNorm)")
    print()

    # Training settings
    print("-" * 40)
    print("TRAINING SETTINGS")
    print("-" * 40)
    training_cfg = (config or {}).get('training', {})
    print(f"  Seeds:            {training_cfg.get('seeds', len(seed_dirs))}")
    print(f"  Epochs:           {training_cfg.get('epochs', '?')}")
    print(f"  Available seeds:  {sorted([d for d in seed_dirs])}")
    print(f"  Model saved:      {has_model_file}")
    print()

    # Hyperparameters
    if hyperparams:
        print("-" * 40)
        print("HYPERPARAMETERS")
        print("-" * 40)
        for key, value in hyperparams.items():
            if isinstance(value, float):
                print(f"  {key:.<30s} {value:.6f}")
            else:
                print(f"  {key:.<30s} {value}")
        print()

    # Tuning configs
    if tuning_configs:
        print("-" * 40)
        print("TUNING CONFIGURATION")
        print("-" * 40)
        print(f"  {json.dumps(tuning_configs, indent=4)}")
        print()

    # Aggregated results summary
    if aggregated:
        print("-" * 40)
        print("PERFORMANCE SUMMARY")
        print("-" * 40)
        for metric in ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc']:
            if metric in aggregated and isinstance(aggregated[metric], dict):
                stats = aggregated[metric]
                print(f"  {metric:.<15s} {stats['mean']*100:.2f}% (std: {stats['std']*100:.2f}%, range: [{stats['min']*100:.2f}%, {stats['max']*100:.2f}%])")
        if 'best_seed' in aggregated:
            print(f"  Best seed:        {aggregated['best_seed']} (F1: {aggregated.get('best_f1', 0)*100:.2f}%)")
        print()

    # Model architecture from weights
    if show_model_weights:
        print("-" * 40)
        print(f"MODEL ARCHITECTURE (from seed_{seed}/model.pth)")
        print("-" * 40)
        model_info = inspect_model_weights(experiment_path, seed=seed)
        if model_info is None:
            print("  No model.pth found for this seed.")
        elif 'note' in model_info:
            print(f"  {model_info['note']}")
        else:
            print(f"  Total parameters: {model_info['num_parameters']:,}")
            if 'inferred_norm' in model_info:
                print(f"  Inferred norm:    {model_info['inferred_norm']}")
            print()
            print("  Layer shapes:")
            for name, shape in model_info['layers'].items():
                print(f"    {name}: {shape}")
        print()

    print("=" * 70)

    return {
        'config': config,
        'hyperparameters': hyperparams,
        'tuning_configs': tuning_configs,
        'aggregated': aggregated,
        'path_settings': path_settings,
        'derived': {
            'norm_type': norm_type,
            'batching': batching,
            'ibm_hp': ibm_hp,
            'fl_algo': fl_algo,
        }
    }


# %%
# ============================================================================
# INSPECT YOUR EXPERIMENT
# ============================================================================

experiment_path = '/home/nam_07/projects/AML_work_study/experiments/small_HI/split_0.6_0.2/FedAvg/GINe/batching__ibm_fe__ibm_hp__train_for_final'

info = inspect_experiment(experiment_path, show_model_weights=True)


# %%
