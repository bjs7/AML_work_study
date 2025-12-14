# %%

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_results import find_experiments, load_experiment
import json
import pickle
import pandas as pd
import numpy as np




# %%
# Find all individual GINe experiments
experiment_paths = find_experiments(
    '/home/nam_07/projects/AML_work_study/experiments',
    fl_algo='individual',
    model_name='GINe'
)

# Shows: ['/path/to/experiments/.../individual/GINe/default', 
#         '/path/to/experiments/.../individual/GINe/ibm_fe']
print(experiment_paths)

# %%




results = load_experiment(experiment_paths[0])

# Access the data
print(results.config)
print(results.aggregated_stats['f1']['mean'])
print(results.hyperparameters)




# %%

paths = find_experiments(
    '/home/nam_07/projects/AML_work_study/experiments',
    size='small',
    fl_algo='individual'
)

# 2. Load each and extract F1 scores
for path in paths:
    exp = load_experiment(path)
    print(f"Experiment: {exp.config['model']['model_name']}")
    print(f"Settings: {path.split('/')[-1]}")  # Shows 'default' or 'ibm_fe' etc.
    print(f"Mean F1: {exp.aggregated_stats['f1']['mean']:.3f}")
    print(f"Best seed: {exp.aggregated_stats['best_seed']}")
    print()

# %%
# ============================================================================
# LOAD YOUR SPECIFIC EXPERIMENT
# ============================================================================

experiment_path = '/home/nam_07/projects/AML_work_study/experiments/small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'

# Load the experiment
results = load_experiment(experiment_path)

print(f"Loaded experiment from: {experiment_path}")
print(f"Available seeds: {list(results.seed_results.keys())}")
print()

# %%
# ============================================================================
# VIEW EXPERIMENT CONFIGURATION
# ============================================================================

print("=== Experiment Configuration ===")
print(json.dumps(results.config, indent=2))
print()

# %%
# ============================================================================
# VIEW HYPERPARAMETERS
# ============================================================================

print("=== Hyperparameters ===")
print(json.dumps(results.hyperparameters, indent=2))
print()

# %%
# ============================================================================
# VIEW AGGREGATED RESULTS (SUMMARY ACROSS ALL SEEDS)
# ============================================================================

print("=== Aggregated Results ===")
print(json.dumps(results.aggregated_stats, indent=2))
print()

# Summary table
metrics_df = pd.DataFrame({
    metric: {
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max']
    }
    for metric, stats in results.aggregated_stats.items()
    if metric != 'best_seed' and isinstance(stats, dict)
}).T

print("\n=== Summary Table ===")
print(metrics_df)
print(f"\nBest seed: {results.aggregated_stats.get('best_seed', 'N/A')}")
print()

# %%
# ============================================================================
# ANALYZE INDIVIDUAL SEED RESULTS
# ============================================================================

print("=== Individual Seed Results ===")
for seed_num, seed_data in sorted(results.seed_results.items()):
    print(f"\n--- Seed {seed_num} ---")

    if 'metrics' in seed_data:
        print("Metrics:", seed_data['metrics'])

    if 'laundering_values' in seed_data:
        lv = seed_data['laundering_values']
        print(f"Laundering values shape/info: {type(lv)}")
        if isinstance(lv, (list, np.ndarray)):
            print(f"  Length: {len(lv)}")
            if len(lv) > 0:
                print(f"  Sample (first 5): {lv[:5]}")

    if 'model' in seed_data:
        print(f"Model loaded: {type(seed_data['model'])}")

print()

# %%
# ============================================================================
# DETAILED ANALYSIS OF BEST SEED
# ============================================================================

best_seed = results.aggregated_stats.get('best_seed')
if best_seed and best_seed in results.seed_results:
    print(f"=== Detailed Analysis of Best Seed ({best_seed}) ===")
    best_seed_data = results.seed_results[best_seed]

    print(f"\nMetrics: {best_seed_data.get('metrics', {})}")

    if 'laundering_values' in best_seed_data:
        lv = best_seed_data['laundering_values']
        print(f"\nLaundering values statistics:")
        if isinstance(lv, (list, np.ndarray)):
            lv_array = np.array(lv)
            print(f"  Count: {len(lv_array)}")
            print(f"  Mean: {np.mean(lv_array):.4f}")
            print(f"  Std: {np.std(lv_array):.4f}")
            print(f"  Min: {np.min(lv_array):.4f}")
            print(f"  Max: {np.max(lv_array):.4f}")

print()

# %%
# ============================================================================
# COMPARE PERFORMANCE ACROSS SEEDS
# ============================================================================

print("=== Seed-by-Seed Comparison ===")

# Extract metrics for each seed from aggregated_stats
seed_comparison = {}
for metric, stats in results.aggregated_stats.items():
    if metric != 'best_seed' and isinstance(stats, dict) and 'values' in stats:
        seed_comparison[metric] = stats['values']

# Create DataFrame
comparison_df = pd.DataFrame(seed_comparison)
comparison_df.index = [f"seed_{i+1}" for i in range(len(comparison_df))]

print(comparison_df)
print()

# %%
