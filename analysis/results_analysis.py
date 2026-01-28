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

def get_experiment_path(fl_algo, model, data_size = 'small', 
                   ilicit_level = 'HI', ibm_fe = False, ibm_hp = False, train_for_final = False,
                   batching = False, emlps = False):
    
    base_path = '/home/nam_07/projects/AML_work_study/experiments'
    base_path += f"/{data_size}_{ilicit_level}/split_0.6_0.2/{fl_algo}/{model}"

    for gnn_arg, str in zip([emlps], ['__emlps']):
        base_path += str if gnn_arg else ""

    data_args = []
    for boolarg, str in zip([batching, ibm_fe, ibm_hp, train_for_final], ["batching", "ibm_fe", "ibm_hp", 'train_for_final']):
        if boolarg: data_args.append(str)
    if data_args != []: data_args = "__".join(data_args)

    return os.path.join(base_path, data_args)


experiments_paths = (
    get_experiment_path(fl_algo='full_info', model='GINe', batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True),
    get_experiment_path(fl_algo='full_info', model='GINe', emlps = True, batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True),
    get_experiment_path(fl_algo='individual', model='GINe', batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True),
    get_experiment_path(fl_algo='individual', model='GINe', emlps = True, batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True),
    get_experiment_path(fl_algo='FedAvg', model='GINe', batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True),
    get_experiment_path(fl_algo='FedAvg', model='GINe', emlps = True, batching=True, ibm_fe=True, ibm_hp=True, train_for_final=True)
)

# %%

results = {}

for idx, path in enumerate(experiments_paths):
    #len('/home/nam_07/projects/AML_work_study/experiments/')
    results[path[49:]] = load_experiment(path)


# %%

def get_f1_scores(results):

    for path, result in results.items():
        f1_mean = result.aggregated_stats['f1']['mean']
        diff_min = f1_mean - min(result.aggregated_stats['f1']['values'])
        diff_max = max(result.aggregated_stats['f1']['values']) - f1_mean
        plus_min_value = max(diff_min, diff_max) * 100
        f1_mean *= 100
        
        print(f"=== f1 score {path}")
        print(f"mean: {round(f1_mean, 2)}, plus_min: {round(plus_min_value, 2)}")


get_f1_scores(results)


# %%

import pickle

for exp_path in experiments_paths:

    laun_values = []
    false_pos = []

    for seed in [1,2,3,4]:
        path = exp_path + f'/seed_{seed}/'

        with open(path + 'metrics_laundering_values.pkl', 'rb') as f:
            data = pickle.load(f)

        launderings_found = sum((data['laundering_values']['true_y'] == 1) & (data['laundering_values']['pred_label'] == 1))
        false_positives = sum((data['laundering_values']['true_y'] == 0) & (data['laundering_values']['pred_label'] == 1))

        laun_values.append(launderings_found)
        false_pos.append(false_positives)

    laun_values_mean = np.mean(laun_values)
    laun_diff_min = laun_values_mean - min(laun_values)
    laun_diff_max = max(laun_values) - laun_values_mean
    laun_plus_min_value = max(laun_diff_max, laun_diff_min)

    false_pos_mean = np.mean(false_pos)
    false_pos_diff_min = false_pos_mean - min(false_pos)
    false_pos_diff_max = max(false_pos) - false_pos_mean
    false_pos_plus_min_value = max(false_pos_diff_max, false_pos_diff_min)

    print(f"=== Found illicit cases {path[49:]}")
    print(f"mean: {laun_values_mean}, plus_min: {laun_plus_min_value}")

    print(f"=== False positive cases {path[49:]}")
    print(f"mean: {false_pos_mean}, plus_min: {false_pos_plus_min_value}")


# %%

experiments_paths


for path, result in results.items():
    test123 = result.aggregated_stats

list(test123)

# Extract all metrics
metrics_to_extract = ['precision', 'recall', 'roc_auc', 'pr_auc']
metrics_results = {}

for path, result in results.items():
    metrics_results[path] = {}
    
    for metric in metrics_to_extract:
        mean = result.aggregated_stats[metric]['mean']
        diff_min = mean - min(result.aggregated_stats[metric]['values'])
        diff_max = max(result.aggregated_stats[metric]['values']) - mean
        plus_min_value = max(diff_min, diff_max) * 100
        mean *= 100
        
        metrics_results[path][metric] = {
            'mean': mean,
            'plus_min': plus_min_value
        }







# %%

top_30_by_edges = [68, 4, 0, 2, 1682, 2279, 9898, 11300, 542, 13342, 1703, 147, 4881, 50, 18, 13341, 1643, 61, 1633, 1533, 12208, 11794, 11, 2278, 5611, 3394, 14091, 22, 108, 25]
top_30_banks_f1 = {}

for seed in [1,2,3,4]:
    path = experiments_paths[2] + f'/seed_{seed}/'
    with open(path + 'metrics_laundering_values.pkl', 'rb') as f:
        data = pickle.load(f)

    top_30 = dict(sorted(data['party_performance'].items(), key=lambda x: x[1]['f1'], reverse=True)[:30])
    for bank_id in top_30_by_edges:
        if seed == 1:
            top_30_banks_f1[bank_id] = [data['party_performance'][bank_id]['f1']]
        else:
            top_30_banks_f1[bank_id].append(data['party_performance'][bank_id]['f1'])


top_30_banks_f1_means_std = {}
for bank_id, values in top_30_banks_f1.items():
    top_30_banks_f1_means_std[bank_id] = {'mean': np.mean(values), 'std': np.std(values)}



import matplotlib.pyplot as plt
import numpy as np

keys = list(top_30_banks_f1_means_std.keys())
means = [top_30_banks_f1_means_std[k]['mean'] for k in keys]
stds = [top_30_banks_f1_means_std[k]['std'] for k in keys]

plt.figure(figsize=(12, 6))
plt.errorbar(range(len(keys)), means, yerr=stds, fmt='o', capsize=3, alpha=0.6)
plt.xlabel('Bank ID')
plt.ylabel('Mean ± range (min/max deviation from mean)')
plt.title('Mean values of individual f1 score')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%








# %%

print(f"Loaded experiment from: {results[0]}")
print(f"Available seeds: {list(results[0].seed_results.keys())}")
print()


print(json.dumps(results['small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats, indent=2))






print("=== Aggregated Results ===")
print(json.dumps(results[0].aggregated_stats, indent=2))
print()

# Summary table
metrics_df = pd.DataFrame({
    metric: {
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max']
    }
    for metric, stats in results['small_HI/split_0.6_0.2/full_info/GINe/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.items()
    if metric != 'best_seed' and isinstance(stats, dict)
}).T

print("\n=== Summary Table ===")
print(metrics_df)
print(f"\nBest seed: {results[0].aggregated_stats.get('best_seed', 'N/A')}")
print()








# %%



# Load the experiment
results = load_experiment(experiment_path)

print(f"Loaded experiment from: {experiment_path}")
print(f"Available seeds: {list(results.seed_results.keys())}")
print()






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

best_seed = results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.get('best_seed')
if best_seed and best_seed in results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].seed_results:
    print(f"=== Detailed Analysis of Best Seed ({best_seed}) ===")
    best_seed_data = results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].seed_results[best_seed]

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
for metric, stats in results['small_HI/split_0.6_0.2/full_info/GINe__emlps/batching__ibm_fe__ibm_hp__train_for_final'].aggregated_stats.items():
    if metric != 'best_seed' and isinstance(stats, dict) and 'values' in stats:
        seed_comparison[metric] = stats['values']

# Create DataFrame
comparison_df = pd.DataFrame(seed_comparison)
comparison_df.index = [f"seed_{i+1}" for i in range(len(comparison_df))]

print(comparison_df)
print()

# %%
