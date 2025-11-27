# %%


from load_results import find_experiments, load_experiment




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
