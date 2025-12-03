"""
Helper script to load and analyze results from main_full_info.py experiments
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

def load_results(results_file):
    """Load results from a pickle file"""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_single_run(results_data):
    """Analyze results from a single training run (with IBM hyperparameters)"""

    results = results_data['results']

    print("="*80)
    print("SINGLE RUN ANALYSIS")
    print("="*80)
    print(f"Timestamp: {results_data['timestamp']}")
    print(f"Seed: {results_data['seed']}")
    print(f"Data size: {results_data['data_size']}, IR: {results_data['ir']}")
    print()

    print("Best Results:")
    print(f"  Train F1: {results['best_train_f1']:.4f}")
    print(f"  Eval F1: {results['best_eval_f1']:.4f}")

    if results['best_eval_results']:
        best_eval = results['best_eval_results']
        print(f"  Eval Precision: {best_eval['precision']:.4f}")
        print(f"  Eval Recall: {best_eval['recall']:.4f}")
        print(f"  Eval ROC-AUC: {best_eval['roc_auc']:.4f}")

    print()
    print("Hyperparameters:")
    for key, value in results['hyperparameters'].items():
        print(f"  {key}: {value}")

    # Analyze epoch history
    if 'epochs_history' in results:
        history = pd.DataFrame(results['epochs_history'])

        print()
        print("Training History Summary:")
        print(f"  Total epochs: {len(history)}")
        print(f"  Final train loss: {history['train_loss'].iloc[-1]:.4f}")
        print(f"  Final train F1: {history['train_f1'].iloc[-1]:.4f}")
        print(f"  Final eval F1: {history['eval_f1'].iloc[-1]:.4f}")
        print(f"  Best epoch: {history.loc[history['eval_f1'].idxmax(), 'epoch']}")

        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(history['epoch'], history['train_loss'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True)

        axes[0, 1].plot(history['epoch'], history['train_f1'], label='Train')
        axes[0, 1].plot(history['epoch'], history['eval_f1'], label='Eval')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(history['epoch'], history['train_precision'], label='Train')
        axes[1, 0].plot(history['epoch'], history['eval_precision'], label='Eval')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(history['epoch'], history['train_recall'], label='Train')
        axes[1, 1].plot(history['epoch'], history['eval_recall'], label='Eval')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save figure
        results_dir = os.path.dirname(os.path.abspath(__file__))
        plot_file = os.path.join(results_dir, f"training_curves_{results_data['timestamp']}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nTraining curves saved to: {plot_file}")
        plt.close()

        return history

def analyze_hp_search(results_data):
    """Analyze results from hyperparameter search"""

    results = results_data['results']

    print("="*80)
    print("HYPERPARAMETER SEARCH ANALYSIS")
    print("="*80)
    print(f"Timestamp: {results_data['timestamp']}")
    print(f"Seed: {results_data['seed']}")
    print(f"Data size: {results_data['data_size']}, IR: {results_data['ir']}")
    print(f"Number of trials: {len(results)}")
    print()

    # Collect results
    trials_data = []
    for trial_id, trial_results in results.items():
        trial_data = {
            'trial': trial_id,
            'best_train_f1': trial_results['best_train_f1'],
            'best_eval_f1': trial_results['best_eval_f1'],
        }
        trial_data.update(trial_results['hyperparameters'])
        trials_data.append(trial_data)

    df = pd.DataFrame(trials_data)

    print("Best Trial:")
    best_trial = df.loc[df['best_eval_f1'].idxmax()]
    print(f"  Trial {int(best_trial['trial'])}")
    print(f"  Eval F1: {best_trial['best_eval_f1']:.4f}")
    print(f"  Train F1: {best_trial['best_train_f1']:.4f}")
    print()
    print("  Hyperparameters:")
    hp_cols = [col for col in df.columns if col not in ['trial', 'best_train_f1', 'best_eval_f1']]
    for col in hp_cols:
        print(f"    {col}: {best_trial[col]}")

    print()
    print("Top 5 Trials:")
    top5 = df.nlargest(5, 'best_eval_f1')[['trial', 'best_eval_f1', 'best_train_f1']]
    print(top5.to_string(index=False))

    print()
    print("Summary Statistics:")
    print(f"  Mean Eval F1: {df['best_eval_f1'].mean():.4f} ± {df['best_eval_f1'].std():.4f}")
    print(f"  Median Eval F1: {df['best_eval_f1'].median():.4f}")
    print(f"  Min Eval F1: {df['best_eval_f1'].min():.4f}")
    print(f"  Max Eval F1: {df['best_eval_f1'].max():.4f}")

    return df

def main():
    """Main analysis function"""

    # Get the most recent results file
    results_dir = os.path.join(utils.get_results_path(), 'full_info_experiments')

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]

    if not result_files:
        print("No result files found")
        return

    # Sort by modification time, most recent first
    result_files = sorted(result_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)

    print(f"Found {len(result_files)} result file(s)")
    print()
    print("Available files:")
    for i, f in enumerate(result_files):
        print(f"  {i}: {f}")

    print()
    file_idx = input(f"Enter file index to analyze (0-{len(result_files)-1}, or press Enter for most recent): ").strip()

    if file_idx == '':
        file_idx = 0
    else:
        file_idx = int(file_idx)

    results_file = os.path.join(results_dir, result_files[file_idx])
    print(f"\nAnalyzing: {results_file}\n")

    # Load results
    results_data = load_results(results_file)

    # Analyze based on type
    if results_data['ibm_hp']:
        analyze_single_run(results_data)
    else:
        analyze_hp_search(results_data)

if __name__ == '__main__':
    main()
