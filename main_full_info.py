# packages
import pandas as pd
from data.raw_data_processing import get_data
from configs.configs import split_perc
import utils
from data.get_indices_type_data import get_indices_bdt
import logging
import copy
import data.data_functions as dfn
from federated_learning.registry import FL_ALGO_REGISTRY_MANAGER, FL_ALGO_REGISTRY_PARTY, FL_REG_MODEL_REGISTRY
from federated_learning.registry import regi_algo_manager, regi_algo_party
import models.gnn_models
from federated_learning.fl_base import Manager, Party
import federated_learning.fl_algos
import data.feature_engi as fe
from relbanks_saving_analysis.save_results import save_results
from models.gnn_models import GINe
from training.utils import ibm_gnn
from models.gnn import add_arange_ids
import torch
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from models.gnn import batching_masker
from training.utils import hyper_sampler
import pickle
import os
import numpy as np

logger = logging.getLogger(__name__)

def evaluate(loader, inds, model, data, device):

    preds = []
    ground_truths = []
    pred_probs = []
    model.eval()

    #total_missing_edges = 0

    for batch in loader:
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        missing_seed_edges = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing_seed_edges.sum() != 0:
            #total_missing_edges += missing_seed_edges.sum().item()
            missing_ids = batch_edge_ids[missing_seed_edges].int()

            edge_labels_add = batch.edge_label_index[:,missing_seed_edges].detach().clone()
            edge_attr_add = data.edge_attr[missing_ids, :].detach().clone()
            #y_add = data.y[missing_ids].detach().clone()
            add_y = data.y[missing_ids].detach().clone()

            batch.edge_index = torch.cat([batch.edge_index, edge_labels_add], dim=1)
            batch.edge_attr = torch.cat([batch.edge_attr, edge_attr_add], dim=0)
            batch.y = torch.cat([batch.y, add_y], dim=0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            probs = torch.softmax(out, dim=-1)[:, 1]  # Probability of positive class
            preds.append(pred)
            ground_truths.append(batch.y[mask])
            pred_probs.append(probs)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    pred_prob = torch.cat(pred_probs, dim=0).cpu().numpy()

    # Calculate metrics
    f1 = f1_score(ground_truth, pred)
    precision = precision_score(ground_truth, pred, zero_division=0)
    recall = recall_score(ground_truth, pred, zero_division=0)
    roc_auc = roc_auc_score(ground_truth, pred_prob) if len(np.unique(ground_truth)) > 1 else 0.5

    # Log evaluation statistics
    logger.info(f"Evaluation - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Evaluation - Positive predictions: {pred.sum()}/{len(pred)} ({100*pred.mean():.2f}%)")
    logger.info(f"Evaluation - True positives in data: {ground_truth.sum()}/{len(ground_truth)} ({100*ground_truth.mean():.2f}%)")
    logger.info(f"Evaluation - Prediction probs - Min: {pred_prob.min():.4f}, Max: {pred_prob.max():.4f}, Mean: {pred_prob.mean():.4f}")

    #if total_missing_edges > 0:
        #logger.warning(f"Evaluation - Total missing seed edges added back: {total_missing_edges}")

    return {'f1': f1, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc,
            'pred': pred, 'ground_truth': ground_truth, 'pred_prob': pred_prob}


def train(df, hyperparameters):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_data = copy.deepcopy(df['train_data']['df'])
    train_indices = copy.deepcopy(df['train_data']['pred_indices'])

    eval_data = copy.deepcopy(df['vali_data']['df'])
    eval_pred_indices = copy.deepcopy(df['vali_data']['pred_indices'])

    logger.info(f"Train data - Nodes: {train_data.num_nodes}, Edges: {train_data.edge_index.shape[1]}")
    logger.info(f"Train data - Positive class: {train_data.y.sum().item()}/{len(train_data.y)} ({100*train_data.y.float().mean():.2f}%)")
    logger.info(f"Eval data - Nodes: {eval_data.num_nodes}, Edges: {eval_data.edge_index.shape[1]}")
    logger.info(f"Eval indices: {len(eval_pred_indices)}")

    model = GINe(num_features=1, num_gnn_layers=hyperparameters['num_gnn_layers'], n_classes=2,
                n_hidden=hyperparameters['hidden_embedding_size'], dropout=hyperparameters['dropout'],
                final_dropout=hyperparameters['final_dropout'],
                edge_dim=train_data.edge_attr.shape[1])

    model.to(device)
    logger.info(f"Model: GINe with {hyperparameters['num_gnn_layers']} layers, hidden={hyperparameters['hidden_embedding_size']}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    add_arange_ids([train_data, eval_data])

    num_neighbors = [100]*ibm_gnn['num_gnn_layers']
    train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors,
                                        edge_label_index = train_data.edge_index,
                                        edge_label = train_data.y,
                                        batch_size=8192, shuffle=True, transform=None)

    eval_loader = LinkNeighborLoader(eval_data, num_neighbors=num_neighbors,
                                        edge_label_index=eval_data.edge_index[:, eval_pred_indices],
                                        edge_label=eval_data.y[eval_pred_indices],
                                        batch_size=8192, shuffle=False, transform=None)

    logger.info(f"Using LinkNeighborLoader with num_neighbors={num_neighbors}, batch_size=8192")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparameters['w_ce1'],
                                                                    hyperparameters['w_ce2']]).to(device))

    logger.info(f"Loss weights: w_ce1={hyperparameters['w_ce1']}, w_ce2={hyperparameters['w_ce2']}")

    best_train_f1 = 0
    best_eval_f1 = 0
    best_eval_results = None
    best_model_state = None
    epochs_history = []

    num_epochs = 100
    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):

        logger.info(f"=" * 80)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        batch_losses = []
        model.train()

        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()

            train_indices = train_indices.detach().cpu()
            batch_edge_inds = train_indices[batch.input_id.detach().cpu()]
            batch_edge_ids = train_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]

            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch.edge_attr = batch.edge_attr[:, 1:]

            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx}! Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            batch_losses.append(float(loss))

        # Training metrics
        avg_loss = total_loss / total_examples
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()

        train_f1 = f1_score(ground_truth, pred)
        train_precision = precision_score(ground_truth, pred, zero_division=0)
        train_recall = recall_score(ground_truth, pred, zero_division=0)

        if train_f1 > best_train_f1:
            best_train_f1 = train_f1

        # Log training stats
        logger.info(f"Train - Loss: {avg_loss:.4f} (min: {min(batch_losses):.4f}, max: {max(batch_losses):.4f})")
        logger.info(f"Train - F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        logger.info(f"Train - Positive predictions: {pred.sum()}/{len(pred)} ({100*pred.mean():.2f}%)")
        logger.info(f"Train - True positives: {ground_truth.sum()}/{len(ground_truth)} ({100*ground_truth.mean():.2f}%)")

        # Evaluation
        logger.info(f"Running evaluation...")
        eval_results = evaluate(eval_loader, eval_pred_indices, model, eval_data, device)

        if eval_results['f1'] > best_eval_f1:
            best_eval_f1 = eval_results['f1']
            best_eval_results = eval_results
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"*** New best eval F1: {best_eval_f1:.4f} at epoch {epoch + 1} ***")

        # Store epoch history
        epochs_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'eval_f1': eval_results['f1'],
            'eval_precision': eval_results['precision'],
            'eval_recall': eval_results['recall'],
            'eval_roc_auc': eval_results['roc_auc']
        })

        # Log every 10 epochs or if significant improvement
        if (epoch + 1) % 10 == 0:
            logger.info(f"Progress summary at epoch {epoch + 1}:")
            logger.info(f"  Best train F1 so far: {best_train_f1:.4f}")
            logger.info(f"  Best eval F1 so far: {best_eval_f1:.4f}")

    logger.info(f"=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best train F1: {best_train_f1:.4f}")
    logger.info(f"Best eval F1: {best_eval_f1:.4f}")

    return {
        'best_train_f1': best_train_f1,
        'best_eval_f1': best_eval_f1,
        'best_eval_results': best_eval_results,
        'best_model_state': best_model_state,
        'epochs_history': epochs_history,
        'hyperparameters': hyperparameters
    }


def main():

    # Setup logger ----------------------------------------------------------------------------------------

    utils.logger_setup()

    # Get parsers and data ----------------------------------------------------------------------------------------

    # Parsers data -------
    #parsers, df, scaler_encoders = utils.setup_get_data()
    parsers = utils.parser_all()
    utils.set_seed(parsers['data_parser'].seed, True)

    parsers['fl_parser'].fl_algo = 'full_info'
    parsers['data_parser'].ibm_fe = True

    logger.info("="*100)
    logger.info("Starting main_full_info experiment")
    logger.info(f"Seed: {parsers['data_parser'].seed}")
    logger.info(f"Data size: {parsers['data_parser'].size}, IR: {parsers['data_parser'].ir}")
    logger.info(f"Using IBM hyperparameters: {parsers['data_parser'].ibm_hp}")
    logger.info("="*100)

    df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")

    df, scaler_encoders  = get_data(df, parsers['data_parser'], split_perc = split_perc)
    df = df['graph_data']

    if parsers['data_parser'].ibm_hp:
        hyperparameters = ibm_gnn
        logger.info("Training with IBM hyperparameters")
        results = train(df, hyperparameters)
    else:
        results = {}
        logger.info("Running hyperparameter search with 50 trials")
        for i in range(50):
            logger.info(f"\n{'='*80}")
            logger.info(f"Hyperparameter trial {i+1}/50")
            logger.info(f"{'='*80}")
            hyperparameters = hyper_sampler(parsers['fl_parser'])
            results[i] = train(df, hyperparameters)

    # Save results
    results_dir = os.path.join(utils.get_results_path(), 'full_info_experiments')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'results_{timestamp}.pkl')

    logger.info(f"\n{'='*100}")
    logger.info(f"Saving results to: {results_file}")

    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'parsers': parsers,
            'timestamp': timestamp,
            'seed': parsers['data_parser'].seed,
            'data_size': parsers['data_parser'].size,
            'ir': parsers['data_parser'].ir,
            'ibm_hp': parsers['data_parser'].ibm_hp
        }, f)

    logger.info("Results saved successfully!")

    # Print summary
    logger.info(f"\n{'='*100}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*100}")

    if parsers['data_parser'].ibm_hp:
        logger.info(f"Best Train F1: {results['best_train_f1']:.4f}")
        logger.info(f"Best Eval F1: {results['best_eval_f1']:.4f}")
        if results['best_eval_results']:
            logger.info(f"Best Eval Precision: {results['best_eval_results']['precision']:.4f}")
            logger.info(f"Best Eval Recall: {results['best_eval_results']['recall']:.4f}")
            logger.info(f"Best Eval ROC-AUC: {results['best_eval_results']['roc_auc']:.4f}")
    else:
        best_trial = max(results.items(), key=lambda x: x[1]['best_eval_f1'])
        logger.info(f"Best trial: {best_trial[0]}")
        logger.info(f"Best Eval F1: {best_trial[1]['best_eval_f1']:.4f}")
        logger.info(f"Hyperparameters: {best_trial[1]['hyperparameters']}")

    logger.info(f"{'='*100}\n")

    return results

if __name__ == '__main__':
    main()

