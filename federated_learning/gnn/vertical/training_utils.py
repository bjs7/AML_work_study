"""Training utility functions for vertical federated learning."""

import copy
import logging
import torch
from sklearn.metrics import f1_score
from inference import metrics, probs_to_binary

logger = logging.getLogger(__name__)


def update_best_model(f1_eval, best_f1, best_model_state, labels, preds, model, epoch):
    """Update best model state if current F1 is better.

    Args:
        f1_eval: Current evaluation F1 score
        best_f1: Best F1 score so far
        best_model_state: Current best model state dict
        labels: Ground truth labels
        preds: Prediction probabilities
        model: Current model
        epoch: Current epoch number

    Returns:
        Tuple of (updated best_f1, updated best_model_state)
    """
    if f1_eval > best_f1:
        best_model_state = {
            'best_pred_probabilities': copy.deepcopy(preds),
            'best_pred_binary': probs_to_binary(preds),
            'best_ground_truths': copy.deepcopy(labels),
            'best_model': copy.deepcopy(model.gnn.state_dict())
        }

        best_f1 = f1_eval
        logger.debug("New best F1: %.4f at epoch %d", best_f1, epoch + 1)

    return best_f1, best_model_state


def log_train_performance(labels, preds, train_loss, epoch, epochs):
    """Log training performance metrics.

    Args:
        labels: List of label tensors
        preds: List of prediction tensors
        train_loss: Total training loss for the epoch
        epoch: Current epoch number
        epochs: Total number of epochs
    """
    preds_np = torch.cat(preds, dim=0).argmax(dim=-1).detach().cpu().numpy()
    labels_np = torch.cat(labels).detach().cpu().numpy()

    f1 = f1_score(labels_np, preds_np)
    logger.info("Epoch %d/%d - Train Loss: %.4f, F1: %.4f", epoch + 1, epochs, train_loss, f1)


def prep_eval_preds_labels(labels, preds):
    """Prepare evaluation predictions and labels.

    Args:
        labels: List of label tensors
        preds: List of prediction tensors

    Returns:
        Tuple of (labels_np, preds_probs, preds_binary)
    """
    preds_cat = torch.cat(preds, dim=0)
    preds_probs = torch.softmax(preds_cat, dim=1)[:, 1].detach().cpu().numpy()

    preds_binary = probs_to_binary(preds_probs)
    labels_np = torch.cat(labels).detach().cpu().numpy()

    return labels_np, preds_probs, preds_binary


def compute_final_metrics(best_model_state):
    """Compute final metrics from best model state.

    Args:
        best_model_state: Dict containing best predictions and labels

    Returns:
        Dict of metrics (f1, roc_auc, pr_auc, etc.)
    """
    return metrics(
        y_true=best_model_state['best_ground_truths'],
        y_pred_probabilities=best_model_state['best_pred_probabilities']
    )
