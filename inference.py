from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def probs_to_binary(y_pred_probabilities, threshold = 0.5):
    return (y_pred_probabilities > threshold) * 1

def metrics(y_true, y_pred_probabilities = None, y_pred_binary = None):

    if y_pred_binary is None:
        y_pred_binary = probs_to_binary(y_pred_probabilities)

    # Check for NaN values before processing
    if isinstance(y_pred_probabilities, pd.Series):
        nan_count = y_pred_probabilities.isna().sum()
        if nan_count > 0:
            logger.warning("Found %d NaN values in predictions (%.2f%%), replacing with 0",
                          nan_count, 100 * nan_count / len(y_pred_probabilities))
        y_pred_probabilities = y_pred_probabilities.fillna(0).values
    else:
        nan_count = np.isnan(y_pred_probabilities).sum()
        if nan_count > 0:
            logger.warning("Found %d NaN values in predictions (%.2f%%), replacing with 0",
                          nan_count, 100 * nan_count / len(y_pred_probabilities))
        y_pred_probabilities = np.nan_to_num(y_pred_probabilities, nan=0.0)

    # roc_auc_score and average_precision_score require both classes in y_true
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        logger.warning("Only one class present in y_true - ROC-AUC and PR-AUC are undefined")

    results = {
        'f1': f1_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'precision': precision_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'recall': recall_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'roc_auc': roc_auc_score(y_true, y_pred_probabilities) if n_classes >= 2 else 0.0,
        'pr_auc': average_precision_score(y_true, y_pred_probabilities) if n_classes >= 2 else 0.0
    }

    # Warn about unusual metric values
    if results['f1'] == 0 and results['precision'] == 0 and results['recall'] == 0:
        logger.warning("All metrics are zero - model may not be making any positive predictions")
    elif results['precision'] == 0:
        logger.warning("Precision is zero - all positive predictions are false positives")
    elif results['recall'] == 0:
        logger.warning("Recall is zero - model is not detecting any true positives")

    if results['roc_auc'] < 0.5:
        logger.warning("ROC-AUC < 0.5 (%.4f) - model performs worse than random", results['roc_auc'])

    return results


def update_laundering_values(party, laundering_values, pred_probabilities=None, mode='vali'):
    """Update laundering values with predictions from a party.

    Args:
        party: Party object with model for predictions
        laundering_values: DataFrame to update with predictions
        pred_probabilities: Pre-computed predictions (optional). If None, will compute via predict.
        mode: 'vali' for validation data (default), 'test' for test data.
    """

    # Get indices based on mode
    get_indices_fn = party.get_test_indices if mode == 'test' else party.get_vali_indices

    # make predictions
    if pred_probabilities is None:
        pred_probabilities = party.get_predictions(mode=mode)

    # Check for unusual predictions
    if np.isnan(pred_probabilities).any():
        nan_count = np.isnan(pred_probabilities).sum()
        logger.warning("Party has %d NaN predictions (%.2f%%)",
                      nan_count, 100 * nan_count / len(pred_probabilities))

    if np.all(pred_probabilities == 0):
        logger.warning("Party has all zero predictions - model may not be learning")

    pred_labels = probs_to_binary(pred_probabilities)
    original_indices = get_indices_fn()

    # update laundering values, where observations have been predicted as 1
    pred_labels_df = pd.DataFrame({'original_indices': original_indices, 'pred_label': pred_labels})
    positive_indices = pred_labels_df['original_indices'][np.array(pred_labels) == 1]
    update_mask = laundering_values['indices'].isin(positive_indices)
    laundering_values.loc[update_mask, 'pred_label'] = 1

    # update observations max probabilities and average

    # get update mask
    update_mask = laundering_values['indices'].isin(original_indices)
    pred_probabilities = np.asarray(pred_probabilities)

    # update max values
    laundering_values.loc[update_mask, 'max_prob'] = np.maximum(laundering_values.loc[update_mask, 'max_prob'].values,
                                                                pred_probabilities)

    # update average probabilities
    current_num = laundering_values.loc[update_mask,'num_prob'].values
    current_avg = laundering_values.loc[update_mask,'avg_prob'].values

    # update values
    new_num = current_num + 1
    new_avg = (current_num * current_avg + pred_probabilities) / new_num

    # update
    laundering_values.loc[update_mask,'num_prob'] = new_num
    laundering_values.loc[update_mask,'avg_prob'] = new_avg


