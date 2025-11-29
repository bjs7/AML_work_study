from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
import pandas as pd
import numpy as np

def predictions_helper(y_pred_probabilities, threshold = 0.5):
    return (y_pred_probabilities > threshold) * 1

def metrics(y_true, y_pred_probabilities = None, y_pred_binary = None):

    if y_pred_binary is None:
        y_pred_binary = predictions_helper(y_pred_probabilities)

    if isinstance(y_pred_probabilities, pd.Series):
        y_pred_probabilities = y_pred_probabilities.fillna(0).values
    else:
        y_pred_probabilities = np.nan_to_num(y_pred_probabilities, nan=0.0)
    
    results = {
        'f1': f1_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'precision': precision_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'recall': recall_score(y_true=y_true, y_pred=y_pred_binary, average='binary', zero_division = 0),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'roc_auc': roc_auc_score(y_true, y_pred_probabilities),
        'pr_auc': average_precision_score(y_true, y_pred_probabilities)
    }

    return results


# only f1 and auccuracy are including, because they use binary values
# if we are to include ROC/AUC probability predictions would also be needed
# Maybe include ROC/AUC too? Because it might be relevant to look at observations that are "close enough",
# like above a certain threshold
def update_laundering_values(party, laundering_values, pred_probabilities=None):
    """Update laundering values with predictions from a party.

    Args:
        party: Party object with model for predictions
        laundering_values: DataFrame to update with predictions
        predictions: Pre-computed predictions (optional). If None, will compute via predict_binary.
                     Used to avoid redundant GNN forward passes when predictions are already available.
    """

    # make predictions and get prediction indicies
    if pred_probabilities is None: 
        pred_probabilities = party.model.predict(party.get_eval_data())
    pred_labels = predictions_helper(pred_probabilities)
    original_indices = party.get_eval_indices()

    # update laundering values, where observations have been predicted as 1
    pred_labels_df = pd.DataFrame({'original_indices': original_indices, 'pred_label': pred_labels})
    positive_indices = pred_labels_df['original_indices'][np.array(pred_labels) == 1]
    update_mask = laundering_values['indices'].isin(positive_indices)
    laundering_values.loc[update_mask, 'pred_label'] = 1


    # average and max probabilities
    #for txt in ['avg_prob', 'num_prob']:
        #if txt not in laundering_values.columns:
            #laundering_values[txt] = 0

    # update observations max probabilities and average

    # get update mask
    update_mask = laundering_values['indices'].isin(original_indices)

    # update max values
    laundering_values.loc[update_mask, 'max_prob'] = np.maximum(laundering_values.loc[update_mask, 'max_prob'],  
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


