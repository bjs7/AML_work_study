from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def predictions_helper(predictions, threshold = 0.5):
    return (predictions >= threshold) * 1

def metrics(y_true, y_pred_binary):
    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division = 0)
    accuracy = accuracy_score(y_true, y_pred_binary)
    return {'f1': f1, 'accuracy': accuracy}

# only f1 and auccuracy are including, because they use binary values
# if we are to include ROC/AUC probability predictions would also be needed
# Maybe include ROC/AUC too? Because it might be relevant to look at observations that are "close enough",
# like above a certain threshold
def update_laundering_values(party, laundering_values):

    predictions = party.model.predict_binary(party.get_eval_data())

    if sum(np.array(predictions) == 1) == 0:
        return None
    
    original_indices = party.get_eval_indices()
    predictions_df = pd.DataFrame({'original_indices': original_indices, 'predictions': predictions})

    positive_indices = predictions_df['original_indices'][np.array(predictions) == 1]
    update_mask = laundering_values['indices'].isin(positive_indices)
    laundering_values.loc[update_mask, 'predictions_fl'] = 1




