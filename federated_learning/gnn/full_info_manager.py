from .manager_mixin import GNNMixinManager
import copy
import logging
import numpy as np
import pandas as pd
import utils
import configs.configs as configs
from inference import metrics, probs_to_binary
import inference as flin
from training.utils import ibm_gnn

from torch_geometric.loader import LinkNeighborLoader
from models.gnn import add_arange_ids, batching_masker
import torch
from models.gnn import get_loaders
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class FullInfoGNNManager(GNNMixinManager):
    """Full information GNN Manager - single party with complete dataset."""
    
    def __init__(self, args):
        super().__init__(args)
        self._party = None  # Single party reference
        self._train_for_final = self.args['data_parser'].train_for_final
        
    def add_party(self, party):
        if self._party is not None:
            raise ValueError("FullInfoGNNManager only supports single party")
        super().add_party(party)
        self._party = party

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        logger.info("Setting up Full Info model (single party with complete dataset)")
        self._add_party(None, df, parsers, scaler_encoders)
        
        logger.info("Starting hyperparameter tuning")
        tuned_hp, _ = self.tuning(laundering_values)
        logger.info("Setup complete")

        return tuned_hp
    
    def tuning(self, laundering_values):

        if self.args['data_parser'].ibm_hp:
            logger.info("Using IBM hyperparameters (skipping tuning)")
            return ibm_gnn, None

        logger.info("Running hyperparameter tuning for full info model")

        self.set_mode('tuning')
        self._party.prep_data()

        tuned_hp, f1_score = self._gnn_tuning(laundering_values)
        logger.info("Tuning complete - Best F1: %.4f", f1_score)

        if f1_score < 0.1:
            logger.warning("Very low F1 score during tuning: %.4f", f1_score)

        return tuned_hp, f1_score
    
    
    def tuning_loop(self, hyperparameters_tuning, laundering_values):
        
        best_f1 = -1
        best_hyperparameters = None
        scores = []
        
        for hyperparams in hyperparameters_tuning:
            self.init_models(hyperparams)
            results = self._train(laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']
            
            scores.append(results['metrics']['f1'])
        
        return best_hyperparameters, scores, best_f1
    
    def train(self, hyperparameters, laundering_values, seeds=4):
        
        self.set_mode('training')
        results_by_seed = {}

        logger.info("="*80)
        logger.info("Starting Full Info training with %d seeds", seeds)
        logger.info("="*80)

        self._party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1

            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)


            utils.set_seed(seed_value)
            self.init_models(hyperparameters)
            results_by_seed[seed_value] = self._train(copy.deepcopy(laundering_values))


            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                       seed_value,
                       results_by_seed[seed_value]['metrics']['f1'],
                       results_by_seed[seed_value]['metrics']['roc_auc'],
                       results_by_seed[seed_value]['metrics']['pr_auc'])
            
        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)

        return results_by_seed
    
    def _get_loaders(self):

        train_data = copy.deepcopy(self._party.procs_data['train_data']['df'])

        if (self.mode == 'tuning') or self.args['data_parser'].train_for_final:
            train_indices = self._party.data['train_data']['pred_indices']
        else:
            train_indices = torch.cat([self._party.data['train_data']['pred_indices'], 
                                       self._party.data['vali_data']['pred_indices']])

        eval_data = copy.deepcopy(self._party.procs_data['eval_data']['df'])
        eval_pred_indices = self._party.procs_data['eval_data']['pred_indices']

        add_arange_ids([train_data, eval_data])

        num_neighbors = [100]*self._party.model.gnn.num_gnn_layers
        train_loader, eval_loader = get_loaders(train_data, eval_data, eval_pred_indices, num_neighbors)

        return train_loader, eval_loader, train_data, eval_data, train_indices, eval_pred_indices
    
    def _train(self, laundering_values):
        
        best_pred_probabilities, best_model, best_f1 = None, None, -1
        laundering_values['pred_probabilities'], laundering_values['pred_label'] = 0, 0
    
        train_loader, eval_loader, train_data, eval_data, train_indices, eval_indices = self._get_loaders()
        epochs = 20 if self.args['data_parser'].testing else configs.epochs
        logger.info("Training for %d epochs", epochs)

        for epoch in range(epochs):

            preds, ground_truths, preds_eval = [], [], []
            ground_truths_eval, eval_pred_ids, total_loss = [], [], 0

            for batch in train_loader:
                mask, _ = batching_masker(batch, train_data, train_loader, train_indices)

                pred, true_y, loss = self._party.model.update_w(batch, mask)
                total_loss += loss

                preds.append(pred.argmax(dim=-1))
                ground_truths.append(true_y)

            if (epoch + 1) % 2 == 0:
                preds = torch.cat(preds, dim=0).detach().cpu().numpy()
                ground_truths = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
                f1 = f1_score(ground_truths, preds)
                logger.info("Epoch %d/%d - Loss: %.4f, F1: %.4f", epoch + 1, epochs, total_loss, f1)
                if len(ground_truths) != len(train_indices):
                    logger.warning("Difference in the size of ground_truths and train_data indices, %d and %d", 
                                   len(ground_truths), len(train_indices))

            for batch in eval_loader:
                mask, pred_ids = batching_masker(batch, eval_data, eval_loader, eval_indices)
                pred = self._party.model.predict(batch, mask)
                preds_eval.append(pred)
                ground_truths_eval.append(batch.y[mask])
                eval_pred_ids.append(pred_ids)

            preds_eval = torch.cat(preds_eval, dim=0).detach().cpu().numpy()
            preb_binary_eval = probs_to_binary(preds_eval)
            ground_truths_eval = torch.cat(ground_truths_eval, dim=0).detach().cpu().numpy()
            pred_ids = torch.cat(eval_pred_ids).detach().cpu().numpy()
            
            if len(ground_truths) != len(eval_indices):
                    logger.warning("Difference in the size of ground_truths and eval_data indices, %d and %d",
                                    len(ground_truths), len(eval_indices))

            f1_eval = f1_score(ground_truths_eval, preb_binary_eval)

            if (epoch + 1) % 2 == 0:
                logging.info(f'Test F1: {f1_eval}')

            if torch.isnan(torch.tensor(preds_eval)).any():
                logger.warning("Model predictions contain NaN values!")

            if f1_eval > best_f1:
                best_pred_probabilities = copy.deepcopy(preds_eval)
                best_pred_binary = probs_to_binary(preds_eval)
                best_ground_truths = copy.deepcopy(ground_truths_eval)
                best_pred_ids = copy.deepcopy(pred_ids)

                best_model = copy.deepcopy(self._party.model.gnn.state_dict())
                best_f1 = f1_eval
                logger.debug("New best F1: %.4f at epoch %d", best_f1, epoch + 1)

        if best_model is None:
            logger.error("No evaluation occurred during training (epochs=%d). Check evaluation frequency.", epochs)
            raise ValueError(f"No evaluation occurred during training (epochs={epochs}). Check evaluation frequency.")

        perform_metrics = metrics(y_true = best_ground_truths, y_pred_probabilities = best_pred_probabilities)

        df_launderings = pd.DataFrame(data = {'indices': best_pred_ids,'true_y': best_ground_truths, 
                                             'pred_probabilities': best_pred_probabilities, 
                                             'pred_label': best_pred_binary})
        df_launderings = df_launderings.sort_values(by=['indices'], ignore_index=True)


        if not np.all(laundering_values['true_y'] == df_launderings['true_y']):
            logging.warning("Difference in the true y from laudering_values and true y from df_launderings")

        logger.info("Training complete - Best F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                   perform_metrics['f1'], perform_metrics['precision'], perform_metrics['recall'],
                   perform_metrics['roc_auc'], perform_metrics['pr_auc'])

        if perform_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", perform_metrics['f1'])
        if perform_metrics['precision'] == 0 or perform_metrics['recall'] == 0:
            logger.warning("Zero precision or recall - Model may not be learning properly")

        return {'metrics': perform_metrics,
                'laundering_values': copy.deepcopy(df_launderings),
                'model': best_model}
    





"""

            #tmp_metrics = metrics(y_true = laundering_values['true_y'],
                                    #y_pred_probabilities = pred_probabilities)

        # test ---------
        test123 = pd.DataFrame(data={})
        test123 = pd.concat([test123, pd.DataFrame(data = {'indices': pred_ids, 'true_y': batch.y[mask], 'preds': pred})])


        test123.sort_values(by = ['indices'], ignore_index=False)
        test123.sort_values(by = ['indices'], ignore_index=True)
        laundering_values.iloc[:8192*3,:]

        #np.all(df_launderings['true_y'] == laundering_values['true_y'][:8192*5])
        #np.where(laundering_values.iloc[:8192*5,:]['true_y'] != test123.sort_values(by = ['indices'], ignore_index=True)['true_y'])

        np.where(df_launderings['true_y'] != laundering_values['true_y'][:8192*5])
        #(array([ 9154,  9915, 11356, 14050, 14399, 14933]),)

        i = 9154

        np.where(df_launderings['indices'] == 4223599)
        df_launderings.iloc[9150:9156]

        df_launderings.iloc[i-3:i+3]
        laundering_values.iloc[i-3:i+3]

        eval_data.y[4223599-3:4223599+3]
        eval_data.y[eval_indices][i-3:i+3]


"""


