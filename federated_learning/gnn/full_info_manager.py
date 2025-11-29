from .manager_mixin import GNNMixinManager
import copy
import logging
import numpy as np
import utils
import configs.configs as configs
from inference import metrics, predictions_helper
import inference as flin
from training.utils import ibm_gnn

logger = logging.getLogger(__name__)


class FullInfoGNNManager(GNNMixinManager):
    """Full information GNN Manager - single party with complete dataset."""
    
    def __init__(self, args):
        super().__init__(args)
        self._party = None  # Single party reference
    
    
    def add_party(self, party):
        """Override to ensure only one party."""
        if self._party is not None:
            raise ValueError("FullInfoGNNManager only supports single party")
        super().add_party(party)
        self._party = party

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup single party with all data."""
        logger.info("Setting up Full Info model (single party with complete dataset)")
        self._add_party(None, df, parsers, scaler_encoders)
        logger.info("Starting hyperparameter tuning")
        tuned_hp, _ = self.tuning(laundering_values)
        logger.info("Setup complete")
        return tuned_hp
    
    def tuning(self, laundering_values):
        """Simplified tuning for single party."""

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
        """Tuning loop for single party."""
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
        """Train with seed loop."""
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
    
    def _train(self, laundering_values):
        """Training loop for single party - no loops over parties."""
        best_metrics = None
        best_pred_label = None
        best_pred_probabilities = None
        best_f1 = -1
        best_model = None

        # reset predictions
        laundering_values['pred_probabilities'] = 0
        laundering_values['pred_label'] = 0

        epochs = 20 if self.args['data_parser'].testing else configs.epochs
        logger.info("Training for %d epochs (evaluating every 20 epochs)", epochs)

        for epoch in range(epochs):
            loss = self._party.update_local_w()

            #if (epoch + 1) % 20 == 0:

            pred_probabilities = self._party.model.predict(self._party.get_eval_data())
            tmp_metrics = metrics(y_true = laundering_values['true_y'],
                                    y_pred_probabilities = pred_probabilities)

            # Debug logging every 20 epochs
            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d - Loss: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f",
                           epoch + 1, epochs, loss, tmp_metrics['f1'], tmp_metrics['precision'],
                           tmp_metrics['recall'], tmp_metrics['roc_auc'])
                logger.debug("  Prediction stats - Min: %.4f, Max: %.4f, Mean: %.4f, Median: %.4f",
                            pred_probabilities.min(), pred_probabilities.max(),
                            pred_probabilities.mean(), np.median(pred_probabilities))
                logger.debug("  True label distribution - Positives: %d (%.2f%%), Negatives: %d",
                            laundering_values['true_y'].sum(),
                            100 * laundering_values['true_y'].mean(),
                            len(laundering_values['true_y']) - laundering_values['true_y'].sum())

            if tmp_metrics['f1'] > best_f1:
                best_metrics = tmp_metrics
                best_pred_label = predictions_helper(pred_probabilities)
                best_pred_probabilities = copy.deepcopy(pred_probabilities)
                best_model = copy.deepcopy(self._party.model.gnn.state_dict())
                best_f1 = tmp_metrics['f1']
                logger.debug("New best F1: %.4f at epoch %d", best_f1, epoch + 1)

        if best_model is None:
            logger.error("No evaluation occurred during training (epochs=%d). Check evaluation frequency.", epochs)
            raise ValueError(f"No evaluation occurred during training (epochs={epochs}). Check evaluation frequency.")

        # Update laundering values with best predictions
        laundering_values['pred_probabilities'] = best_pred_probabilities
        laundering_values['pred_label'] = best_pred_label

        logger.info("Training complete - Best F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                   best_metrics['f1'], best_metrics['precision'], best_metrics['recall'],
                   best_metrics['roc_auc'], best_metrics['pr_auc'])

        # Warnings for unusual results
        if best_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", best_metrics['f1'])
        if best_metrics['precision'] == 0 or best_metrics['recall'] == 0:
            logger.warning("Zero precision or recall - Model may not be learning properly")

        return {'metrics': best_metrics,
                'laundering_values': copy.deepcopy(laundering_values),
                'model': best_model}
    

