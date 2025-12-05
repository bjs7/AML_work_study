"""Individual GNN Manager - trains each party independently with separate models."""

import copy
import logging
import numpy as np
import utils
import configs.configs as configs
from inference import metrics, probs_to_binary
import inference as flin
from .manager_mixin import GNNMixinManager
from relbanks_saving_analysis.relevant_banks import get_relevant_banks
from training.utils import ibm_gnn
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class IndividualGNNManager(GNNMixinManager):

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        """Setup fr_banks, tune them, then add sr_banks with best hyperparameters."""
        fr_banks, sr_banks = get_relevant_banks(parsers)

        # tmp 
        bank_to_get = [idx for idx in range(len(fr_banks)) if fr_banks[idx] == 68]
        fr_banks = [fr_banks[bank_to_get[0]]]
        sr_banks = []

        if parsers['data_parser'].testing:
            fr_banks = fr_banks[0:5]
            sr_banks = sr_banks[0:2]
            logger.info("Testing mode: Limited to %d fr_banks and %d sr_banks", len(fr_banks), len(sr_banks))
        else:
            logger.info("Production mode: Using %d fr_banks and %d sr_banks", len(fr_banks), len(sr_banks))

        # Add and tune fr_banks
        logger.info("Adding %d fr_banks to manager", len(fr_banks))
        utils.add_banks_to_manager(parsers, fr_banks, self, df, scaler_encoders)
        logger.info("Starting hyperparameter tuning for fr_banks")
        tuned_hp = self.tuning(laundering_values)
        logger.info("Hyperparameter tuning completed")

        # Add sr_banks with best hyperparameters
        if self.args['data_parser'].train_for_final:
            sr_banks = []
            logger.info("train_for_final=True: Skipping sr_banks")
        else:
            logger.info("Adding %d sr_banks with tuned hyperparameters", len(sr_banks))
        tuned_hp = utils.add_banks_to_manager(parsers, sr_banks, self, df, scaler_encoders, tuned_hp)

        logger.info("Setup complete: Total %d banks", len(self.parties))
        return tuned_hp

    def _helper_party_tuning(self, party, laundering_values):
        mask = np.isin(laundering_values['indices'], party.get_eval_indices())
        return laundering_values.iloc[mask,].reset_index(drop=True)

    def tuning(self, laundering_values):

        results = {}
        self.set_mode('tuning')

        if self.args['data_parser'].ibm_hp:
            logger.info("Using IBM hyperparameters (skipping tuning)")
        else:
            logger.info("Running hyperparameter tuning for %d banks", len(self.parties))

        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):

            if self.args['data_parser'].ibm_hp:
                tuned_hyparameters, f1_score_for_hp = ibm_gnn, 1
                logger.debug("Bank %s (%d/%d): Using IBM hyperparameters", bank_id, idx, len(self.parties))
            else:
                logger.info("Tuning bank %s (%d/%d)", bank_id, idx, len(self.parties))
                party.prep_data()
                party_laundering_values = self._helper_party_tuning(party, laundering_values)
                tuned_hyparameters, f1_score_for_hp = self._gnn_tuning(party_laundering_values, bank_id = bank_id)
                logger.info("Bank %s: Best F1=%.4f", bank_id, f1_score_for_hp)

                if f1_score_for_hp < 0.1:
                    logger.warning("Bank %s: Very low F1 score (%.4f) during tuning", bank_id, f1_score_for_hp)

            results[bank_id] = {'hyperparameters': tuned_hyparameters, 'f1_score': f1_score_for_hp}

        return results

    def tuning_loop(self, hyperparameters_tuning, party_laundering_values, bank_id):

        best_f1 = -1
        best_hyperparameters = None
        scores = []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams, bank_id)
            results = self.party_train(self.parties[bank_id], party_laundering_values)
            
            if results['metrics']['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['metrics']['f1']

            scores.append(results['metrics']['f1'])

        return best_hyperparameters, scores, best_f1
    

    def party_train(self, party, party_laundering_values):


        from torch_geometric.loader import LinkNeighborLoader
        from models.gnn import add_arange_ids, batching_masker
        import torch
        from models.gnn import get_loaders

        self.set_mode('training')

        for bank_id, party in self.parties.items():
            party.prep_data()

        #hyperparameters = tuned_hp
        hyperparameters  = {}
        hyperparameters[68] = {'learning_rate': 0.026563807693306837,
                                'hidden_embedding_size': 28, 'n_mlp_layers': 1, 
                                'num_gnn_layers': 3, 'loss': 'ce', 'w_ce1': 1.0000182882773443, 
                                'w_ce2': 7.075395081105593, 'norm_method': 'z_normalize', 
                                'dropout': 0.19692771081793753, 'final_dropout': 0.19692771081793753}
        
        laundering_values = laundering_values_test


        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
            self.init_models(hyperparameters[bank_id], bank_id)
            party_laundering_values = self._helper_party_tuning(party, laundering_values)
        
        train_data = copy.deepcopy(self.parties[68].procs_data['train_data']['df'])
        train_indices = copy.deepcopy(self.parties[68].procs_data['train_data']['pred_indices'])
        
        eval_data = copy.deepcopy(self.parties[68].procs_data['eval_data']['df'])
        eval_pred_indices = self.parties[68].procs_data['eval_data']['pred_indices']
    
        add_arange_ids([train_data, eval_data])

        #num_neighbors = [100]*self.parties[68].model.gnn.num_gnn_layers
        num_neighbors = [20,20,10]

        #train_loader, eval_loader = get_loaders(train_data, eval_data, eval_pred_indices, num_neighbors)
        train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors, 
                                        edge_label_index = train_data.edge_index,
                                        edge_label = train_data.y, 
                                        batch_size=4096, shuffle=True, transform=None)

        from models.gnn import GNN
        utils.set_seed(1)
        #model = GNN._create_gnn_model(manager, hyperparameters[68], 1, 6)
        model = GNN._create_gnn_model(manager, hyperparameters[68], 1, 33)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters[68]['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparameters[68]['w_ce1'], 
                                                                      hyperparameters[68]['w_ce2']]))
        

        for i in range(0, 100):
            
            total_loss = total_examples = 0
            preds = []
            ground_truths = []

            for batch in train_loader:
                #batch = next(iter(train_loader))

                train_indices = train_indices.detach().cpu()
                batch_edge_inds = train_indices[batch.input_id.detach().cpu()]
                batch_edge_ids = train_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]

                mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

                batch.edge_attr = batch.edge_attr[:, 1:]

                #out = self.parties[68].model.gnn(batch.x, batch.edge_index, batch.edge_attr)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                pred = out[mask]
                ground_truth = batch.y[mask]
                preds.append(pred.argmax(dim=-1))
                ground_truths.append(ground_truth)
                loss = loss_fn(pred, ground_truth)



                loss.backward()
                optimizer.step()
                #self.parties[68].model.optimizer.step()

                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

            pred = torch.cat(preds, dim=0).detach().cpu().numpy()
            ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
            f1 = f1_score(ground_truth, pred)
            print(total_loss)
            print(f1)



        # ------------------------------------------------
        best_metrics = None
        #best_preditcions = None

        best_pred_label = None
        best_pred_probabilities = None

        best_f1 = -1
        best_model = None

        party_laundering_values['pred_label'] = 0
        party_laundering_values['pred_probabilities'] = 0


        epochs = 20 if self.args['data_parser'].testing else configs.epochs

        for i in range(0, epochs):

            

            #party.update_local_w()

            if (i+1) % 20 == 0:  

                pred_probabilities = party.model.predict(party.get_eval_data())
                tmp_metrics = metrics(party_laundering_values['true_y'], pred_probabilities)
                
                if tmp_metrics['f1'] > best_f1:
                    best_metrics = tmp_metrics
                    best_pred_label = probs_to_binary(pred_probabilities)
                    best_pred_probabilities = copy.deepcopy(pred_probabilities)
                    best_model = copy.deepcopy(party.model.gnn.state_dict())
                    best_f1 = tmp_metrics['f1']

        party_laundering_values['pred_label'] = best_pred_label
        party_laundering_values['pred_probabilities'] = best_pred_probabilities
        
        return {'model': best_model, 'metrics': best_metrics, 'laundering_values': party_laundering_values}

    def train(self, hyperparameters, laundering_values, seeds = 4):

        self.set_mode('training')
        results_by_seed = {}

        logger.info("="*80)
        logger.info("Starting final training with %d seeds for %d banks", seeds, len(self.parties))
        logger.info("="*80)

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)
            # Pass a fresh copy to each seed iteration
            results_by_seed[seed_value] = self._train(hyperparameters, copy.deepcopy(laundering_values))
            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                       seed_value,
                       results_by_seed[seed_value]['metrics']['f1'],
                       results_by_seed[seed_value]['metrics']['roc_auc'],
                       results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)
        return results_by_seed

    def _train(self, hyperparameters, laundering_values):

        models_hyperparameters = {}
        party_predictions = {}

        logger.info("Training %d individual banks", len(self.parties))
        for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
            logger.debug("Training bank %s (%d/%d)", bank_id, idx, len(self.parties))
            self.init_models(hyperparameters[bank_id], bank_id)
            party_laundering_values = self._helper_party_tuning(party, laundering_values)

            # train individual party
            tmp_model = self.party_train(party, party_laundering_values)
            models_hyperparameters[bank_id] = {'model': tmp_model['model'],
                                               'hyperparameters': hyperparameters[bank_id]}

            party_predictions[bank_id] = tmp_model['laundering_values']['pred_probabilities']

            # Check for unusual predictions
            if np.all(party_predictions[bank_id] == 0):
                logger.warning("Bank %s: All predictions are zero!", bank_id)
            elif np.isnan(party_predictions[bank_id]).any():
                logger.warning("Bank %s: Predictions contain NaN values", bank_id)

        logger.info("Loading best models and aggregating predictions")
        for bank_id, party in self.parties.items():
            party.model.gnn.load_state_dict(models_hyperparameters[bank_id]['model'])
            flin.update_laundering_values(party, laundering_values, pred_probabilities=party_predictions[bank_id])

        collective_metrics = metrics(y_true = laundering_values['true_y'],
                                     y_pred_probabilities = laundering_values['avg_prob'],
                                     y_pred_binary = laundering_values['pred_label'])

        logger.info("Final metrics - F1: %.4f, Precision: %.4f, Recall: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                   collective_metrics['f1'], collective_metrics['precision'], collective_metrics['recall'],
                   collective_metrics['roc_auc'], collective_metrics['pr_auc'])

        # Warnings for unusual results
        if collective_metrics['f1'] < 0.1:
            logger.warning("Very low F1 score: %.4f - Check data and model configuration", collective_metrics['f1'])
        if collective_metrics['precision'] == 0 or collective_metrics['recall'] == 0:
            logger.warning("Zero precision or recall - Model may not be learning properly")

        return {
            'metrics': collective_metrics,
            'laundering_values': copy.deepcopy(laundering_values),
            'models': models_hyperparameters
        }




