"""Shared GNN Manager functionality for hyperparameter tuning and model initialization."""

from training.utils import ibm_gnn
import training.utils as tr_utils
from models.gnn import GNN
import logging
import utils
import copy

logger = logging.getLogger(__name__)


class GNNMixinManager:

    def init_hyperparams(self, sample_intervals = None):
            
        x_0 = tr_utils.get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]['x_0']
        hp_list = [tr_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    
    def init_models(self, hyperparams, bank_id = None, gnn_batching = False):

        sample_party = next(iter(self.parties.values()))
        node_features = sample_party.procs_data['train_data']['df'].x.shape[1]
        edge_dim = sample_party.procs_data['train_data']['df'].edge_attr.shape[1]
        if gnn_batching:
            edge_dim -= 1
   
        if bank_id is not None:
            self.parties[bank_id].model = GNN(self, hyperparams, node_features, edge_dim)
        else:
            for bank_id, party in self.parties.items():
                party.model = GNN(self, hyperparams, node_features, edge_dim)

        # need to 'save' parameters from model 1 as starting, or not, because the models will vary
        # Here get the relevant parameters etc. or it can be placed in get_global_w

    def _gnn_tuning(self, laundering_values, **kwargs):
        
        # first loop
        hyperparameters_tuning = self.init_hyperparams()
        _, scores, _ = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        # get top 5 of parameters and set sample space
        params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
        sample_space = self._get_search_space(top_parameters)

        # second loop
        hyperparameters_tuning = self.init_hyperparams(sample_space)
        tuned_hyparameters, _, f1_score_for_hp = self.tuning_loop(hyperparameters_tuning, laundering_values, **kwargs)

        return tuned_hyparameters, f1_score_for_hp


    def _get_search_space(self, hyperparameters_tuning):

        intervals = {
            'hid_em_size_interval': [float('inf'), float('-inf')],
            'lr_interval': [float('inf'), float('-inf')],
            'gnn_layer_interval': [float('inf'), float('-inf')],
            'dropout_interval': [float('inf'), float('-inf')],
            'final_dropout_interval': [float('inf'), float('-inf')],
            'w_ce2_interval': [float('inf'), float('-inf')],
        }

        for params in hyperparameters_tuning:
            intervals['hid_em_size_interval'] = tr_utils.update_interval(params['hidden_embedding_size'], intervals['hid_em_size_interval'])
            intervals['lr_interval'] = tr_utils.update_interval(params['learning_rate'], intervals['lr_interval'])
            intervals['gnn_layer_interval'] = tr_utils.update_interval(params['num_gnn_layers'], intervals['gnn_layer_interval'])
            intervals['dropout_interval'] = tr_utils.update_interval(params['dropout'], intervals['dropout_interval'])
            intervals['final_dropout_interval'] = tr_utils.update_interval(params['final_dropout'], intervals['final_dropout_interval'])
            intervals['w_ce2_interval'] = tr_utils.update_interval(params['w_ce2'], intervals['w_ce2_interval'])

        return intervals


class GNNMixinManager_Fullinfo_Indi(GNNMixinManager):

    def tuning_loop(self, hyperparameters_tuning, laundering_values, **kwargs):

        best_hyperparameters, best_f1, scores = None, -1, []

        for hyperparams in hyperparameters_tuning:
            
            self.init_models(hyperparams, **kwargs) 
            results = self._train_party(laundering_values, **kwargs)
               
            if results['f1'] > best_f1:
                best_hyperparameters = hyperparams
                best_f1 = results['f1']

            scores.append(results['f1'])

        return best_hyperparameters, scores, best_f1

    def _train_party(self, laundering_values, **kwargs):
        raise NotImplementedError
    
    def train(self, hyperparameters, laundering_values):

        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds

        results_by_seed = {}
        bank_str = f'{len(self.parties)} banks' if self.args['fl_parser'].fl_algo != 'full_info' else 'full info'

        logger.info("="*80)
        logger.info("Starting training with %d seeds for %s", seeds, bank_str)
        logger.info("="*80)

        for bank_id, party in self.parties.items():
            party.prep_data()

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train_helper(hyperparameters, copy.deepcopy(laundering_values))

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                       seed_value,
                       results_by_seed[seed_value]['metrics']['f1'],
                       results_by_seed[seed_value]['metrics']['roc_auc'],
                       results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)
        
        return results_by_seed
    
    def _train_helper(self, hyperparameters, laundering_values):
        raise NotImplementedError

    def _tuning_helper(self, laundering_values, party, bank_id):
        raise NotImplementedError

    def tuning(self, laundering_values):

        results = {}
        self.set_mode('tuning')

        if self.args['data_parser'].ibm_hp:
            logger.info("Using IBM hyperparameters (skipping tuning)")
            tuned_hyparameters, f1_score_for_hp = ibm_gnn, 1

            for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
                results[bank_id] = {'hyperparameters': tuned_hyparameters, 'f1_score': f1_score_for_hp}
        
        else:
            bank_str = f'{len(self.parties)} banks' if self.args['fl_parser'].fl_algo != 'full_info' else 'full info'
            logger.info("Running hyperparameter tuning for %s", bank_str)

            for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
                if self.args['fl_parser'].fl_algo == 'full_info':
                    logger.info("Tuning full_info")
                else:
                    logger.info("Tuning bank %s (%d/%d)", bank_id, idx, len(self.parties))

                party.prep_data()
                tuned_hyparameters, f1_score_for_hp = self._tuning_helper(laundering_values, party, bank_id)   
                
                if self.args['fl_parser'].fl_algo == 'full_info':
                    logger.info("Tuning complete - Best F1: %.4f", f1_score_for_hp)
                else:
                    logger.info("Bank %s: Best F1=%.4f", bank_id, f1_score_for_hp)

                results[bank_id] = {'hyperparameters': tuned_hyparameters, 'f1_score': f1_score_for_hp}

        return results
