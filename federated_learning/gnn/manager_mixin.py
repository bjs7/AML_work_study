"""Shared GNN Manager functionality for hyperparameter tuning and model initialization."""

from federated_learning.hp_tuning import ibm_gnn
import federated_learning.hp_tuning as tr_utils
from models.gnn_base import GNN
from result_io.save_results import build_save_dir, save_seed_result
from configs.paths import get_full_info_hp_path
from pathlib import Path
import json
import logging
import os
import pickle
import utils
import copy
import torch

logger = logging.getLogger(__name__)


class GNNMixinManager:

    def _prep_parties_data(self):
        for _, party in self.parties.items():
            party.prep_data()

    def train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds
        results_by_seed = {}

        logger.info("="*80)
        logger.info("Starting training with %d seeds", seeds)
        logger.info("="*80)

        self._prep_parties_data()
        self.save_dir = build_save_dir(self, hyperparameters)

        first_seed = getattr(self.args['data_parser'], 'first_seed', 1)
        for seed in range(seeds):
            seed_value = seed + first_seed
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(
                hyperparameters, copy.deepcopy(laundering_values_vali), copy.deepcopy(laundering_values_test))

            save_seed_result(self.save_dir, seed_value, results_by_seed[seed_value], self)

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                       seed_value,
                       results_by_seed[seed_value]['metrics']['f1'],
                       results_by_seed[seed_value]['metrics']['roc_auc'],
                       results_by_seed[seed_value]['metrics']['pr_auc'])

        logger.info("\n" + "="*80)
        logger.info("All seeds completed")
        logger.info("="*80)

        # If a parallel job ran other seeds in the same directory (via --run_id +
        # --first_seed), load their results so the final aggregation covers all seeds.
        for pkl_file in sorted(self.save_dir.glob('seed_*/metrics_laundering_values.pkl')):
            seed_num = int(pkl_file.parent.name.split('_')[1])
            if seed_num not in results_by_seed:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                results_by_seed[seed_num] = {'metrics': data['metrics']}

        return results_by_seed

    def init_hyperparams(self, sample_intervals = None):
            
        x_0 = tr_utils.get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]['x_0']
        hp_list = [tr_utils.hyper_sampler(self.args['fl_parser'], None, sample_intervals) for i in range(x_0)]

        return hp_list
    
    def init_models(self, hyperparams, bank_id=None):

        sample_party = next(iter(self.parties.values()))
        node_features = sample_party.procs_data['train_data']['df'].x.shape[1]
        edge_dim = sample_party.procs_data['train_data']['df'].edge_attr.shape[1]
        # Exclude ID column from edge dimension for FedGraph
        edge_dim -= self.edge_feat_start
        device = None

        if bank_id is not None:
            if torch.cuda.is_available():
                gpu_idx = self.bank_device.get(bank_id, 0)
                device = torch.device(f"cuda:{gpu_idx}")
            self.parties[bank_id].model = GNN(self, hyperparams, node_features, edge_dim, device=device)
        else:
            #available_gpus = get_available_gpus()
            include_test = self.mode == 'training'
            for idx, (bank_id, party) in enumerate(self.iter_parties(include_test=include_test)):
                if torch.cuda.is_available():
                    gpu_idx = self.bank_device.get(bank_id, 0)
                    device = torch.device(f"cuda:{gpu_idx}")
                    #device = get_device_for_party(idx, available_gpus)
                party.model = GNN(self, hyperparams, node_features, edge_dim, device=device)

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


    @staticmethod
    def _get_search_space(hyperparameters_tuning):

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


class GNNMixinManagerBaseline(GNNMixinManager):

    def _save_tuned_hp(self, hp):
        path = get_full_info_hp_path(self.args)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(hp, f, indent=4)
        logger.info("Saved full_info GNN hyperparameters to %s", path)

    def _load_tuned_hp(self):
        path = get_full_info_hp_path(self.args)
        if os.path.exists(path):
            with open(path, 'r') as f:
                hp = json.load(f)
            logger.info("Loaded full_info GNN hyperparameters from %s", path)
            return hp
        logger.warning("No saved GNN hyperparameters found at: %s", path)
        return None

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

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
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
