import copy
import os
import logging
import training.utils as tr_utils
import utils
from models.booster import Booster
from training.utils import hyper_sampler, f1_eval
from configs.paths import get_tuning_configs
import xgboost as xgb
from sklearn.metrics import f1_score
from inference import metrics, probs_to_binary

logger = logging.getLogger(__name__)


class BoosterMixinManager:

    def init_hyperparams(self):

        _configs = get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]
        x_0, eta, r_0 = _configs.get('x_0'), _configs.get('eta'), _configs.get('r_0')
        model_hyper_params = [hyper_sampler(self.args['fl_parser'], None) for _ in range(x_0)]

        return model_hyper_params, x_0, eta, r_0

    #hyperparams = hyperparameters[bank_id]['hyperparameters']
    def init_models(self, hyperparams, bank_id=None):
        """Create Booster model(s) for parties.

        Args:
            hyperparams: dict with 'num_rounds' and 'params' from hyper_sampler.
            bank_id: if provided, only init model for that party.
                     Otherwise init for all parties.
        """
        # Divide CPU threads across parties when running in parallel
        num_parties = len(self.parties)
        total_cpus = os.cpu_count() or 1
        nthread = max(1, total_cpus // num_parties)

        if bank_id is not None:
            self.parties[bank_id].model = Booster(hyperparams, nthread=nthread)
        else:
            include_test = self.mode == 'training'
            for bid, party in self.iter_parties(include_test=include_test):
                party.model = Booster(hyperparams, nthread=nthread)

    def tune(self, party, laundering_values):

        frac_not_reached = True
        model_hyper_params, x_0, eta, r_0 = self.init_hyperparams()
        
        while frac_not_reached:

            f1s = []
            r_0 = min(r_0, 1)
            index_slice = round(party.procs_data['train_data']['x'].shape[0] * r_0)
            sliced_data = {'x': party.procs_data['train_data']['x'].iloc[:index_slice,:], 'y': party.procs_data['train_data']['y'][:index_slice]}

            for hp in model_hyper_params:
                try:
                    booster = Booster(hp)
                    booster.fit(sliced_data['x'], sliced_data['y'],
                                X_eval=party.procs_data['vali_data']['x'],
                                y_eval=laundering_values['true_y'])
                    preds = booster.predict(party.procs_data['vali_data']['x'])
                    f1s.append(f1_score(laundering_values['true_y'], probs_to_binary(preds)))
                except Exception as e:
                    print(f"Error with hyperparameters {hp}: {str(e)}")
                    f1s.append(0)

            x_0 = max(1, round(x_0/eta))
            params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:x_0]
            model_hyper_params = [model_hyper_params[i] for i in params_to_keep]

            if r_0 >= 1 or x_0 == 1:
                frac_not_reached = False
            r_0 *= eta
        
        return model_hyper_params[0]
    
    def tuning(self, laundering_values):

        results = {}
        self.set_mode('tuning')

        if self.args['data_parser'].ibm_hp:
            tuned_hyparameters, f1_score_for_hp = None, 1
            for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
                results[bank_id] = {'hyperparameters': tuned_hyparameters}
        else:
            bank_str = f'{len(self.parties)} banks' if self.args['fl_parser'].fl_algo != 'full_info' else 'full info'

            for idx, (bank_id, party) in enumerate(self.parties.items(), 1):
                party.prep_data()
                tuned_hyparameters = self._tuning_helper(laundering_values, party, bank_id)
                results[bank_id] = {'hyperparameters': tuned_hyparameters}

        return results
    

    def _prep_parties_data(self):
        for _, party in self.parties.items():
            party.prep_data()

    def train(self, hyperparameters, laundering_values_vali, laundering_values_test):

        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds
        self._prep_parties_data()

        logger.info("="*80)
        logger.info("Starting Booster training with %d seeds", seeds)
        logger.info("="*80)

        results_by_seed = {}
        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("\n" + "-"*80)
            logger.info("Training with seed %d/%d", seed_value, seeds)
            logger.info("-"*80)
            utils.set_seed(seed_value)

            result = self._train(
                hyperparameters,
                copy.deepcopy(laundering_values_vali),
                copy.deepcopy(laundering_values_test)
            )
            results_by_seed[seed_value] = result

            logger.info("Seed %d complete - F1: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f",
                        seed_value,
                        result['metrics']['f1'],
                        result['metrics']['roc_auc'],
                        result['metrics']['pr_auc'])

        logger.info("="*80)
        logger.info("All seeds completed")
        logger.info("="*80)

        return results_by_seed
    


#booster_data_train = xgb.DMatrix(sliced_data['x'], sliced_data['y'])
#booster_data_vali = xgb.DMatrix(party.procs_data['vali_data']['x'], laundering_values['true_y'])

#model = xgb.train(hp['params'], booster_data_train, num_boost_round = hp['num_rounds'], 
#                  evals = [(booster_data_vali, 'eval')], 
#                  custom_metric = f1_eval,
#                  maximize=True,
#                  early_stopping_rounds = 50, verbose_eval=False)
#preds = model.predict(xgb.DMatrix(party.procs_data['vali_data']['x']), iteration_range=(0, model.best_iteration + 1))


