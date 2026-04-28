import copy
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import federated_learning.hp_tuning as tr_utils
import utils
from models.booster import Booster
from federated_learning.hp_tuning import hyper_sampler, f1_eval
from configs.paths import get_tuning_configs, get_full_info_hp_path
from results.save_results import build_save_dir, save_seed_result
import xgboost as xgb
from sklearn.metrics import f1_score
from inference import metrics, probs_to_binary

logger = logging.getLogger(__name__)

# Simple fallback hyperparameters for fast local test runs (--testing mode).
# Used by _load_tuned_hp when no saved HP file is found.
_FALLBACK_HP = {
    'xgboost': {
        'num_rounds': 10,
        'params': {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.1,
            'lambda': 1.0,
            'scale_pos_weight': 5.0,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'tree_method': 'hist',
            'random_state': 1,
        }
    },
    'light_gbm': {
        'num_rounds': 10,
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'lambda_l2': 0.1,
            'scale_pos_weight': 5.0,
            'lambda_l1': 1.0,
            'random_state': 1,
            'verbose': -1,
        }
    },
}


def _eval_hp(hp, sliced_x, sliced_y, vali_x, vali_y, nthread):
    """Fit one HP config and return its validation F1. Module-level so it
    can be submitted to a ThreadPoolExecutor without pickling issues."""
    try:
        booster = Booster(hp, nthread=nthread)
        booster.fit(sliced_x, sliced_y, X_eval=vali_x, y_eval=vali_y)
        preds = booster.predict(vali_x)
        return f1_score(vali_y, probs_to_binary(preds))
    except Exception as e:
        logger.warning("HP eval failed: %s", e)
        return 0.0


class BoosterMixinManager:

    def _save_tuned_hp(self, hp, model=None):
        """Save tuned hyperparameters to a shared location for reuse by other scenarios."""
        path = get_full_info_hp_path(self.args, model=model)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(hp, f, indent=4)
        logger.info("Saved full_info tuned hyperparameters to %s", path)

    def _save_top_n_hp(self, ranked_list, model=None):
        """Save top N HP configs with validation F1 scores for manual inspection.

        Writes to <size>_<ir>_top<n>.json alongside the normal best-HP file.
        Each entry has rank, vali_f1, and the full hyperparameter dict so the
        user can compare num_rounds vs F1 and pick a feasible config.

        Args:
            ranked_list: list of (hp_dict, f1) tuples, sorted best first.
            model: model name override (passed through to get_full_info_hp_path).
        """
        if not ranked_list:
            return
        n = len(ranked_list)
        path = get_full_info_hp_path(self.args, model=model)
        stem, ext = os.path.splitext(path)
        top_n_path = f"{stem}_top{n}{ext}"
        Path(top_n_path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {'rank': i + 1, 'vali_f1': round(float(f1), 6), 'hyperparameters': hp}
            for i, (hp, f1) in enumerate(ranked_list)
        ]
        with open(top_n_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info("Saved top %d HP configs to %s", n, top_n_path)

    def _load_tuned_hp(self, model=None):
        """Load tuned hyperparameters saved by the full_info scenario.

        Returns None if no saved HPs are found, in which case the caller
        should fall back to running its own tuning.

        When --testing is set and no file is found, returns simple fallback
        hyperparameters so local test runs skip tuning entirely.

        If --hp_path is set, it overrides the auto-detected path entirely.
        The path is interpreted relative to the project root on local, or
        relative to $VSC_DATA/AML_work_study/AML_work_study on HPC.
        """
        hp_path_override = getattr(self.args['data_parser'], 'hp_path', None)
        if hp_path_override:
            from configs.paths import get_data_path
            base = get_data_path()
            if base == '/data/leuven/362/vsc36278':
                path = os.path.join(base, 'AML_work_study', 'AML_work_study', hp_path_override)
            else:
                path = hp_path_override
        else:
            path = get_full_info_hp_path(self.args, model=model)
        if os.path.exists(path):
            with open(path, 'r') as f:
                hp = json.load(f)
            logger.info("Loaded full_info tuned hyperparameters from %s", path)
            return hp
        if self.args['data_parser'].testing:
            fallback = _FALLBACK_HP.get(self.args['fl_parser'].model)
            if fallback is not None:
                logger.info("No saved HPs found; using fallback hyperparameters (--testing mode)")
                return fallback
        return None

    def init_hyperparams(self):

        _configs = get_tuning_configs(self.args)[self.args['data_parser'].scenario][self.args['data_parser'].size]
        x_0, eta, r_0 = _configs.get('x_0'), _configs.get('eta'), _configs.get('r_0')
        max_rounds = self.args['data_parser'].tune_max_rounds
        model_hyper_params = [hyper_sampler(self.args['fl_parser'], None, max_rounds=max_rounds) for _ in range(x_0)]

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
        total_cpus = utils.get_cpu_count()
        nthread = max(1, total_cpus // num_parties)

        if bank_id is not None:
            self.parties[bank_id].model = Booster(hyperparams, nthread=nthread)
        else:
            include_test = self.mode == 'training'
            for bid, party in self.iter_parties(include_test=include_test):
                party.model = Booster(hyperparams, nthread=nthread)

    def tune(self, party, laundering_values, top_n=1):
        """Run SHA tuning and return the best HP config (or top N configs with scores).

        Args:
            party: the party whose data is used for tuning.
            laundering_values: dict with 'true_y' for the validation split.
            top_n: if 1 (default), returns the single best HP dict (backward
                   compatible). If > 1, returns a list of (hp_dict, vali_f1)
                   tuples from the final SHA round, sorted best first, capped
                   at min(top_n, survivors).

        Returns:
            best HP dict (top_n==1) or list of (hp, f1) tuples (top_n>1).
        """
        frac_not_reached = True
        model_hyper_params, x_0, eta, r_0 = self.init_hyperparams()

        total_cpus = utils.get_cpu_count()
        vali_x = party.procs_data['vali_data']['x']
        vali_y = laundering_values['true_y']

        final_ranked = []  # populated in the last SHA iteration

        while frac_not_reached:

            r_0 = min(r_0, 1)
            index_slice = round(party.procs_data['train_data']['x'].shape[0] * r_0)
            sliced_x = party.procs_data['train_data']['x'].iloc[:index_slice]
            sliced_y = party.procs_data['train_data']['y'][:index_slice]

            n_configs = len(model_hyper_params)
            max_workers = min(n_configs, total_cpus)
            nthread     = max(1, total_cpus // max_workers)
            logger.info("SHA round: %d configs, %d parallel workers, %d threads/fit",
                        n_configs, max_workers, nthread)

            f1s = [None] * n_configs
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_eval_hp, hp, sliced_x, sliced_y, vali_x, vali_y, nthread): i
                    for i, hp in enumerate(model_hyper_params)
                }
                for future in as_completed(future_to_idx):
                    f1s[future_to_idx[future]] = future.result()

            x_0 = max(1, round(x_0/eta))
            params_to_keep = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)[:x_0]
            model_hyper_params = [model_hyper_params[i] for i in params_to_keep]

            if r_0 >= 1 or x_0 == 1:
                frac_not_reached = False
                # params_to_keep is sorted desc by f1; capture surviving configs + scores
                f1s_kept = [f1s[i] for i in params_to_keep]
                final_ranked = list(zip(model_hyper_params, f1s_kept))
            r_0 *= eta

        if top_n == 1:
            return model_hyper_params[0]
        return final_ranked[:top_n]
    
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
                import time as _time
                t0 = _time.time()
                party.prep_data()
                logger.info("prep_data done in %.1fs", _time.time() - t0)
                t1 = _time.time()
                tuned_hyparameters = self._tuning_helper(laundering_values, party, bank_id)
                logger.info("HP tuning done in %.1fs", _time.time() - t1)
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
        self.save_dir = build_save_dir(self, hyperparameters)
        first_seed = getattr(self.args['data_parser'], 'first_seed', 1)

        for seed in range(seeds):
            seed_value = seed + first_seed
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

            save_seed_result(self.save_dir, seed_value, result, self)

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


