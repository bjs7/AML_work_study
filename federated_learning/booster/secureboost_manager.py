"""SecureBoost Manager — vertical federated gradient boosting.

Each tree is built collaboratively:
  - Manager holds labels and computes (g, h) gradients per transaction.
  - Parties hold all features locally, search their own columns for best split
    candidates, and answer routing queries at inference.
  - Only gradient summary statistics (GL, GR, HL, HR) cross party boundaries —
    never raw feature values. These would be encrypted in production.

Design notes:
  - Encryption is a future one-place change: wrap find_best_split returns
    in encrypt() before sending and decrypt() before comparing at the manager.
  - Parties are queried sequentially within each node for simplicity.
    Parallelism can be added later via parallel_party_execute.
  - The feature naming convention (graph_feature_1..N) is the same across
    parties. After the winning party routes its transactions, other parties
    apply the same feature+threshold to the transactions they hold, reducing
    default-direction routing. Only truly unresolved transactions (held by
    no party for that feature) fall back to the default direction.
"""

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import utils
from inference import metrics
from .manager_mixin import BoosterMixinManager
from .vertical.setup import set_manager_data, setup_secureboost_post_prep
from data.relevant_banks import load_relevant_banks
from models.secureboost_tree import SecureBoostEnsemble, SBSplitNode, SBLeafNode, _batch_route
from result_io.save_results import build_save_dir, save_seed_result
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def _apply_tree(F: np.ndarray, indices: list, tree_root, all_parties: dict,
                learning_rate: float, txn_party_map: dict = None) -> None:
    """Batch-route all indices through tree_root and add the leaf values to F in-place.

    Uses _batch_route for cache-friendly traversal rather than per-sample recursion.

    Args:
        F: running raw-score array (one entry per index, in order).
        indices: global transaction indices aligned with F.
        tree_root: root of the newly built tree.
        all_parties: {bank_id: party} dict.
        learning_rate: shrinkage to apply to leaf values.
        txn_party_map: {global_idx: [party_ids]} for fast unresolved routing.
    """
    leaf_dict = _batch_route(tree_root, indices, all_parties, txn_party_map)
    for i, idx in enumerate(indices):
        F[i] += learning_rate * leaf_dict[idx]


class SecureBoostManager(BoosterMixinManager):
    """Vertical federated SecureBoost manager.

    Builds one shared gradient-boosted ensemble where the manager drives
    tree construction using gradient histograms from parties.
    """

    def __init__(self, args):
        super().__init__(args)
        self.data = {}
        self.txn_party_map = {}
        self.ensemble = None

        # Tree hyperparameters — set in init_models
        self.num_rounds   = 100
        self.max_depth    = 6
        self.learning_rate = 0.1
        self.lambda_reg   = 1.0
        self.min_gain     = 0.0
        self.min_child_weight = 1
        self.n_bins       = 32

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_parties(self, df, parsers, scaler_encoders, laundering_values):
        # Store manager data before any per-party index reset
        set_manager_data(self, df['regular_data'])

        if parsers['data_parser'].eval_mode == 'comparable':
            train_banks = load_relevant_banks(parsers['data_parser']).get('individual').get('banks')
            vali_banks, test_banks = train_banks, train_banks
        else:
            fedgraph_banks = load_relevant_banks(parsers['data_parser']).get('FedGraph')
            train_banks = fedgraph_banks['train_banks']
            vali_banks  = fedgraph_banks['vali_banks']
            test_banks  = fedgraph_banks['test_banks']

        if parsers['data_parser'].testing:
            train_banks = train_banks[:5]
            vali_banks  = vali_banks[:5]
            test_banks  = test_banks[:5]

        for banks, bank_type in zip([train_banks, vali_banks, test_banks], ['train', 'vali', 'test']):
            if banks:
                utils.add_banks_to_manager(parsers, banks, self, df, scaler_encoders, bank_type=bank_type)

        return self.tuning(laundering_values)

    def tuning(self, laundering_values):
        """Return hyperparameters for SecureBoost tree building.

        SecureBoost uses the same HP space as xgboost, so it reuses HPs
        tuned by the full_info xgboost scenario. Pass --tune on the full_info
        xgboost run to generate (or refresh) the saved HPs.
        """
        loaded_hp = self._load_tuned_hp()
        if loaded_hp is not None:
            logger.info("Loaded tuned hyperparameters for SecureBoost")
            return loaded_hp

        if not self.args['fl_parser'].tune:
            raise FileNotFoundError(
                "No saved hyperparameters found for SecureBoost. "
                "Run --fl_algo full_info --model xgboost --tune first to generate them."
            )

        logger.warning("No full_info tuned HPs found — using SecureBoost defaults.")
        return {
            "num_rounds": 100,
            "params": {
                "objective": "binary:logistic",
                "max_depth": 6,
                "learning_rate": 0.1,
                "lambda": 1.0,
                "scale_pos_weight": 5.0,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
                "tree_method": "hist",
                "random_state": 1,
            }
        }

    def init_models(self, hyperparams, bank_id=None):
        """Initialise the SecureBoost ensemble from a hyperparameter dict."""
        self.num_rounds       = hyperparams.get('num_rounds', 100)
        params                = hyperparams.get('params', {})
        self.max_depth        = int(params.get('max_depth', 6))
        self.learning_rate    = float(params.get('learning_rate', 0.1))
        self.lambda_reg       = float(params.get('lambda', 1.0))
        self.scale_pos_weight = float(params.get('scale_pos_weight', 1.0))
        self.subsample        = float(params.get('subsample', 1.0))

        # base_score is overwritten in _run_training once y_train is available
        self.ensemble = SecureBoostEnsemble(
            learning_rate=self.learning_rate,
            base_score=0.0,
        )

    def _prep_parties_data(self):
        for _, party in self.iter_parties(include_test=True):
            party.prep_data()

    # ------------------------------------------------------------------
    # Training entry points
    # ------------------------------------------------------------------

    def train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        """Override base train to add vertical setup after prep_data."""
        self.set_mode('training')
        seeds = self.args['data_parser'].testing_seeds
        results_by_seed = {}

        self._prep_parties_data()
        setup_secureboost_post_prep(self)

        logger.info("=" * 80)
        logger.info("Starting SecureBoost training with %d seeds", seeds)
        logger.info("=" * 80)

        self.save_dir = build_save_dir(self, hyperparameters)
        first_seed = getattr(self.args['data_parser'], 'first_seed', 1)

        for seed in range(seeds):
            seed_value = seed + first_seed
            logger.info("-" * 80)
            logger.info("Seed %d/%d", seed_value, seeds)
            logger.info("-" * 80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(
                hyperparameters,
                copy.deepcopy(laundering_values_vali),
                copy.deepcopy(laundering_values_test),
            )
            save_seed_result(self.save_dir, seed_value, results_by_seed[seed_value], self)
            m = results_by_seed[seed_value]['metrics']
            logger.info("Seed %d — F1: %.4f  ROC-AUC: %.4f  PR-AUC: %.4f",
                        seed_value, m['f1'], m['roc_auc'], m['pr_auc'])

        logger.info("=" * 80)
        logger.info("All seeds completed")
        logger.info("=" * 80)
        return results_by_seed

    def _train(self, hyperparameters, laundering_values_vali, laundering_values_test):
        self.init_models(hyperparameters)
        return self._run_training(laundering_values_vali, laundering_values_test)

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _run_training(self, laundering_values_vali, laundering_values_test):
        train_indices = list(self.data['train_data'].index)
        vali_indices  = list(self.data['vali_data'].index)
        test_indices  = list(self.data['test_data'].index)

        y_train = self.data['train_data']['Is Laundering'].values
        y_vali  = self.data['vali_data']['Is Laundering'].values
        y_test  = self.data['test_data']['Is Laundering'].values

        # --- Data-estimated base score (mirrors XGBoost 2.x behaviour) ---
        label_rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
        base_score = np.log(label_rate / (1.0 - label_rate))
        self.ensemble.base_score = base_score
        logger.info("base_score set to %.4f (label rate %.4f)", base_score, label_rate)

        # --- Position mapping: global_idx → array position (built once) ---
        idx_to_pos = {idx: i for i, idx in enumerate(train_indices)}

        # --- Incremental raw scores (avoids re-predicting all trees each round) ---
        F_train = np.full(len(train_indices), base_score)
        F_vali  = np.full(len(vali_indices),  base_score)
        F_test  = np.full(len(test_indices),  base_score)

        sb_override = self.args['data_parser'].sb_num_rounds
        if sb_override is not None:
            epochs = sb_override
        elif self.args['data_parser'].testing:
            epochs = 2
        else:
            epochs = self.num_rounds

        best_f1             = -1.0
        best_ensemble_state = None
        best_F_test         = None

        all_parties = {bank_id: party for bank_id, party in self.iter_parties(include_test=True)}

        # Ensure membership sets are built (setup.py does this, guard here too)
        for party in all_parties.values():
            if not hasattr(party, '_global_idx_set'):
                party._global_idx_set = frozenset(party._global_to_local.keys())

        # Precompute {global_idx: [party_ids]} — avoids O(N_parties) scan per node
        txn_party_list_map = {
            idx: [pid for pid in (info['sender_party'], info['receiver_party'])
                  if pid in all_parties]
            for idx, info in self.txn_party_map.items()
        }

        n_train_parties = sum(1 for _ in self.iter_parties(include_test=False))
        max_workers = min(utils.get_cpu_count(), n_train_parties)
        logger.info("SecureBoost: %d rounds, %d parties, %d parallel workers",
                    epochs, n_train_parties, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for round_num in range(epochs):

                # --- Row subsampling (mirrors XGBoost 'subsample' HP) ---
                if self.subsample < 1.0:
                    n_sample = max(1, int(len(train_indices) * self.subsample))
                    sample_pos = np.sort(np.random.choice(len(train_indices), n_sample, replace=False))
                    round_indices = [train_indices[i] for i in sample_pos]
                    round_F       = F_train[sample_pos]
                    round_y       = y_train[sample_pos]
                    round_idx_to_pos = {idx: i for i, idx in enumerate(round_indices)}
                else:
                    round_indices    = train_indices
                    round_F          = F_train
                    round_y          = y_train
                    round_idx_to_pos = idx_to_pos

                # --- Vectorised gradient computation ---
                p_arr = 1.0 / (1.0 + np.exp(-np.clip(round_F, -500, 500)))
                g_arr = p_arr - round_y
                h_arr = p_arr * (1.0 - p_arr)
                if self.scale_pos_weight != 1.0:
                    pos_mask = (round_y == 1)
                    g_arr = np.where(pos_mask, g_arr * self.scale_pos_weight, g_arr)
                    h_arr = np.where(pos_mask, h_arr * self.scale_pos_weight, h_arr)

                # --- Build one tree (parallel split-finding inside) ---
                tree_root = self._build_tree(
                    round_indices, g_arr, h_arr, round_idx_to_pos, depth=0,
                    executor=executor, txn_party_map=txn_party_list_map
                )
                self.ensemble.add_tree(tree_root)

                # --- Incremental score updates (one tree pass each, not full ensemble) ---
                _apply_tree(F_train, train_indices, tree_root, all_parties, self.learning_rate, txn_party_list_map)
                _apply_tree(F_vali,  vali_indices,  tree_root, all_parties, self.learning_rate, txn_party_list_map)
                _apply_tree(F_test,  test_indices,  tree_root, all_parties, self.learning_rate, txn_party_list_map)

                # --- Validate ---
                vali_probs  = 1.0 / (1.0 + np.exp(-np.clip(F_vali, -500, 500)))
                vali_binary = (vali_probs > 0.5).astype(int)
                f1_vali     = f1_score(y_vali, vali_binary, zero_division=0)

                logger.info("Round %d/%d  Vali F1: %.4f", round_num + 1, epochs, f1_vali)

                if f1_vali > best_f1:
                    best_f1             = f1_vali
                    best_ensemble_state = copy.deepcopy(self.ensemble)
                    best_F_test         = F_test.copy()

        # --- Final test evaluation using saved scores at best round ---
        assert best_ensemble_state is not None, "No best ensemble found"
        self.ensemble = best_ensemble_state

        test_probs  = 1.0 / (1.0 + np.exp(-np.clip(best_F_test, -500, 500)))
        test_binary = (test_probs > 0.5).astype(int)

        final_metrics = metrics(y_true=y_test, y_pred_probabilities=test_probs)
        logger.info("Test F1: %.4f", final_metrics['f1'])

        laundering_values_test['pred_probabilities'] = test_probs
        laundering_values_test['pred_label']         = test_binary
        laundering_values_test['true_y']             = y_test

        return {
            'metrics':            final_metrics,
            'laundering_values':  laundering_values_test,
            'ensemble':           self.ensemble,
        }

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_tree(self, node_indices: list, g_arr: np.ndarray, h_arr: np.ndarray,
                    idx_to_pos: dict, depth: int, executor=None, txn_party_map: dict = None):
        """Recursively build one CART tree node using gradient histograms from parties.

        Args:
            node_indices: global transaction indices routed to this node.
            g_arr: first-order gradient array indexed by idx_to_pos.
            h_arr: second-order gradient (Hessian) array indexed by idx_to_pos.
            idx_to_pos: {global_idx: position_in_g_arr} for all training transactions.
            depth: current depth (0 = root).
            executor: ThreadPoolExecutor for parallel split-finding (None = sequential).

        Returns:
            SBLeafNode or SBSplitNode.
        """
        pos        = np.array([idx_to_pos[i] for i in node_indices])
        G_node     = g_arr[pos].sum()
        H_node     = h_arr[pos].sum()
        leaf_value = -G_node / (H_node + self.lambda_reg)

        # Stopping conditions
        if depth >= self.max_depth or len(node_indices) <= self.min_child_weight:
            return SBLeafNode(value=leaf_value)

        node_set   = set(node_indices)
        best_gain      = self.min_gain
        best_party_id  = None
        best_feature   = None
        best_threshold = None

        # --- Query every party for their best split candidate ---
        # Parallel when executor is provided; sequential fallback otherwise.
        if executor is not None:
            futures = {}
            for bank_id, party in self.iter_parties(include_test=False):
                party_indices = list(node_set & party._global_idx_set)
                if not party_indices:
                    continue
                fut = executor.submit(
                    party.find_best_split,
                    party_indices, g_arr, h_arr, idx_to_pos, self.lambda_reg, self.n_bins
                )
                futures[fut] = bank_id

            for fut in as_completed(futures):
                bank_id = futures[fut]
                try:
                    gain, feature, threshold = fut.result()
                except Exception as exc:
                    logger.warning("Party %s split-find failed: %s", bank_id, exc)
                    continue
                if gain > best_gain:
                    best_gain      = gain
                    best_party_id  = bank_id
                    best_feature   = feature
                    best_threshold = threshold
        else:
            for bank_id, party in self.iter_parties(include_test=False):
                party_indices = list(node_set & party._global_idx_set)
                if not party_indices:
                    continue
                gain, feature, threshold = party.find_best_split(
                    party_indices, g_arr, h_arr, idx_to_pos, self.lambda_reg, self.n_bins
                )
                if gain > best_gain:
                    best_gain      = gain
                    best_party_id  = bank_id
                    best_feature   = feature
                    best_threshold = threshold

        if best_party_id is None:
            return SBLeafNode(value=leaf_value)

        # --- Ask winning party to route its transactions ---
        winning_party   = self.parties[best_party_id]
        party_indices   = list(node_set & winning_party._global_idx_set)
        left_p, right_p = winning_party.route_samples(party_indices, best_feature, best_threshold)
        left_set        = set(left_p)
        right_set       = set(right_p)

        # --- Ask other parties to route unresolved transactions with same feature+threshold ---
        # Each transaction is held by at most 2 parties (sender + receiver).
        # txn_party_map gives a direct lookup, avoiding an O(N_parties) scan.
        unresolved_set = node_set - left_set - right_set
        if unresolved_set and txn_party_map is not None:
            for idx in list(unresolved_set):
                for pid in txn_party_map.get(idx, []):
                    if pid == best_party_id:
                        continue
                    party = self.parties.get(pid)
                    if party is None:
                        continue
                    val = party.get_feature_value(idx, best_feature)
                    if val is not None:
                        (left_set if val <= best_threshold else right_set).add(idx)
                        unresolved_set.discard(idx)
                        break

        # Truly unresolved → default direction based on gradient signal.
        lp_pos = np.array([idx_to_pos[i] for i in left_set])  if left_set  else np.array([], dtype=int)
        rp_pos = np.array([idx_to_pos[i] for i in right_set]) if right_set else np.array([], dtype=int)
        G_left       = g_arr[lp_pos].sum() if len(lp_pos) else 0.0
        G_right      = g_arr[rp_pos].sum() if len(rp_pos) else 0.0
        default_left = abs(G_left) >= abs(G_right)

        if unresolved_set:
            if default_left:
                left_set.update(unresolved_set)
            else:
                right_set.update(unresolved_set)

        left_indices  = list(left_set)
        right_indices = list(right_set)

        # Guard: don't create empty children
        if not left_indices or not right_indices:
            return SBLeafNode(value=leaf_value)

        left_child  = self._build_tree(left_indices,  g_arr, h_arr, idx_to_pos, depth + 1, executor, txn_party_map)
        right_child = self._build_tree(right_indices, g_arr, h_arr, idx_to_pos, depth + 1, executor, txn_party_map)

        return SBSplitNode(
            party_id=best_party_id,
            feature=best_feature,
            threshold=best_threshold,
            default_left=default_left,
            left=left_child,
            right=right_child,
        )
