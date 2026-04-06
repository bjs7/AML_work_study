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
    parties. The tree node's party_id tells the manager exactly which party
    to ask at inference — no renaming needed.
"""

import copy
import logging
import numpy as np

import utils
from inference import metrics
from .manager_mixin import BoosterMixinManager
from .vertical.setup import set_manager_data, setup_secureboost_post_prep
from data.relevant_banks import load_relevant_banks
from models.secureboost_tree import SecureBoostEnsemble, SBSplitNode, SBLeafNode, _sigmoid
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


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
        """Return a single default hyperparameter configuration.

        Full tuning is omitted for now — building the full vertical ensemble
        per HP config would be prohibitively expensive. A proper tuning loop
        can be added later once the core training is validated.
        """
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
        self.num_rounds    = hyperparams.get('num_rounds', 100)
        params             = hyperparams.get('params', {})
        self.max_depth     = int(params.get('max_depth', 6))
        self.learning_rate = float(params.get('learning_rate', 0.1))
        self.lambda_reg    = float(params.get('lambda', 1.0))

        # base score: log-odds of 0.5 (neutral prior)
        base_score = 0.5
        self.ensemble = SecureBoostEnsemble(
            learning_rate=self.learning_rate,
            base_score=float(np.log(base_score / (1.0 - base_score))),
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

        for seed in range(seeds):
            seed_value = seed + 1
            logger.info("-" * 80)
            logger.info("Seed %d/%d", seed_value, seeds)
            logger.info("-" * 80)
            utils.set_seed(seed_value)

            results_by_seed[seed_value] = self._train(
                hyperparameters,
                copy.deepcopy(laundering_values_vali),
                copy.deepcopy(laundering_values_test),
            )
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

        # Running raw scores for train (updated incrementally)
        F_train = {idx: self.ensemble.base_score for idx in train_indices}

        epochs = 2 if self.args['data_parser'].testing else self.num_rounds

        best_f1            = -1.0
        best_ensemble_state = None

        all_parties = {bank_id: party for bank_id, party in self.iter_parties(include_test=True)}

        for round_num in range(epochs):

            # --- Compute gradients from current predictions ---
            p_arr   = np.array([_sigmoid(F_train[idx]) for idx in train_indices])
            g_arr   = p_arr - y_train
            h_arr   = p_arr * (1.0 - p_arr)
            g_dict  = {idx: float(g_arr[i]) for i, idx in enumerate(train_indices)}
            h_dict  = {idx: float(h_arr[i]) for i, idx in enumerate(train_indices)}

            # --- Build one tree ---
            tree_root = self._build_tree(train_indices, g_dict, h_dict, depth=0)
            self.ensemble.add_tree(tree_root)

            # --- Update running scores ---
            for idx in train_indices:
                from models.secureboost_tree import _route
                F_train[idx] += self.learning_rate * _route(idx, tree_root, all_parties)

            # --- Validate ---
            vali_preds  = self.ensemble.predict_proba(vali_indices, all_parties)
            vali_probs  = np.array([vali_preds[idx] for idx in vali_indices])
            vali_binary = (vali_probs > 0.5).astype(int)
            f1_vali     = f1_score(y_vali, vali_binary, zero_division=0)

            logger.info("Round %d/%d  Vali F1: %.4f", round_num + 1, epochs, f1_vali)

            if f1_vali > best_f1:
                best_f1             = f1_vali
                best_ensemble_state = copy.deepcopy(self.ensemble)

        # --- Final test evaluation with best ensemble ---
        assert best_ensemble_state is not None, "No best ensemble found"
        self.ensemble = best_ensemble_state

        test_preds  = self.ensemble.predict_proba(test_indices, all_parties)
        test_probs  = np.array([test_preds[idx] for idx in test_indices])
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

    def _build_tree(self, node_indices: list, g_dict: dict, h_dict: dict, depth: int):
        """Recursively build one CART tree node using gradient histograms from parties.

        Args:
            node_indices: global transaction indices routed to this node.
            g_dict: {global_idx: g_i} for all training transactions.
            h_dict: {global_idx: h_i} for all training transactions.
            depth: current depth (0 = root).

        Returns:
            SBLeafNode or SBSplitNode.
        """
        G_node = sum(g_dict[i] for i in node_indices)
        H_node = sum(h_dict[i] for i in node_indices)
        leaf_value = -G_node / (H_node + self.lambda_reg)

        # Stopping conditions
        if depth >= self.max_depth or len(node_indices) <= self.min_child_weight:
            return SBLeafNode(value=leaf_value)

        # --- Query every party for their best split candidate ---
        best_gain      = self.min_gain
        best_party_id  = None
        best_feature   = None
        best_threshold = None

        for bank_id, party in self.iter_parties(include_test=False):
            # Transactions in this node that the party holds
            party_indices = [i for i in node_indices if i in party._global_to_local]
            if not party_indices:
                continue

            gain, feature, threshold = party.find_best_split(
                party_indices, g_dict, h_dict, self.lambda_reg, self.n_bins
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
        party_indices   = [i for i in node_indices if i in winning_party._global_to_local]
        left_p, right_p = winning_party.route_samples(party_indices, best_feature, best_threshold)
        left_set        = set(left_p)
        right_set       = set(right_p)

        # Transactions not held by the winning party get the default direction.
        # Default: whichever child has the larger total |G| (more gradient signal).
        unresolved = [i for i in node_indices if i not in left_set and i not in right_set]
        if unresolved:
            G_left  = sum(g_dict[i] for i in left_p)
            G_right = sum(g_dict[i] for i in right_p)
            default_left = abs(G_left) >= abs(G_right)
            if default_left:
                left_set.update(unresolved)
            else:
                right_set.update(unresolved)
        else:
            G_left       = sum(g_dict[i] for i in left_p)
            G_right      = sum(g_dict[i] for i in right_p)
            default_left = abs(G_left) >= abs(G_right)

        left_indices  = list(left_set)
        right_indices = list(right_set)

        # Guard: don't create empty children
        if not left_indices or not right_indices:
            return SBLeafNode(value=leaf_value)

        left_child  = self._build_tree(left_indices,  g_dict, h_dict, depth + 1)
        right_child = self._build_tree(right_indices, g_dict, h_dict, depth + 1)

        return SBSplitNode(
            party_id=best_party_id,
            feature=best_feature,
            threshold=best_threshold,
            default_left=default_left,
            left=left_child,
            right=right_child,
        )
