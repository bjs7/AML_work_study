"""SecureBoost tree data structures and ensemble prediction.

The tree is built by the manager from gradient histograms supplied by parties.
Each internal node records which party answered the split query, so at inference
time the manager knows exactly which party to ask for the routing decision.
No raw feature values ever leave a party.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


# ---------------------------------------------------------------------------
# Tree node types
# ---------------------------------------------------------------------------

@dataclass
class SBLeafNode:
    value: float  # raw score contribution (before learning rate scaling)


@dataclass
class SBSplitNode:
    party_id: int    # which party owns this split
    feature: str     # feature name in that party's local procs_data
    threshold: float
    default_left: bool  # routing direction when the party has no data for a sample
    left: SBNode
    right: SBNode


SBNode = Union[SBSplitNode, SBLeafNode]


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class SecureBoostEnsemble:
    """Gradient-boosted ensemble of SecureBoost trees.

    Held entirely by the manager. Prediction requires routing queries to
    parties (via _route), so parties must be available at inference time.
    """

    def __init__(self, learning_rate: float = 0.1, base_score: float = 0.0):
        """
        Args:
            learning_rate: shrinkage applied to each tree's leaf values.
            base_score: initial raw score (log-odds of the prior class probability).
        """
        self.trees: list[SBNode] = []
        self.learning_rate = learning_rate
        self.base_score = base_score

    def add_tree(self, root: SBNode):
        self.trees.append(root)

    def predict_raw(self, global_indices: list, parties: dict) -> dict:
        """Return raw scores (pre-sigmoid) for each global transaction index.

        Args:
            global_indices: list of global transaction indices to predict.
            parties: dict of {bank_id: party} — all parties available for routing.

        Returns:
            {global_idx: raw_score}
        """
        scores = {idx: self.base_score for idx in global_indices}
        for tree_root in self.trees:
            for idx in global_indices:
                scores[idx] += self.learning_rate * _route(idx, tree_root, parties)
        return scores

    def predict_proba(self, global_indices: list, parties: dict) -> dict:
        """Return predicted probabilities (sigmoid of raw scores).

        Returns:
            {global_idx: probability}
        """
        raw = self.predict_raw(global_indices, parties)
        return {idx: _sigmoid(score) for idx, score in raw.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _route(global_idx: int, node: SBNode, parties: dict) -> float:
    """Recursively route one sample through a tree and return its leaf value.

    At each internal node, asks the owning party for the feature value.
    Falls back to the node's default direction if the party has no data
    for this transaction (e.g. intra-bank transaction at a different bank).
    """
    if isinstance(node, SBLeafNode):
        return node.value

    party = parties.get(node.party_id)
    feature_val = None
    if party is not None:
        feature_val = party.get_feature_value(global_idx, node.feature)

    if feature_val is None:
        child = node.left if node.default_left else node.right
    else:
        child = node.left if feature_val <= node.threshold else node.right

    return _route(global_idx, child, parties)
