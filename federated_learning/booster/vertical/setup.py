"""Setup functions for SecureBoost vertical federated learning (booster).

Adapted from federated_learning/gnn/vertical/setup.py but for tabular data:
  - Three independent splits (not cumulative supersets like GNN vertical)
  - No graph structure — only row-level index mappings are needed
  - Manager holds only From Bank, To Bank, Is Laundering (no raw features)

Keep this file self-contained so it can evolve independently from the GNN
vertical framework until the design is stable.
"""

import logging

logger = logging.getLogger(__name__)


def set_manager_data(manager, regular_data: dict):
    """Store minimal manager data with original global indices preserved.

    Called early in setup_parties, before any per-party index reset, so that
    global indices are still intact. The manager only needs three columns:
    From Bank and To Bank (to build the transaction-party map) and
    Is Laundering (to compute gradients during training).

    Args:
        manager: SecureBoostManager instance.
        regular_data: df['regular_data'] — the raw per-split dict with original indices.
    """
    cols = ['From Bank', 'To Bank', 'Is Laundering']
    manager.data = {
        'train_data': regular_data['train_data']['x'][cols].copy(),
        'vali_data':  regular_data['vali_data']['x'][cols].copy(),
        'test_data':  regular_data['test_data']['x'][cols].copy(),
    }


def build_txn_party_map(manager):
    """Build a global-index-to-party-participation map for all splits.

    For each transaction (identified by its global DataFrame index), records:
      - sender_party: bank_id that owns the sending account (From Bank)
      - receiver_party: bank_id that owns the receiving account (To Bank)
      - is_cross_bank: whether the two parties differ
      - mode: which split this transaction belongs to

    Global indices are unique across splits because train/vali/test are
    non-overlapping row slices of the same original DataFrame (never re-indexed
    before this point). So a single dict keyed by global_idx is unambiguous.
    """
    manager.txn_party_map = {}
    for mode in ['train', 'vali', 'test']:
        df = manager.data[f'{mode}_data']
        for global_idx, row in df.iterrows():
            from_bank = int(row['From Bank'])
            to_bank   = int(row['To Bank'])
            manager.txn_party_map[global_idx] = {
                'sender_party':   from_bank,
                'receiver_party': to_bank,
                'is_cross_bank':  from_bank != to_bank,
                'mode':           mode,
            }


def build_party_index_maps(manager):
    """Build global_index → (mode, local_row) maps for every party.

    Each party's procs_data uses reset indices (rows 0..N-1 per split), but
    transactions are identified by global DataFrame indices. This map lets
    both the manager and the party translate between the two.

    Must be called after party.prep_data() so that procs_data is populated.

    Uses iter_parties(include_test=True) rather than test_parties alone,
    because for the booster framework the splits are independent — a party
    with only train transactions is in self.parties but not in test_parties.
    """
    for _, party in manager.iter_parties(include_test=True):
        party._global_to_local = {}
        for mode in ['train', 'vali', 'test']:
            idx_key = f'{mode}_indices'
            indices = party.indices.get(idx_key, [])
            for local_row, global_idx in enumerate(indices):
                # global indices are disjoint across splits so no collision
                party._global_to_local[global_idx] = (mode, local_row)

        # Fast membership check: set intersection instead of per-element lookup
        party._global_idx_set = frozenset(party._global_to_local.keys())

        # Pre-computed numpy arrays per feature — replaces slow x.iloc[row][col]
        party._feature_arrays = {}
        for mode in ['train', 'vali', 'test']:
            data_key = f'{mode}_data'
            if data_key in party.procs_data:
                x = party.procs_data[data_key]['x']
                party._feature_arrays[mode] = {col: x[col].values for col in x.columns}


def setup_secureboost_post_prep(manager):
    """Complete vertical setup after parties have prepped their data.

    Called from SecureBoostManager.train, after prep_data() has been called
    on all parties (so procs_data is available for the index map build).

    Args:
        manager: SecureBoostManager instance with all parties already added
                 and prep_data() already called.
    """
    build_txn_party_map(manager)
    build_party_index_maps(manager)

    n_parties = sum(1 for _ in manager.iter_parties(include_test=True))
    logger.info(
        "SecureBoost vertical setup complete: %d transactions mapped, %d parties",
        len(manager.txn_party_map), n_parties
    )
