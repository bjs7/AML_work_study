# %%
# Step-by-step walk-through of build_edge_lookup and _party_edge_positions
# (federated_learning/gnn/splitfed/batching.py), with the function bodies written out.
# Run cell by cell. Only needs torch.
import torch

# %%
# Toy party graphs: for each bank, the global transaction IDs of its edges, in the
# order they sit in its graph (= edge_attr[:, 0]). Transactions:
#   100: bank 10 → 20     101: 10 → 10 (intra)   102: 20 → 30   103: 30 → 10
#   104: 20 → 20 (intra)  105: 10 → 30           106: 30 → 20
party_gids = {
    10: torch.tensor([100, 101, 103, 105]),
    20: torch.tensor([100, 102, 104, 106]),
    30: torch.tensor([102, 103, 105, 106]),
}

# %%
# ===== build_edge_lookup =====
# One tensor per party: its transaction IDs, its bank ID repeated, and positions 0..n-1.
gids, banks, positions = [], [], []
for bank_id, g in party_gids.items():
    gids.append(g)
    banks.append(torch.full_like(g, bank_id))
    positions.append(torch.arange(len(g), dtype=torch.long))

print('gids     ', gids)
print('banks    ', banks)
print('positions', positions)

# %%
# Concatenate: entry k = (transaction gids[k], bank banks[k], position positions[k]).
# Cross-bank transactions appear twice, intra-bank once.
gids = torch.cat(gids)
banks = torch.cat(banks)
positions = torch.cat(positions)
print('gids     ', gids.tolist())
print('banks    ', banks.tolist())
print('positions', positions.tolist())

# %%
# Sort all three by transaction ID, so each transaction's entries sit next to each other.
order = torch.argsort(gids, stable=True)
sorted_gids, entry_banks, entry_pos = gids[order], banks[order], positions[order]
print('sorted_gids', sorted_gids.tolist())
print('entry_banks', entry_banks.tolist())
print('entry_pos  ', entry_pos.tolist())

# %%
# ===== _party_edge_positions, for one batch =====
# The batch's transaction IDs (seeds + sampled neighbours, sorted), and the banks that
# build a subgraph (the banks of the seed transactions 102 and 104).
batch_global_ids = torch.tensor([100, 102, 104, 106])
seed_banks = [20, 30]

# %%
# For each batch transaction, the range [left, right) of its entries in sorted_gids.
left = torch.searchsorted(sorted_gids, batch_global_ids, right=False)
right = torch.searchsorted(sorted_gids, batch_global_ids, right=True)
counts = right - left
print('left  ', left.tolist())
print('right ', right.tolist())
print('counts', counts.tolist(), '(2 = in two party graphs, 1 = intra-bank)')

# %%
# Turn the ranges into a flat list of entry indices into the lookup.
total = int(counts.sum())
starts = torch.repeat_interleave(left, counts)
offsets = torch.arange(total) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
entries = starts + offsets
print('starts ', starts.tolist())
print('offsets', offsets.tolist())
print('entries', entries.tolist())

# %%
# Which bank and position each of those entries is.
sel_banks, sel_pos = entry_banks[entries], entry_pos[entries]
print('sel_banks', sel_banks.tolist())
print('sel_pos  ', sel_pos.tolist())

# %%
# Keep only the banks that build a subgraph in this batch.
keep = torch.isin(sel_banks, torch.tensor(sorted(seed_banks), dtype=torch.long))
sel_banks, sel_pos = sel_banks[keep], sel_pos[keep]
print('keep     ', keep.tolist())
print('sel_banks', sel_banks.tolist())
print('sel_pos  ', sel_pos.tolist())

# %%
# Group by bank, positions ascending within each bank (the party graph's own order).
key = sel_banks * (int(sel_pos.max()) + 1) + sel_pos
order = torch.argsort(key)
sel_banks, sel_pos = sel_banks[order], sel_pos[order]
print('key      ', key.tolist())
print('sel_banks', sel_banks.tolist())
print('sel_pos  ', sel_pos.tolist())

# %%
uniq_banks, bank_counts = torch.unique_consecutive(sel_banks, return_counts=True)
party_positions = dict(zip(uniq_banks.tolist(), torch.split(sel_pos, bank_counts.tolist())))
print('uniq_banks ', uniq_banks.tolist(), ' bank_counts', bank_counts.tolist())
for b, p in party_positions.items():
    print(f'bank {b}: positions {p.tolist()}')

# %%
# Check against the original approach: torch.isin over each bank's full graph.
for b in seed_banks:
    original = torch.isin(party_gids[b], batch_global_ids).nonzero(as_tuple=True)[0]
    print(f'bank {b}: lookup {party_positions[b].tolist()}   isin {original.tolist()}   '
          f'{"OK" if torch.equal(party_positions[b], original) else "MISMATCH"}')
