"""Simplified vertical FL — no per-layer embedding exchange.

Each party runs all GNN layers locally on its own graph, then provides
final edge embeddings to the manager. The manager concatenates the
from-bank and to-bank embeddings per transaction and classifies.
"""

from . import setup
from . import forward
from . import batching
