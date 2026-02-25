import sys
import traceback

print("=" * 80)
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("=" * 80)

# ----------------------------
# Core imports and version info
# ----------------------------
print("\n[1] Core package imports")
try:
    import torch
    import torch_geometric
    import pyg_lib
    import torch_scatter
    import torch_sparse
    import torch_cluster
    import torch_spline_conv

    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch_geometric:", torch_geometric.__version__)
    print("pyg_lib: OK")
    print("torch_scatter: OK")
    print("torch_sparse: OK")
    print("torch_cluster: OK")
    print("torch_spline_conv: OK")
except Exception:
    print("FAILED importing PyTorch/PyG stack")
    traceback.print_exc()
    sys.exit(1)

# ----------------------------
# CPU sanity check
# ----------------------------
print("\n[2] CPU execution sanity check")
try:
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = x @ y
    print("Basic CPU matmul: OK, result shape =", tuple(z.shape))
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception:
    print("FAILED CPU tensor test")
    traceback.print_exc()
    sys.exit(2)

# ----------------------------
# PyG NeighborLoader test
# ----------------------------
print("\n[3] NeighborLoader test")
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

    num_nodes = 100
    num_edges = 400

    src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    x = torch.randn(num_nodes, 16)
    y = torch.randint(0, 3, (num_nodes,), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes

    input_nodes = torch.arange(0, min(32, num_nodes))

    nloader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=8,
        input_nodes=input_nodes,
        shuffle=False,
    )

    batch = next(iter(nloader))
    print("NeighborLoader batch: OK")
    print("  batch.x shape:", tuple(batch.x.shape))
    print("  batch.edge_index shape:", tuple(batch.edge_index.shape))
    print("  batch.batch_size:", int(batch.batch_size))
except Exception:
    print("FAILED NeighborLoader test")
    traceback.print_exc()
    sys.exit(3)

# ----------------------------
# PyG LinkNeighborLoader test
# ----------------------------
print("\n[4] LinkNeighborLoader test")
try:
    edge_label_index = edge_index[:, : min(64, edge_index.size(1))]
    edge_label = torch.ones(edge_label_index.size(1), dtype=torch.float)

    lloader = LinkNeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=8,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        shuffle=False,
    )

    lbatch = next(iter(lloader))
    print("LinkNeighborLoader batch: OK")
    print("  lbatch.x shape:", tuple(lbatch.x.shape))
    print("  lbatch.edge_index shape:", tuple(lbatch.edge_index.shape))
    if hasattr(lbatch, "edge_label_index"):
        print("  lbatch.edge_label_index shape:", tuple(lbatch.edge_label_index.shape))
    if hasattr(lbatch, "edge_label"):
        print("  lbatch.edge_label shape:", tuple(lbatch.edge_label.shape))
except Exception:
    print("FAILED LinkNeighborLoader test")
    traceback.print_exc()
    sys.exit(4)

# ----------------------------
# Other package imports (quick checks)
# ----------------------------
print("\n[5] Other package import checks")
checks = [
    ("numpy", "import numpy as np; print(np.__version__)"),
    ("pandas", "import pandas as pd; print(pd.__version__)"),
    ("scipy", "import scipy; print(scipy.__version__)"),
    ("sklearn", "import sklearn; print(sklearn.__version__)"),
    ("matplotlib", "import matplotlib; print(matplotlib.__version__)"),
    ("networkx", "import networkx as nx; print(nx.__version__)"),
    ("tqdm", "import tqdm; print(tqdm.__version__)"),
    ("datatable", "import datatable as dt; print(dt.__version__)"),
    ("xgboost", "import xgboost as xgb; print(xgb.__version__)"),
    ("snapml", "import snapml; print(snapml.__version__)"),
]

for name, code in checks:
    try:
        ns = {}
        exec(code, ns, ns)
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAILED ({e})")

print("\nAll critical local checks passed.")
print("=" * 80)