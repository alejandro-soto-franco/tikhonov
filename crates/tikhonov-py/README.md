# tikhonov (Python)

Python bindings for [tikhonov](https://github.com/alejandro-soto-franco/tikhonov), a pure-Rust port of Harmony2 for single-cell data integration.

## Install

```bash
pip install tikhonov
```

## Quickstart (AnnData)

```python
import tikhonov
import scanpy as sc

adata = sc.read_h5ad("my_data.h5ad")
# PCA already computed → obsm["X_pca"]
tikhonov.integrate(adata, key="sample", basis="X_pca", adjusted_basis="X_harmony")
# adata.obsm["X_harmony"] now holds the corrected embedding
```

## Quickstart (arrays, harmonypy-compatible signature)

```python
import numpy as np
import pandas as pd
import tikhonov

Z = np.random.randn(1000, 20)                # (n_cells, n_PCs)
meta = pd.DataFrame({"sample": np.random.choice(["a", "b"], 1000)})
out = tikhonov.run_harmony(Z, meta, "sample", sigma=0.1, theta=2.0, max_iter=10)
print(out.Z_corr.shape)                      # (1000, 20)
print(out.converged, out.n_iter)
```

## License

Apache-2.0.
