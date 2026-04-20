"""AnnData adapter: unpack obsm/obs into numpy arrays and call the Rust core.

The Rust FFI never sees AnnData; all AnnData handling lives here in Python,
so we inherit upstream improvements to the `anndata` package for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tikhonov.harmony import _build_label_matrix, run_harmony

if TYPE_CHECKING:
    import anndata as ad

    from tikhonov._core import HarmonyResult


def integrate(
    adata: "ad.AnnData",
    key: str | list[str],
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_harmony",
    copy: bool = False,
    **kwargs: Any,
) -> "ad.AnnData | HarmonyResult | None":
    """Run Harmony2 on `adata.obsm[basis]` and write to `adata.obsm[adjusted_basis]`.

    Args:
        adata: AnnData object. Must have `obsm[basis]` populated.
        key: `obs` column name (or list of names) to use as batch covariates.
        basis: key in `obsm` for the PC embedding (default `"X_pca"`).
        adjusted_basis: key in `obsm` where the corrected embedding is written.
        copy: if True, modify a copy of `adata` and return it; otherwise modify in place.
        **kwargs: passed through to `run_harmony`.

    Returns:
        If `copy=True`: the modified copy.
        If `copy=False`: None (in-place); the `HarmonyResult` is attached as
        `adata.uns["tikhonov"]` for downstream inspection.
    """
    target = adata.copy() if copy else adata

    if basis not in target.obsm:
        raise KeyError(f"adata.obsm[{basis!r}] not found")
    z = np.asarray(target.obsm[basis], dtype=np.float64)

    result = run_harmony(z, target.obs, key, **kwargs)
    target.obsm[adjusted_basis] = np.asarray(result.Z_corr, dtype=np.float64)

    # Stash metadata for downstream inspection.
    target.uns.setdefault("tikhonov", {})
    target.uns["tikhonov"] = {
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "history": list(result.history),
    }

    return target if copy else None
