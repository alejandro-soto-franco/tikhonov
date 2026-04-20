"""AnnData `integrate` wrapper tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import tikhonov


def _make_toy_adata(n_per_batch: int = 40, d: int = 8, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_batch
    x = rng.standard_normal((n, 100))   # dummy counts
    z = rng.standard_normal((n, d))
    z[:n_per_batch, 0] += 1.5
    obs = pd.DataFrame(
        {"sample": ["a"] * n_per_batch + ["b"] * n_per_batch},
        index=[f"cell{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.obsm["X_pca"] = z
    return adata


def test_integrate_in_place_writes_obsm():
    adata = _make_toy_adata()
    result = tikhonov.integrate(adata, key="sample", max_iter=3, block_size=0.2)
    assert result is None  # in-place
    assert "X_harmony" in adata.obsm
    assert adata.obsm["X_harmony"].shape == adata.obsm["X_pca"].shape
    assert not np.allclose(adata.obsm["X_harmony"], adata.obsm["X_pca"])


def test_integrate_stores_metadata():
    adata = _make_toy_adata()
    tikhonov.integrate(adata, key="sample", max_iter=3, block_size=0.2)
    meta = adata.uns["tikhonov"]
    assert "n_iter" in meta
    assert "converged" in meta
    assert isinstance(meta["history"], list)


def test_integrate_copy_does_not_mutate():
    adata = _make_toy_adata()
    original = adata.obsm["X_pca"].copy()
    copied = tikhonov.integrate(adata, key="sample", copy=True, max_iter=3, block_size=0.2)
    assert copied is not None
    assert "X_harmony" not in adata.obsm
    assert "X_harmony" in copied.obsm
    np.testing.assert_array_equal(adata.obsm["X_pca"], original)


def test_integrate_missing_basis_errors():
    adata = _make_toy_adata()
    del adata.obsm["X_pca"]
    import pytest
    with pytest.raises(KeyError):
        tikhonov.integrate(adata, key="sample")
