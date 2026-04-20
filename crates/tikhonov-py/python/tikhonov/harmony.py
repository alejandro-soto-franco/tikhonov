"""Array-level `run_harmony` with a harmonypy-compatible signature."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tikhonov._core import py_run_harmony

if TYPE_CHECKING:
    import pandas as pd
    from tikhonov._core import HarmonyResult


def _as_dxn(data_mat: np.ndarray, n_meta: int) -> np.ndarray:
    """Orient `data_mat` to (d, n). Harmonypy accepts either (n, d) or (d, n).

    We match harmonypy's heuristic: if either axis matches `n_meta`, that axis
    is `n`. If both match (square), assume (n, d). If neither matches, error.
    """
    rows, cols = data_mat.shape
    if cols == n_meta and rows != n_meta:
        return np.ascontiguousarray(data_mat, dtype=np.float64)
    if rows == n_meta and cols != n_meta:
        return np.ascontiguousarray(data_mat.T, dtype=np.float64)
    if rows == n_meta and cols == n_meta:
        return np.ascontiguousarray(data_mat.T, dtype=np.float64)
    raise ValueError(
        f"data_mat shape {data_mat.shape} cannot be oriented against meta_data "
        f"with {n_meta} rows"
    )


def _build_label_matrix(
    meta_data: "pd.DataFrame", vars_use: str | list[str]
) -> tuple[np.ndarray, list[list[str]]]:
    """Convert `meta_data[vars_use]` into an (n, n_cov) uint32 code matrix."""
    if isinstance(vars_use, str):
        vars_use = [vars_use]
    n = len(meta_data)
    labels = np.zeros((n, len(vars_use)), dtype=np.uint32)
    level_names: list[list[str]] = []
    for c, col in enumerate(vars_use):
        series = meta_data[col]
        if hasattr(series, "cat") and hasattr(series.cat, "codes"):
            codes = series.cat.codes.to_numpy()
            cats = list(series.cat.categories.astype(str))
        else:
            cats_unique, codes = np.unique(series.to_numpy(), return_inverse=True)
            cats = [str(c) for c in cats_unique]
        labels[:, c] = codes.astype(np.uint32)
        level_names.append(cats)
    return labels, level_names


def run_harmony(
    data_mat: np.ndarray,
    meta_data: "pd.DataFrame",
    vars_use: str | list[str],
    *,
    nclust: int | None = None,
    sigma: float = 0.1,
    theta: float | list[float] = 2.0,
    lambda_: float | list[float] | None = 1.0,
    max_iter: int = 10,
    max_iter_cluster: int = 200,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
    block_size: float = 0.05,
    random_state: int = 0,
    verbose: bool = False,
    n_threads: int | None = None,
) -> "HarmonyResult":
    """Run Harmony2 on a PC embedding.

    Args match harmonypy's `run_harmony` signature for drop-in replacement.
    `data_mat` is auto-oriented from `(n, d)` (harmonypy-style) or `(d, n)`.
    """
    z_dxn = _as_dxn(data_mat, len(meta_data))
    labels, _ = _build_label_matrix(meta_data, vars_use)

    kwargs: dict = {
        "sigma": float(sigma),
        "theta": theta,
        "max_iter": int(max_iter),
        "max_iter_cluster": int(max_iter_cluster),
        "epsilon_cluster": float(epsilon_cluster),
        "epsilon_harmony": float(epsilon_harmony),
        "block_size": float(block_size),
        "random_state": int(random_state),
        "verbose": bool(verbose),
    }
    if nclust is not None:
        kwargs["nclust"] = int(nclust)
    if lambda_ is not None:
        kwargs["lambda_"] = lambda_
    if n_threads is not None:
        kwargs["n_threads"] = int(n_threads)

    return py_run_harmony(z_dxn, labels, **kwargs)
