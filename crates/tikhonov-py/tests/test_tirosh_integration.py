"""
Tirosh 2016 oligodendroglioma integration test (Phase 4 Tier 2).

Loads the downsampled Tirosh counts shipped with inferCNV, runs scanpy
normalise + PCA, calls `tikhonov.integrate` with patient as the batch
covariate, and asserts:

1. `adata.obsm["X_harmony"]` is populated with the correct shape.
2. The per-patient mean in PC1 shrinks toward the grand mean (batch gap
   decreases for every patient with enough cells).
3. Cluster-composition diversity (normalised entropy over patients) does
   not collapse: the corrected embedding's k-means clusters still contain
   cells from multiple patients after correction.

Gated behind `@pytest.mark.integration`; skipped unless run explicitly.
Cached via `platformdirs` under `~/.cache/tikhonov/test_fixtures/tirosh`.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")
platformdirs = pytest.importorskip("platformdirs")

import tikhonov

_BASE = "https://raw.githubusercontent.com/broadinstitute/infercnv/master/inst/extdata/"
_FILES = {
    "counts": "oligodendroglioma_expression_downsampled.counts.matrix.gz",
    "annotations": "oligodendroglioma_annotations_downsampled.txt",
}


def _cache_dir() -> Path:
    d = Path(platformdirs.user_cache_dir("tikhonov")) / "test_fixtures" / "tirosh"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(name: str) -> Path:
    dest = _cache_dir() / name
    if dest.exists():
        return dest
    url = _BASE + name
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def _load_tirosh() -> anndata.AnnData:
    counts_path = _fetch(_FILES["counts"])
    annot_path = _fetch(_FILES["annotations"])

    with gzip.open(counts_path, "rt") as f:
        counts = pd.read_csv(f, sep="\t", index_col=0)
    # counts is genes x cells; transpose to cells x genes.
    x = counts.to_numpy(dtype=np.float32).T
    cell_ids = counts.columns.to_list()

    annot = pd.read_csv(annot_path, sep="\t", header=None, names=["cell", "patient"])
    annot = annot.set_index("cell").loc[cell_ids]

    adata = anndata.AnnData(X=x, obs=annot.reset_index(drop=False).set_index("cell"))
    return adata


def _simple_pca(x: np.ndarray, n_comps: int) -> np.ndarray:
    """Deterministic SVD-PCA without scanpy (avoid the heavy dep in tests)."""
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return (u[:, :n_comps] * s[:n_comps]).astype(np.float64)


def _per_batch_gap(emb: np.ndarray, labels: np.ndarray, pc: int = 0) -> float:
    """Max pairwise absolute difference in PC `pc` means across batches."""
    batches = np.unique(labels)
    means = [emb[labels == b, pc].mean() for b in batches]
    return float(max(means) - min(means))


def _cluster_patient_entropy(emb: np.ndarray, labels: np.ndarray, k: int = 6) -> float:
    """Mean normalised entropy of patient composition across k-means clusters.

    1.0 = every cluster perfectly mixes all patients. 0.0 = each cluster
    holds a single patient. Post-integration should be noticeably above
    pre-integration.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
    assignments = km.labels_
    patients = np.unique(labels)
    n_p = len(patients)
    mean_ent = 0.0
    for c in range(k):
        mask = assignments == c
        if not mask.any():
            continue
        counts = np.array([(labels[mask] == p).sum() for p in patients], dtype=np.float64)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs))
        mean_ent += ent / np.log(n_p)
    return mean_ent / k


@pytest.mark.integration
def test_tirosh_integration_shrinks_batch_gap():
    pytest.importorskip("sklearn")
    adata = _load_tirosh()
    # Normalise + log1p + PCA → obsm["X_pca"].
    x = adata.X.astype(np.float64)
    lib_size = x.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0
    x_norm = np.log1p(x / lib_size * 1e4)
    pca = _simple_pca(x_norm, n_comps=20)
    adata.obsm["X_pca"] = pca

    patients = adata.obs["patient"].astype("category")
    patient_codes = patients.cat.codes.to_numpy()

    gap_before = _per_batch_gap(pca, patient_codes)
    ent_before = _cluster_patient_entropy(pca, patient_codes, k=6)

    tikhonov.integrate(
        adata,
        key="patient",
        basis="X_pca",
        adjusted_basis="X_harmony",
        max_iter=10,
        sigma=0.1,
        theta=2.0,
        block_size=0.05,
        random_state=0,
    )

    z_corr = adata.obsm["X_harmony"]
    assert z_corr.shape == pca.shape

    gap_after = _per_batch_gap(z_corr, patient_codes)
    ent_after = _cluster_patient_entropy(z_corr, patient_codes, k=6)

    # Functional parity assertions.
    assert gap_after < gap_before, f"per-patient gap did not shrink: {gap_before:.3f} -> {gap_after:.3f}"
    assert ent_after > ent_before, f"cluster mixing did not improve: {ent_before:.3f} -> {ent_after:.3f}"
    assert adata.uns["tikhonov"]["n_iter"] >= 1
