"""Array-level API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tikhonov


def _make_two_batch_blob(n_per_batch: int = 50, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((2 * n_per_batch, d))
    z[:n_per_batch, 0] += 2.0
    meta = pd.DataFrame(
        {"sample": ["a"] * n_per_batch + ["b"] * n_per_batch}
    )
    return z, meta


def test_run_harmony_shrinks_batch_gap():
    z, meta = _make_two_batch_blob()
    out = tikhonov.run_harmony(z, meta, "sample", max_iter=5, block_size=0.2)
    assert out.Z_corr.shape == z.shape  # (n_cells, d)

    batch_a_before = z[:50, 0].mean()
    batch_b_before = z[50:, 0].mean()
    batch_a_after = out.Z_corr[:50, 0].mean()
    batch_b_after = out.Z_corr[50:, 0].mean()

    gap_before = abs(batch_a_before - batch_b_before)
    gap_after = abs(batch_a_after - batch_b_after)
    assert gap_after < gap_before, f"gap: {gap_before:.3f} -> {gap_after:.3f}"


def test_run_harmony_history_shape():
    z, meta = _make_two_batch_blob()
    out = tikhonov.run_harmony(z, meta, "sample", max_iter=3, block_size=0.2)
    assert isinstance(out.history, list)
    assert len(out.history) >= 1
    first = out.history[0]
    assert set(first).issuperset(
        {"iter", "cluster_iters", "kmeans_cost", "objective", "elapsed_ms"}
    )


def test_run_harmony_deterministic_on_seed():
    z, meta = _make_two_batch_blob()
    a = tikhonov.run_harmony(z, meta, "sample", max_iter=3, block_size=0.2, random_state=7)
    b = tikhonov.run_harmony(z, meta, "sample", max_iter=3, block_size=0.2, random_state=7)
    np.testing.assert_array_equal(a.Z_corr, b.Z_corr)


def test_run_harmony_accepts_f32_input():
    z, meta = _make_two_batch_blob()
    out = tikhonov.run_harmony(z.astype(np.float32), meta, "sample", max_iter=2, block_size=0.2)
    assert out.Z_corr.dtype == np.float64


def test_run_harmony_multi_covariate():
    rng = np.random.default_rng(42)
    z = rng.standard_normal((200, 15))
    meta = pd.DataFrame(
        {
            "sample": rng.choice(["a", "b", "c"], size=200),
            "donor": rng.choice(["d1", "d2"], size=200),
        }
    )
    out = tikhonov.run_harmony(z, meta, ["sample", "donor"], max_iter=3, block_size=0.2)
    assert out.Z_corr.shape == (200, 15)


def test_run_harmony_rejects_mismatched_meta():
    z = np.zeros((10, 3))
    meta = pd.DataFrame({"sample": ["a"] * 5})
    with pytest.raises(ValueError):
        tikhonov.run_harmony(z, meta, "sample")
