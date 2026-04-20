//! Cosine embedding helpers and kmeans++ centroid initialisation.
//!
//! Matches the behaviour of `arma::normalise(Z, 2, 0)` (column-wise L2) and
//! Armadillo's `kmeans_centers` as used by harmony-R 1.2.4's
//! `harmony::init_cluster_cpp`.

use ndarray::{Array2, ArrayView2, Axis};
use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// Return a new `d × n` matrix with each column L2-normalised. Zero columns stay zero.
pub fn l2_normalize_cols(z: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut out = z.to_owned();
    for mut col in out.axis_iter_mut(Axis(1)) {
        let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            col.mapv_inplace(|v| v / norm);
        }
    }
    out
}

/// kmeans++ centroid initialisation on `z_cos` (`d × n`, columns unit-norm).
///
/// Returns a `d × k` matrix whose columns are centroids, L2-normalised.
/// Determinism: seeded `ChaCha8Rng`; two calls with the same seed produce the same centers.
pub fn kmeans_pp_init(z_cos: ArrayView2<'_, f64>, k: usize, seed: u64) -> Array2<f64> {
    let (d, n) = z_cos.dim();
    assert!(k >= 1 && k <= n, "k must be in 1..=n (got k={k}, n={n})");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centers = Array2::<f64>::zeros((d, k));
    let indices: Vec<usize> = (0..n).collect();

    // Pick first centroid uniformly at random.
    let first = *indices.choose(&mut rng).unwrap();
    centers.column_mut(0).assign(&z_cos.column(first));

    // Squared distance to the nearest chosen centroid per cell.
    let mut dists = vec![f64::INFINITY; n];
    for (i, col) in z_cos.axis_iter(Axis(1)).enumerate() {
        let diff = &col - &centers.column(0);
        dists[i] = diff.dot(&diff);
    }

    for c in 1..k {
        // Sample proportional to dists.
        let total: f64 = dists.iter().sum();
        if total == 0.0 {
            let idx = rng.random_range(0..n);
            centers.column_mut(c).assign(&z_cos.column(idx));
        } else {
            let mut pick: f64 = rng.random::<f64>() * total;
            let mut idx = 0usize;
            for (i, &d2) in dists.iter().enumerate() {
                pick -= d2;
                if pick <= 0.0 {
                    idx = i;
                    break;
                }
            }
            centers.column_mut(c).assign(&z_cos.column(idx));
        }

        // Update per-cell distance to the new closest centre.
        for (i, col) in z_cos.axis_iter(Axis(1)).enumerate() {
            let diff = &col - &centers.column(c);
            let d2 = diff.dot(&diff);
            if d2 < dists[i] {
                dists[i] = d2;
            }
        }
    }

    // L2-normalise centroid columns (matches `arma::normalise(Y, 2, 0)` in harmony-R).
    l2_normalize_cols(centers.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn l2_normalize_unit_columns() {
        let z = array![[3.0, 0.0], [4.0, 5.0]];
        let out = l2_normalize_cols(z.view());
        assert_abs_diff_eq!(out[[0, 0]], 0.6, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[1, 0]], 0.8, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn l2_normalize_zero_column_stays_zero() {
        let z = array![[0.0], [0.0]];
        let out = l2_normalize_cols(z.view());
        assert_eq!(out, array![[0.0], [0.0]]);
    }

    #[test]
    fn kmeans_pp_returns_unit_norm_centroids() {
        let z = array![[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, 0.5, -0.5]];
        let z_cos = l2_normalize_cols(z.view());
        let centers = kmeans_pp_init(z_cos.view(), 2, 42);
        for c in 0..2 {
            let norm = centers.column(c).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn kmeans_pp_deterministic_on_seed() {
        let z = array![[1.0, 0.0, 0.7, 0.3], [0.0, 1.0, 0.7, 0.3]];
        let z_cos = l2_normalize_cols(z.view());
        let a = kmeans_pp_init(z_cos.view(), 2, 7);
        let b = kmeans_pp_init(z_cos.view(), 2, 7);
        assert_eq!(a, b);
    }
}
