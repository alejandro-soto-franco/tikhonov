//! Soft-clustering E-step (`update_r_block`) and M-step (`update_y`).
//!
//! Mirrors `harmony::update_R` and the first two lines of
//! `harmony::cluster_cpp` in harmony-R 1.2.4.

use crate::Phi;
use crate::embed::l2_normalize_cols;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// Distance matrix `dist[k, i] = 2 * (1 - Y[:, k] . Z_cos[:, i])`.
pub fn dist_mat(y: ArrayView2<'_, f64>, z_cos: ArrayView2<'_, f64>) -> Array2<f64> {
    let k = y.ncols();
    let n = z_cos.ncols();
    let mut out = Array2::<f64>::zeros((k, n));
    for kk in 0..k {
        for i in 0..n {
            let dot = y.column(kk).dot(&z_cos.column(i));
            out[[kk, i]] = 2.0 * (1.0 - dot);
        }
    }
    out
}

/// M-step: `Y = normalise(Z_cos * R', 2, 0)`.
pub fn update_y(z_cos: ArrayView2<'_, f64>, r: ArrayView2<'_, f64>) -> Array2<f64> {
    // Z_cos is (d, n); R is (k, n); Y = Z_cos * R' → (d, k).
    let (d, n) = z_cos.dim();
    let k = r.nrows();
    let mut y = Array2::<f64>::zeros((d, k));
    for row in 0..d {
        for col in 0..k {
            let mut acc = 0.0;
            for i in 0..n {
                acc += z_cos[[row, i]] * r[[col, i]];
            }
            y[[row, col]] = acc;
        }
    }
    l2_normalize_cols(y.view())
}

/// Column-L1 normalise in place: for each column, divide by its sum.
pub fn normalise_cols_l1(m: &mut Array2<f64>) {
    for mut col in m.axis_iter_mut(Axis(1)) {
        let s: f64 = col.iter().sum();
        if s > 0.0 {
            col.mapv_inplace(|v| v / s);
        }
    }
}

/// Block-randomised R update (harmony-R's `update_R`).
///
/// Modifies `r`, `o`, `e` in place; expects `dist_mat` already computed
/// against the current `Y`. Returns the `scale_dist` used (for potential
/// re-use in objective calculations).
#[allow(clippy::too_many_arguments)]
pub fn update_r_block(
    r: &mut Array2<f64>,
    o: &mut Array2<f64>,
    e: &mut Array2<f64>,
    dist_mat: ArrayView2<'_, f64>,
    phi: &Phi,
    pr_b: ArrayView1<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    theta: ArrayView1<'_, f64>,
    block_size: f64,
    rng: &mut ChaCha8Rng,
) -> Array2<f64> {
    let (k, n) = dist_mat.dim();

    // scale_dist = exp(-dist / sigma), column-L1-normalised.
    let mut scale_dist = Array2::<f64>::zeros((k, n));
    for kk in 0..k {
        for i in 0..n {
            scale_dist[[kk, i]] = (-dist_mat[[kk, i]] / sigma[kk]).exp();
        }
    }
    normalise_cols_l1(&mut scale_dist);

    // Build shuffle.
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);

    let cells_per_block = ((n as f64) * block_size).floor() as usize;
    let n_blocks = ((1.0 / block_size).ceil() as usize).max(1);

    for block_i in 0..n_blocks {
        let idx_min = block_i * cells_per_block;
        let idx_max = if block_i == n_blocks - 1 {
            n
        } else {
            (block_i + 1) * cells_per_block
        };
        if idx_min >= n {
            break;
        }
        let slice = &order[idx_min..idx_max.min(n)];

        // Step 1: subtract this block's contribution from O and E.
        for &i in slice {
            for kk in 0..k {
                let r_ki = r[[kk, i]];
                for b in 0..o.ncols() {
                    e[[kk, b]] -= r_ki * pr_b[b];
                }
                for c in 0..phi.n_cov {
                    let b = phi.row_of_cell[c * phi.n + i] as usize;
                    o[[kk, b]] -= r_ki;
                }
            }
        }

        // Step 2: recompute R for the block.
        for &i in slice {
            for kk in 0..k {
                let mut prod = 1.0;
                for c in 0..phi.n_cov {
                    let b = phi.row_of_cell[c * phi.n + i] as usize;
                    let ratio = e[[kk, b]] / (o[[kk, b]] + e[[kk, b]]);
                    if ratio > 0.0 {
                        prod *= ratio.powf(theta[c]);
                    } else {
                        prod = 0.0;
                    }
                }
                r[[kk, i]] = scale_dist[[kk, i]] * prod;
            }
        }
        // Column-L1 normalise just the block's columns.
        for &i in slice {
            let s: f64 = (0..k).map(|kk| r[[kk, i]]).sum();
            if s > 0.0 {
                for kk in 0..k {
                    r[[kk, i]] /= s;
                }
            }
        }

        // Step 3: add the block's new contribution back to O and E.
        for &i in slice {
            for kk in 0..k {
                let r_ki = r[[kk, i]];
                for b in 0..o.ncols() {
                    e[[kk, b]] += r_ki * pr_b[b];
                }
                for c in 0..phi.n_cov {
                    let b = phi.row_of_cell[c * phi.n + i] as usize;
                    o[[kk, b]] += r_ki;
                }
            }
        }
    }

    scale_dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn dist_mat_unit_cosine() {
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let z_cos = array![[1.0, 0.0], [0.0, 1.0]];
        let d = dist_mat(y.view(), z_cos.view());
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[[1, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[[0, 1]], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn update_y_preserves_unit_norm() {
        let z = array![[0.6, 0.8, 0.0], [0.8, -0.6, 1.0]];
        let r = array![[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
        let y = update_y(z.view(), r.view());
        for k in 0..y.ncols() {
            let n: f64 = y.column(k).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(n, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn normalise_cols_l1_sums_to_one() {
        let mut m = array![[1.0, 2.0], [3.0, 0.0]];
        normalise_cols_l1(&mut m);
        assert_abs_diff_eq!(m[[0, 0]] + m[[1, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m[[0, 1]] + m[[1, 1]], 1.0, epsilon = 1e-12);
    }
}
