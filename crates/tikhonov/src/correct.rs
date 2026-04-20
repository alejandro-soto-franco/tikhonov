//! Mixture-of-experts ridge regression correction.
//!
//! Per cluster `k`, with `Phi_moe = [1; Phi]` (B+1 × N):
//!
//! ```text
//! A_k = Phi_moe · diag(R_k) · Phi_moe' + diag(lambda)    # (B+1) × (B+1), SPD
//! b_k = Phi_moe · diag(R_k) · Z_orig'                    # (B+1) × d
//! W_k = A_k^{-1} · b_k                                   # via Cholesky
//! W_k[0, :] = 0                                          # keep intercept
//! Z_corr -= W_k' · (Phi_moe · diag(R_k))                 # d × N delta
//! ```

use crate::{HarmonyError, Phi};
use faer::Side;
use faer::linalg::solvers::Solve;
use faer::mat::Mat;
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Default ridge penalty per row of `Phi_moe`: `[0.0, 1.0, 1.0, ..., 1.0]`.
pub fn default_lambda(b: usize) -> Vec<f64> {
    let mut v = vec![1.0; b + 1];
    v[0] = 0.0;
    v
}

/// Build `A_k` = Phi_moe · diag(R_k) · Phi_moe' + diag(lambda).
fn build_a_k(r_k: ArrayView1<'_, f64>, phi: &Phi, lambda: &[f64]) -> Mat<f64> {
    let bp1 = phi.b + 1;
    let mut a = Mat::<f64>::zeros(bp1, bp1);

    let r_sum: f64 = r_k.iter().sum();
    a[(0, 0)] = r_sum + lambda[0];

    // row_mass[b] = sum_{i,c: row_of_cell[c,i]==b} R_k[i].
    let mut row_mass = vec![0.0f64; phi.b];
    for c in 0..phi.n_cov {
        for i in 0..phi.n {
            let row = phi.row_of_cell[c * phi.n + i] as usize;
            row_mass[row] += r_k[i];
        }
    }
    for b in 0..phi.b {
        a[(0, b + 1)] = row_mass[b];
        a[(b + 1, 0)] = row_mass[b];
        a[(b + 1, b + 1)] = row_mass[b] + lambda[b + 1];
    }

    // Cross-covariate off-diagonals.
    if phi.n_cov > 1 {
        for i in 0..phi.n {
            let ri = r_k[i];
            for c1 in 0..phi.n_cov {
                let b1 = phi.row_of_cell[c1 * phi.n + i] as usize;
                for c2 in (c1 + 1)..phi.n_cov {
                    let b2 = phi.row_of_cell[c2 * phi.n + i] as usize;
                    a[(b1 + 1, b2 + 1)] += ri;
                    a[(b2 + 1, b1 + 1)] += ri;
                }
            }
        }
    }
    a
}

/// Build `b_k` = Phi_moe · diag(R_k) · Z_orig' as a `(B+1) × d` dense matrix.
fn build_b_k(r_k: ArrayView1<'_, f64>, phi: &Phi, z_orig: ArrayView2<'_, f64>) -> Mat<f64> {
    let d = z_orig.nrows();
    let bp1 = phi.b + 1;
    let mut out = Mat::<f64>::zeros(bp1, d);

    // Row 0 (intercept): sum_i R_k[i] * Z[:, i].
    for i in 0..phi.n {
        let ri = r_k[i];
        if ri == 0.0 {
            continue;
        }
        for row in 0..d {
            out[(0, row)] += ri * z_orig[[row, i]];
        }
    }
    // Rows 1..B+1: scatter each cell into its batch row per covariate.
    for c in 0..phi.n_cov {
        for i in 0..phi.n {
            let ri = r_k[i];
            if ri == 0.0 {
                continue;
            }
            let row = phi.row_of_cell[c * phi.n + i] as usize;
            for d_row in 0..d {
                out[(row + 1, d_row)] += ri * z_orig[[d_row, i]];
            }
        }
    }
    out
}

/// Apply the per-cluster ridge correction to `Z_corr` in place.
///
/// `z_orig` and `z_corr` are both `d × N`. `z_corr` must start as a copy of `z_orig`.
pub fn apply_moe_ridge(
    z_orig: ArrayView2<'_, f64>,
    z_corr: &mut Array2<f64>,
    r: ArrayView2<'_, f64>,
    phi: &Phi,
    lambda: &[f64],
) -> Result<(), HarmonyError> {
    let (d, n) = z_orig.dim();
    debug_assert_eq!(z_corr.dim(), (d, n));
    let k = r.nrows();

    for kk in 0..k {
        let r_k = r.row(kk);
        let a_k = build_a_k(r_k, phi, lambda);
        let b_k = build_b_k(r_k, phi, z_orig);

        let llt = a_k
            .llt(Side::Lower)
            .map_err(|e| HarmonyError::SingularRidge {
                cluster: kk,
                reason: format!("{e:?}"),
            })?;
        let mut w = b_k.clone();
        llt.solve_in_place(w.as_mut());

        // Zero intercept row.
        for col in 0..d {
            w[(0, col)] = 0.0;
        }

        // Z_corr -= W' · (Phi_moe · diag(R_k)).
        // For each cell i: delta[:, i] = R_k[i] * sum_c W[row_of_cell[c,i]+1, :]
        // (intercept row is zero so it drops out).
        for i in 0..phi.n {
            let ri = r_k[i];
            if ri == 0.0 {
                continue;
            }
            for c in 0..phi.n_cov {
                let row = phi.row_of_cell[c * phi.n + i] as usize + 1;
                for d_row in 0..d {
                    z_corr[[d_row, i]] -= ri * w[(row, d_row)];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn default_lambda_has_zero_intercept() {
        let l = default_lambda(3);
        assert_eq!(l, vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn single_cluster_reduces_batch_mean_gap() {
        let z = array![[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.5, -1.5]];
        let labels = array![[0u32], [0], [1], [1]];
        let phi = Phi::from_codes(labels.view()).unwrap();
        let r = array![[1.0, 1.0, 1.0, 1.0]];
        let mut z_corr = z.clone();
        let lam = default_lambda(phi.b);
        apply_moe_ridge(z.view(), &mut z_corr, r.view(), &phi, &lam).unwrap();
        let mean_b0_before = (z[[0, 0]] + z[[0, 1]]) / 2.0;
        let mean_b1_before = (z[[0, 2]] + z[[0, 3]]) / 2.0;
        let mean_b0_after = (z_corr[[0, 0]] + z_corr[[0, 1]]) / 2.0;
        let mean_b1_after = (z_corr[[0, 2]] + z_corr[[0, 3]]) / 2.0;
        let gap_before = (mean_b0_before - mean_b1_before).abs();
        let gap_after = (mean_b0_after - mean_b1_after).abs();
        assert!(gap_after < gap_before, "gap: {gap_before} -> {gap_after}");
    }
}
