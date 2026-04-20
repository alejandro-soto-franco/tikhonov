//! Kmeans + entropy + cross-entropy objective, matching
//! `harmony::compute_objective` in harmony-R 1.2.4.
//!
//! The composite objective is
//!
//! ```text
//! objective = (kmeans_error + entropy + cross_entropy) * (2000 / N)
//! ```
//!
//! where
//!
//! * `kmeans_error = sum(R .* dist_mat)`,
//! * `entropy = sum( (R * log(R)) .* sigma )` (column-broadcast sigma),
//! * `cross_entropy = sum( (R .* sigma) .* ( (theta .* log((O+E)/E)) * Phi ) )`.

use crate::Phi;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

/// Safe entropy term: `R * log(R)`, with `0 * log(0) = 0`.
pub fn safe_entropy(r: ArrayView2<'_, f64>) -> Array2<f64> {
    r.mapv(|v| if v <= 0.0 { 0.0 } else { v * v.ln() })
}

/// Component objectives from harmony-R: `(kmeans, entropy, cross_entropy, total)`.
/// All four are pre-scaled by `2000 / N`.
pub fn compute(
    r: ArrayView2<'_, f64>,
    dist_mat: ArrayView2<'_, f64>,
    o: ArrayView2<'_, f64>,
    e: ArrayView2<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    theta: ArrayView1<'_, f64>,
    phi: &Phi,
) -> (f64, f64, f64, f64) {
    let n = r.ncols() as f64;
    let norm_const = 2000.0 / n;

    // kmeans_error = sum_{k,i} R[k,i] * dist_mat[k,i]
    let mut kmeans_error = 0.0;
    for (r_row, d_row) in r.axis_iter(Axis(0)).zip(dist_mat.axis_iter(Axis(0))) {
        kmeans_error += r_row
            .iter()
            .zip(d_row.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    }

    // entropy = sum_{k,i} sigma[k] * R[k,i] * log(R[k,i])
    let entr = safe_entropy(r);
    let mut entropy = 0.0;
    for (k, row) in entr.axis_iter(Axis(0)).enumerate() {
        entropy += sigma[k] * row.sum();
    }

    // cross_entropy: build a K×B "ratio" matrix, then scatter by covariate.
    let (k_n, b_n) = o.dim();
    let mut ratio = Array2::<f64>::zeros((k_n, b_n));
    for c in 0..phi.n_cov {
        let t = theta[c];
        for b in phi.offset[c]..phi.offset[c + 1] {
            for k in 0..k_n {
                let denom = e[[k, b]];
                if denom > 0.0 {
                    let q = (o[[k, b]] + denom) / denom;
                    ratio[[k, b]] = t * q.ln();
                }
            }
        }
    }
    let mut cross_entropy = 0.0;
    for k in 0..k_n {
        let sigma_k = sigma[k];
        for i in 0..phi.n {
            let mut cell_sum = 0.0;
            for c in 0..phi.n_cov {
                let row = phi.row_of_cell[c * phi.n + i] as usize;
                cell_sum += ratio[[k, row]];
            }
            cross_entropy += sigma_k * r[[k, i]] * cell_sum;
        }
    }

    let total = kmeans_error + entropy + cross_entropy;
    (
        kmeans_error * norm_const,
        entropy * norm_const,
        cross_entropy * norm_const,
        total * norm_const,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, array};

    #[test]
    fn safe_entropy_handles_zero() {
        let r = array![[0.0, 0.5], [1.0, 0.5]];
        let h = safe_entropy(r.view());
        assert_abs_diff_eq!(h[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(h[[0, 1]], 0.5 * (0.5_f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(h[[1, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn objective_on_toy_inputs() {
        // K=2, N=3, B=2, single covariate.
        let r = array![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];
        let dist = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let o = array![[1.0, 0.5], [0.5, 1.0]];
        let e = array![[0.75, 0.75], [0.75, 0.75]];
        let sigma = Array1::from(vec![0.1, 0.1]);
        let theta = Array1::from(vec![2.0]);

        let labels = array![[0u32], [1], [0]];
        let phi = crate::Phi::from_codes(labels.view()).unwrap();

        let (km, ent, cross, total) = compute(
            r.view(),
            dist.view(),
            o.view(),
            e.view(),
            sigma.view(),
            theta.view(),
            &phi,
        );
        assert!(km.is_finite());
        assert!(ent.is_finite());
        assert!(cross.is_finite());
        assert!(total.is_finite());
        assert!(km > 0.0);
        assert!(ent < 0.0);
    }
}
