//! One-hot design matrix for batch covariates.
//!
//! `Phi` is a `B × N` sparse 0/1 matrix where `B = sum_c n_levels_c` is the
//! total number of batch levels across all covariates, and `N` is the number
//! of cells. Each cell has exactly one `1` per covariate.
//!
//! The structure is one-hot per covariate with a known sparsity pattern, so
//! we store it as `(offset_per_covariate, row_of_cell_per_covariate)` plus
//! level names. No general sparse library needed.

use crate::HarmonyError;
use ndarray::{Array1, ArrayView2};

/// One-hot design matrix for batch covariates.
#[derive(Debug, Clone)]
pub struct Phi {
    /// Total levels across all covariates (= number of rows in the dense Phi).
    pub b: usize,
    /// Number of cells (= number of columns in the dense Phi).
    pub n: usize,
    /// Number of covariates.
    pub n_cov: usize,
    /// `offset[c]` = index of first row belonging to covariate `c`.
    /// `offset[n_cov]` = `b` (sentinel).
    pub offset: Vec<usize>,
    /// Per-covariate active row per cell: `row_of_cell[c * n + i]` = global row index for cell `i` in covariate `c`.
    pub row_of_cell: Vec<u32>,
    /// Categorical names, flattened covariate by covariate, in the same row order as `Phi`.
    pub level_names: Vec<Vec<String>>,
}

impl Phi {
    /// Build from a `(n, n_cov)` label matrix whose entries are per-covariate codes.
    ///
    /// Level names default to `c{cov}_l{level}` since the caller (CLI / Python) owns the
    /// real names and attaches them separately if needed.
    pub fn from_codes(labels: ArrayView2<'_, u32>) -> Result<Self, HarmonyError> {
        let (n, n_cov) = labels.dim();
        if n == 0 || n_cov == 0 {
            return Err(HarmonyError::ShapeMismatch(format!(
                "labels is {n}x{n_cov}; must be non-empty in both dims"
            )));
        }

        let mut level_names: Vec<Vec<String>> = Vec::with_capacity(n_cov);
        let mut offset = Vec::with_capacity(n_cov + 1);
        offset.push(0);

        for c in 0..n_cov {
            let col = labels.column(c);
            let max_code = col.iter().copied().max().unwrap_or(0) as usize;
            let n_levels = max_code + 1;
            level_names.push((0..n_levels).map(|l| format!("c{c}_l{l}")).collect());
            offset.push(offset[c] + n_levels);
        }

        let b = offset[n_cov];
        let mut row_of_cell = vec![0u32; n_cov * n];
        for c in 0..n_cov {
            let base = offset[c] as u32;
            for i in 0..n {
                row_of_cell[c * n + i] = base + labels[[i, c]];
            }
        }

        Ok(Self {
            b,
            n,
            n_cov,
            offset,
            row_of_cell,
            level_names,
        })
    }

    /// Row-wise sum of `Phi` (i.e., count of cells per batch level). `b`-length.
    pub fn n_b(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.b);
        for c in 0..self.n_cov {
            for i in 0..self.n {
                let row = self.row_of_cell[c * self.n + i] as usize;
                out[row] += 1.0;
            }
        }
        out
    }

    /// Prior probability per batch level: `Pr_b = N_b / N`.
    ///
    /// Harmony-R uses `sum(Phi, 1) / N`. With multi-covariate, each cell contributes
    /// `n_cov` ones to the row sums, so the per-covariate columns of `Pr_b` sum to 1.
    pub fn pr_b(&self) -> Array1<f64> {
        self.n_b() / (self.n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn single_covariate_three_levels() {
        let labels = array![[0u32], [1], [2], [0], [1]];
        let phi = Phi::from_codes(labels.view()).unwrap();
        assert_eq!(phi.b, 3);
        assert_eq!(phi.n, 5);
        assert_eq!(phi.n_cov, 1);
        assert_eq!(phi.offset, vec![0, 3]);
        assert_eq!(phi.row_of_cell, vec![0, 1, 2, 0, 1]);
        let counts = phi.n_b();
        assert_eq!(counts.to_vec(), vec![2.0, 2.0, 1.0]);
    }

    #[test]
    fn two_covariates_offsets_stack() {
        let labels = array![[0u32, 0], [0, 1], [1, 0], [1, 1]];
        let phi = Phi::from_codes(labels.view()).unwrap();
        assert_eq!(phi.b, 4);
        assert_eq!(phi.n, 4);
        assert_eq!(phi.n_cov, 2);
        assert_eq!(phi.offset, vec![0, 2, 4]);
        // First covariate: rows 0,1. Second: rows 2,3.
        assert_eq!(phi.row_of_cell, vec![0, 0, 1, 1, 2, 3, 2, 3]);
    }

    #[test]
    fn empty_labels_rejected() {
        let labels = ndarray::Array2::<u32>::zeros((0, 1));
        assert!(Phi::from_codes(labels.view()).is_err());
    }
}
