//! Hyperparameters for [`crate::run_harmony`].
//!
//! Every parameter accepted by `harmony-R` 1.2.4's `RunHarmony` has a home
//! here. Use [`HarmonyConfig::new`] plus the `with_*` builder methods to
//! construct a config, or [`HarmonyConfig::default`] for harmony-R defaults.

use serde::{Deserialize, Serialize};

/// Full harmony-R 1.2.4 hyperparameter surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarmonyConfig {
    /// Number of soft clusters. `None` resolves to `min(100, n / 30).max(1)` at runtime.
    pub nclust: Option<usize>,

    /// Maximum outer (harmony) iterations.
    pub max_iter: usize,

    /// Maximum inner (kmeans) iterations per outer step.
    pub max_iter_cluster: usize,

    /// Soft-clustering temperature. Scalar broadcasts across clusters.
    pub sigma: f64,

    /// Per-covariate diversity penalty. Length-1 vec broadcasts across all covariates.
    pub theta: Vec<f64>,

    /// Per-batch ridge penalty. `None` triggers harmony-R's automatic lambda estimation.
    pub lambda: Option<Vec<f64>>,

    /// Kmeans convergence tolerance (relative objective change, window-3).
    pub epsilon_cluster: f64,

    /// Harmony convergence tolerance (single-step relative objective change).
    pub epsilon_harmony: f64,

    /// Share of cells whose cluster assignments may flip per kmeans iteration.
    /// Value in `(0, 1]`. Harmony-R auto-sets this to `0.2` when `n < 40`.
    pub block_size: f64,

    /// Per-covariate reference level (None = no reference; Some(code) = hold fixed).
    pub reference_values: Option<Vec<Option<u32>>>,

    /// RNG seed for the shuffle inside the block-randomised R update.
    pub seed: u64,

    /// Emit progress to stderr.
    pub verbose: bool,

    /// Rayon thread pool size. `None` uses the rayon global default.
    pub n_threads: Option<usize>,
}

impl HarmonyConfig {
    /// Construct a config with harmony-R 1.2.4 defaults.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_nclust(mut self, nclust: usize) -> Self {
        self.nclust = Some(nclust);
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_max_iter_cluster(mut self, max_iter_cluster: usize) -> Self {
        self.max_iter_cluster = max_iter_cluster;
        self
    }

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    pub fn with_theta<I: IntoIterator<Item = f64>>(mut self, theta: I) -> Self {
        self.theta = theta.into_iter().collect();
        self
    }

    pub fn with_lambda<I: IntoIterator<Item = f64>>(mut self, lambda: I) -> Self {
        self.lambda = Some(lambda.into_iter().collect());
        self
    }

    pub fn with_epsilon_cluster(mut self, eps: f64) -> Self {
        self.epsilon_cluster = eps;
        self
    }

    pub fn with_epsilon_harmony(mut self, eps: f64) -> Self {
        self.epsilon_harmony = eps;
        self
    }

    pub fn with_block_size(mut self, block_size: f64) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_reference_values(mut self, refs: Vec<Option<u32>>) -> Self {
        self.reference_values = Some(refs);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    /// Resolve [`HarmonyConfig::nclust`] for a dataset of `n` cells.
    #[allow(dead_code)]
    pub(crate) fn resolved_nclust(&self, n: usize) -> usize {
        self.nclust.unwrap_or_else(|| (n / 30).clamp(1, 100))
    }

    /// Resolve [`HarmonyConfig::theta`] for `n_cov` covariates.
    #[allow(dead_code)]
    pub(crate) fn resolved_theta(&self, n_cov: usize) -> Vec<f64> {
        match self.theta.len() {
            0 => vec![2.0; n_cov],
            1 => vec![self.theta[0]; n_cov],
            k if k == n_cov => self.theta.clone(),
            _ => panic!(
                "theta must have length 0, 1, or {}; got {}",
                n_cov,
                self.theta.len()
            ),
        }
    }
}

impl Default for HarmonyConfig {
    fn default() -> Self {
        Self {
            nclust: None,
            max_iter: 10,
            max_iter_cluster: 200,
            sigma: 0.1,
            theta: vec![2.0],
            lambda: None,
            epsilon_cluster: 1e-5,
            epsilon_harmony: 1e-4,
            block_size: 0.05,
            reference_values: None,
            seed: 0,
            verbose: false,
            n_threads: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_harmony_r_1_2_4() {
        let c = HarmonyConfig::new();
        assert_eq!(c.max_iter, 10);
        assert_eq!(c.max_iter_cluster, 200);
        assert!((c.sigma - 0.1).abs() < 1e-12);
        assert_eq!(c.theta, vec![2.0]);
        assert!((c.epsilon_cluster - 1e-5).abs() < 1e-18);
        assert!((c.epsilon_harmony - 1e-4).abs() < 1e-18);
        assert!((c.block_size - 0.05).abs() < 1e-12);
        assert_eq!(c.seed, 0);
    }

    #[test]
    fn nclust_resolves() {
        let c = HarmonyConfig::new();
        assert_eq!(c.resolved_nclust(30), 1);
        assert_eq!(c.resolved_nclust(300), 10);
        assert_eq!(c.resolved_nclust(1_000_000), 100);
    }

    #[test]
    fn theta_broadcasts() {
        let c = HarmonyConfig::new().with_theta([2.0]);
        assert_eq!(c.resolved_theta(3), vec![2.0, 2.0, 2.0]);

        let c = HarmonyConfig::new().with_theta([1.0, 3.0]);
        assert_eq!(c.resolved_theta(2), vec![1.0, 3.0]);
    }

    #[test]
    fn builder_is_chainable() {
        let c = HarmonyConfig::new()
            .with_nclust(50)
            .with_max_iter(20)
            .with_sigma(0.05)
            .with_seed(42);
        assert_eq!(c.nclust, Some(50));
        assert_eq!(c.max_iter, 20);
        assert!((c.sigma - 0.05).abs() < 1e-12);
        assert_eq!(c.seed, 42);
    }
}
