//! Error types for the tikhonov crate.

use thiserror::Error;

/// All failure modes of [`crate::run_harmony`].
#[derive(Debug, Error)]
pub enum HarmonyError {
    /// Input shapes are inconsistent.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A ridge regression system was singular or ill-conditioned.
    #[error("singular ridge system in cluster {cluster}: {reason}")]
    SingularRidge { cluster: usize, reason: String },

    /// Optimization failed to converge within [`crate::HarmonyConfig::max_iter`].
    #[error("harmony objective failed to converge in {iters} iterations")]
    NotConverged { iters: usize },

    /// Configuration is invalid or contradictory.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = HarmonyError::ShapeMismatch("z has 5 cols, labels has 3 rows".into());
        assert_eq!(
            e.to_string(),
            "shape mismatch: z has 5 cols, labels has 3 rows"
        );
    }
}
