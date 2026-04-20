//! Convergence diagnostics emitted by [`crate::run_harmony`].

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub iter: usize,
    pub cluster_iters: usize,
    pub kmeans_cost: f64,
    pub kl_cost: f64,
    pub ridge_cost: f64,
    pub objective: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HarmonyHistory {
    pub entries: Vec<HistoryEntry>,
}

impl HarmonyHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, e: HistoryEntry) {
        self.entries.push(e)
    }

    pub fn last(&self) -> Option<&HistoryEntry> {
        self.entries.last()
    }

    /// Relative single-step change in `objective` from the last two entries.
    /// Returns `f64::INFINITY` if fewer than two entries.
    pub fn last_rel_change(&self) -> f64 {
        let n = self.entries.len();
        if n < 2 {
            return f64::INFINITY;
        }
        let prev = self.entries[n - 2].objective;
        let curr = self.entries[n - 1].objective;
        if prev.abs() < 1e-300 {
            return f64::INFINITY;
        }
        ((prev - curr) / prev.abs()).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_change_basic() {
        let mut h = HarmonyHistory::new();
        h.push(HistoryEntry {
            iter: 0,
            cluster_iters: 1,
            kmeans_cost: 0.0,
            kl_cost: 0.0,
            ridge_cost: 0.0,
            objective: 100.0,
            elapsed_ms: 0,
        });
        h.push(HistoryEntry {
            iter: 1,
            cluster_iters: 1,
            kmeans_cost: 0.0,
            kl_cost: 0.0,
            ridge_cost: 0.0,
            objective: 99.0,
            elapsed_ms: 0,
        });
        assert!((h.last_rel_change() - 0.01).abs() < 1e-12);
    }
}
