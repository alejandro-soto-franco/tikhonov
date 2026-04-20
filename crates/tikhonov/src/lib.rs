//! Tikhonov: pure-Rust Harmony2 single-cell integration.
#![forbid(unsafe_code)]

pub mod cluster;
pub mod config;
pub mod correct;
pub mod embed;
pub mod error;
pub mod history;
pub mod objective;
pub mod phi;

pub use config::HarmonyConfig;
pub use error::HarmonyError;
pub use history::{HarmonyHistory, HistoryEntry};
pub use phi::Phi;
