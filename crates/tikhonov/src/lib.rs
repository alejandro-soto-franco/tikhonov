//! Tikhonov: pure-Rust Harmony2 single-cell integration.
#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod phi;

pub use config::HarmonyConfig;
pub use error::HarmonyError;
pub use phi::Phi;
