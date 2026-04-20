//! Patikas-schema benchmark JSON emitter.
//!
//! Key order and types mirror
//! `~/tikhonov-research/harmony2-ms/tahoe/benchmark/results/*.txt` exactly so
//! that our results drop into their comparison notebook without transformation.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchRecord {
    pub covariate_factors: String,
    pub dataset: String,
    pub harmony_clusters: usize,
    pub harmony_version: String,
    pub iterations: String,
    pub magic: String,
    pub ncpus: String,
    pub number_of_pcs: usize,
    pub output_dir: String,
    pub replicate: String,
    pub sigma: f64,
    pub theta: serde_json::Value,
    pub batches_included: usize,
    pub covariates: usize,
    pub cells_included: usize,
    pub peak_memory: f64,
    pub elapsed_secs: f64,
}

/// Peak resident set size in MiB, sampled from `/proc/self/status` on Linux.
///
/// Returns `0.0` on other platforms.
pub fn peak_rss_mib() -> f64 {
    #[cfg(target_os = "linux")]
    {
        let status = match fs::read_to_string("/proc/self/status") {
            Ok(s) => s,
            Err(_) => return 0.0,
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kib: f64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0.0);
                return kib / 1024.0;
            }
        }
        0.0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0.0
    }
}

/// MD5 hex of a `BenchRecord`'s JSON representation. Used as the Patikas
/// output filename stem (e.g. `0525fae61a2917ecc6fe7bcf5d730425.txt`).
pub fn md5_filename(record: &BenchRecord) -> String {
    let s = serde_json::to_string(record).expect("bench record must serialise");
    format!("{:x}", md5::compute(s.as_bytes()))
}

/// Write a record to `dir/<md5>.txt` (matching Patikas convention).
pub fn write_bench(dir: &Path, record: &BenchRecord) -> std::io::Result<std::path::PathBuf> {
    fs::create_dir_all(dir)?;
    let name = md5_filename(record);
    let path = dir.join(format!("{name}.txt"));
    let pretty = serde_json::to_string_pretty(record).expect("bench record must serialise");
    fs::write(&path, pretty)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> BenchRecord {
        BenchRecord {
            covariate_factors: "sample".into(),
            dataset: "tahoe-profile".into(),
            harmony_clusters: 100,
            harmony_version: "tikhonov-0.1.0".into(),
            iterations: "10".into(),
            magic: "2M".into(),
            ncpus: "16".into(),
            number_of_pcs: 20,
            output_dir: "outs/".into(),
            replicate: "3".into(),
            sigma: 0.1,
            theta: serde_json::json!({}),
            batches_included: 800,
            covariates: 1,
            cells_included: 2_000_000,
            peak_memory: 76333.4961,
            elapsed_secs: 9511.165,
        }
    }

    #[test]
    fn record_serialises_with_patikas_key_order() {
        let r = sample_record();
        let s = serde_json::to_string(&r).unwrap();
        // serde preserves struct field order.
        let expected_prefix = r#"{"covariate_factors":"sample","dataset":"tahoe-profile","harmony_clusters":100,"harmony_version":"tikhonov-0.1.0","iterations":"10","magic":"2M","ncpus":"16","number_of_pcs":20"#;
        assert!(s.starts_with(expected_prefix), "got: {s}");
    }

    #[test]
    fn record_roundtrips() {
        let r = sample_record();
        let s = serde_json::to_string(&r).unwrap();
        let back: BenchRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn md5_filename_is_stable() {
        let r = sample_record();
        let a = md5_filename(&r);
        let b = md5_filename(&r);
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn write_bench_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_bench(dir.path(), &sample_record()).unwrap();
        assert!(path.exists());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("tahoe-profile"));
    }
}
