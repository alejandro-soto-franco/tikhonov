//! CLI roundtrip: build a tiny h5ad, invoke `tikhonov`, verify corrected
//! obsm is written and the Patikas JSON appears.

use assert_cmd::prelude::*;
use hdf5_metno as hdf5;
use ndarray::{Array1, Array2};
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn build_synthetic_h5ad(path: &Path) {
    let file = hdf5::File::create(path).unwrap();
    let obsm = file.create_group("obsm").unwrap();
    // 6 cells, 3 PCs; batch-effect on PC0.
    let mut z = Array2::<f64>::zeros((6, 3));
    for i in 0..3 {
        z[[i, 0]] = 1.0 + 0.01 * i as f64;
        z[[i, 1]] = 0.5;
        z[[i, 2]] = 0.0;
    }
    for i in 3..6 {
        z[[i, 0]] = -1.0 + 0.01 * (i - 3) as f64;
        z[[i, 1]] = -0.5;
        z[[i, 2]] = 0.0;
    }
    obsm.new_dataset::<f64>()
        .shape(z.dim())
        .create("X_pca")
        .unwrap()
        .write(&z.view())
        .unwrap();

    let obs = file.create_group("obs").unwrap();
    let sample = obs.create_group("sample").unwrap();
    let codes: Array1<i8> = Array1::from(vec![0, 0, 0, 1, 1, 1]);
    sample
        .new_dataset::<i8>()
        .shape(codes.dim())
        .create("codes")
        .unwrap()
        .write(&codes.view())
        .unwrap();
    use hdf5::types::VarLenUnicode;
    let cats: Array1<VarLenUnicode> =
        Array1::from(vec!["a".parse().unwrap(), "b".parse().unwrap()]);
    sample
        .new_dataset::<VarLenUnicode>()
        .shape(cats.dim())
        .create("categories")
        .unwrap()
        .write(&cats.view())
        .unwrap();
}

#[test]
fn cli_roundtrip_produces_obsm_and_json() {
    let tmp = TempDir::new().unwrap();
    let h5ad_path = tmp.path().join("toy.h5ad");
    let bench_path = tmp.path().join("bench.json");
    build_synthetic_h5ad(&h5ad_path);

    let mut cmd = Command::cargo_bin("tikhonov").unwrap();
    cmd.args([
        "--dataset",
        "toy",
        "--num-pcs",
        "3",
        "--iterations",
        "5",
        "--num-clusters",
        "2",
        "--covariate-factors",
        "sample",
        "--sigma",
        "0.1",
        "--theta",
        "2.0",
        "--magic",
        "6B",
        "--ncpus",
        "1",
        "--seed",
        "0",
        "--input",
    ])
    .arg(&h5ad_path)
    .arg("--bench-json")
    .arg(&bench_path);
    cmd.assert().success();

    // Verify obsm/X_harmony was written.
    let file = hdf5::File::open(&h5ad_path).unwrap();
    let obsm = file.group("obsm").unwrap();
    let x_harm = obsm.dataset("X_harmony").unwrap();
    let shape = x_harm.shape();
    assert_eq!(shape, vec![6, 3]);

    // Verify bench JSON parses and has expected keys.
    let text = std::fs::read_to_string(&bench_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["dataset"], "toy");
    assert_eq!(v["magic"], "6B");
    assert_eq!(v["cells_included"], 6);
    assert!(v["elapsed_secs"].as_f64().unwrap() >= 0.0);
    assert!(
        v["harmony_version"]
            .as_str()
            .unwrap()
            .starts_with("tikhonov-")
    );
}
