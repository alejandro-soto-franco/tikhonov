//! Parity with a tight Python reference on the scVelo pancreas data at k = 15.
//!
//! Ignored by default: it needs the pancreas `.h5ad` with `obsm/X_pca` and
//! `obsm/velocity_pca`, and a reference fixture directory. Point
//! `TIKHONOV_PARITY_H5AD` and `TIKHONOV_PARITY_FIXTURE` at them and run
//! `cargo test -p tikhonov-velocity --release --test parity_pancreas -- --ignored`.
//! With either variable unset the test returns without checking anything.

use std::collections::BTreeSet;
use std::path::PathBuf;

use tikhonov_velocity::fixture;
use tikhonov_velocity::pipeline::{RunConfig, run};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    let n: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    d / n.max(f64::MIN_POSITIVE)
}

fn six(a: f64, b: f64) -> bool {
    (a - b).abs() < 5e-7
}

#[test]
#[ignore]
fn pancreas_matches_the_tight_reference() {
    let (Some(h5ad), Some(dir)) = (
        env_path("TIKHONOV_PARITY_H5AD"),
        env_path("TIKHONOV_PARITY_FIXTURE"),
    ) else {
        eprintln!("TIKHONOV_PARITY_H5AD or TIKHONOV_PARITY_FIXTURE unset; skipping");
        return;
    };
    let fx = fixture::load(&dir).expect("fixture loads");
    let cfg = RunConfig {
        k: fx.meta.k,
        n_pcs: fx.meta.n_pcs,
        ..RunConfig::default()
    };
    let rep = run(&h5ad, &cfg, Some(&fx.triangles)).expect("pipeline runs");

    if rep.edges != fx.edges {
        let a: BTreeSet<_> = rep.edges.iter().collect();
        let b: BTreeSet<_> = fx.edges.iter().collect();
        let only_rust: Vec<_> = a.difference(&b).take(5).collect();
        let only_ref: Vec<_> = b.difference(&a).take(5).collect();
        panic!(
            "edge sets differ: {} only in Rust, {} only in the reference; first {:?} / {:?}",
            a.difference(&b).count(),
            b.difference(&a).count(),
            only_rust,
            only_ref
        );
    }
    assert_eq!(rep.n_triangles_total, fx.meta.n_triangles);
    let fr = rel(&rep.flow, &fx.flow);
    assert!(fr <= 1e-12, "flow differs by {fr:e}");

    let by = |fill: f64| rep.fills.iter().find(|f| f.fill == fill).unwrap();
    for (fill, grad, curl, harm) in [
        (0.0, &fx.grad_fill0, &fx.curl_fill0, &fx.harm_fill0),
        (1.0, &fx.grad_fill1, &fx.curl_fill1, &fx.harm_fill1),
    ] {
        let d = &by(fill).decomposition;
        let (g, h) = (rel(&d.grad, grad), rel(&d.harm, harm));
        assert!(g <= 1e-10, "fill {fill}: grad differs by {g:e}");
        assert!(h <= 1e-10, "fill {fill}: harm differs by {h:e}");
        if fill == 1.0 {
            let c = rel(&d.curl, curl);
            assert!(c <= 1e-10, "fill {fill}: curl differs by {c:e}");
        } else {
            assert!(d.curl.iter().all(|&v| v == 0.0));
        }
    }

    // Fractions reported by the Python implementation the fixture reproduces.
    let expected = [
        (0.0, 0.0, 0.487992),
        (0.25, 0.203618, 0.443482),
        (0.5, 0.354553, 0.335304),
        (0.75, 0.435710, 0.219757),
        (1.0, 0.451820, 0.184378),
    ];
    for (fill, curl, harm) in expected {
        let f = by(fill);
        assert!(
            six(f.grad_norm_frac, 0.872848),
            "fill {fill}: grad {}",
            f.grad_norm_frac
        );
        assert!(
            six(f.curl_norm_frac, curl),
            "fill {fill}: curl {}",
            f.curl_norm_frac
        );
        assert!(
            six(f.harm_norm_frac, harm),
            "fill {fill}: harm {}",
            f.harm_norm_frac
        );
    }
}
