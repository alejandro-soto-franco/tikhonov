//! Hodge decomposition of an `.h5ad` velocity field.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use tikhonov_velocity::fixture;
use tikhonov_velocity::pipeline::{RunConfig, run};

#[derive(Parser)]
#[command(about = "Hodge decomposition of an .h5ad velocity field")]
struct Cli {
    /// Input `.h5ad` with `obsm/X_pca` and `obsm/velocity_pca`.
    h5ad: PathBuf,
    #[arg(long, default_value_t = 15)]
    k: usize,
    #[arg(long, default_value_t = 30)]
    n_pcs: usize,
    /// Reference fixture directory; supplies the partial-fill triangles.
    #[arg(long)]
    fixture: Option<PathBuf>,
    /// Write the per-fill fractions as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let fx = match cli.fixture.as_deref().map(fixture::load).transpose() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let fills = if fx.is_some() {
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    } else {
        vec![0.0, 1.0]
    };
    let cfg = RunConfig {
        k: cli.k,
        n_pcs: cli.n_pcs,
        fills,
    };
    let t0 = Instant::now();
    let rep = match run(&cli.h5ad, &cfg, fx.as_ref().map(|f| &f.triangles)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{} cells, {} edges, {} triangles, {:.2} s",
        rep.n_cells,
        rep.edges.len(),
        rep.n_triangles_total,
        secs
    );
    let mut rows = Vec::new();
    for f in &rep.fills {
        println!(
            "fill={:4.2} tri={:6} grad={:.6} curl={:.6} harm={:.6} cg_iter={} cg_res={:.1e}",
            f.fill,
            f.n_triangles,
            f.grad_norm_frac,
            f.curl_norm_frac,
            f.harm_norm_frac,
            f.decomposition.curl_iterations,
            f.decomposition.curl_residual
        );
        rows.push(serde_json::json!({
            "fill": f.fill, "n_triangles": f.n_triangles,
            "grad_norm_frac": f.grad_norm_frac, "curl_norm_frac": f.curl_norm_frac,
            "harm_norm_frac": f.harm_norm_frac,
        }));
    }
    if let Some(p) = cli.json {
        let doc = serde_json::json!({
            "n_cells": rep.n_cells, "n_edges": rep.edges.len(),
            "n_triangles": rep.n_triangles_total, "seconds": secs, "fills": rows,
        });
        if let Err(e) = std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()) {
            eprintln!("error: {}: {e}", p.display());
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
