//! tikhonov CLI: runs Harmony2 on an `.h5ad`, writes corrected embedding
//! back, and emits Patikas-schema benchmark JSON.

pub mod bench_json;
pub mod h5ad;

use bench_json::{BenchRecord, peak_rss_mib, write_bench};
use clap::Parser;
use ndarray::{Array2, ArrayView2};
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use tikhonov::{HarmonyConfig, run_harmony};

/// Harmony2 integration (Rust port). CLI flag parity with harmony-R 1.2.4's
/// `RunHarmony.R`, with one addition (`--input`) for the explicit h5ad path.
#[derive(Debug, Parser)]
#[command(
    name = "tikhonov",
    version,
    about = "Harmony2 single-cell integration (pure Rust)"
)]
struct Args {
    /// Dataset slug (emitted in benchmark JSON).
    #[arg(short = 'd', long)]
    dataset: String,

    /// Replicate index (emitted in benchmark JSON).
    #[arg(short = 'r', long, default_value = "1")]
    replicate: String,

    /// Number of PCs used.
    #[arg(short = 'p', long)]
    num_pcs: usize,

    /// Outer harmony iterations.
    #[arg(short = 'i', long, default_value_t = 10)]
    iterations: usize,

    /// Number of soft clusters; 0 = auto.
    #[arg(short = 'k', long, default_value_t = 0)]
    num_clusters: usize,

    /// Comma-separated covariate column names in `obs`.
    #[arg(short = 'f', long)]
    covariate_factors: String,

    /// Soft-clustering temperature.
    #[arg(short = 's', long, default_value_t = 0.1)]
    sigma: f64,

    /// Per-covariate diversity penalty (comma-separated).
    #[arg(short = 't', long, default_value = "2.0")]
    theta: String,

    /// Per-batch ridge penalty (comma-separated). Empty string = harmony-R defaults.
    #[arg(short = 'l', long, default_value = "")]
    lambda: String,

    /// Patikas magic tag (e.g. `2M`, `800B`). Passthrough for JSON output.
    #[arg(short = 'm', long, default_value = "")]
    magic: String,

    /// Rayon thread pool size.
    #[arg(short = 'n', long, default_value_t = num_cpus())]
    ncpus: usize,

    /// Version identifier emitted in benchmark JSON.
    #[arg(short = 'c', long, default_value_t = default_harmony_version())]
    code: String,

    /// Output directory for the Patikas-schema JSON.
    #[arg(short = 'o', long, default_value = "outs/")]
    output_dir: PathBuf,

    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Input `.h5ad` file.
    #[arg(long)]
    input: PathBuf,

    /// Key in `obsm` for the PC embedding.
    #[arg(long, default_value = "X_pca")]
    pc_key: String,

    /// Output key in `obsm` for the corrected embedding.
    #[arg(long, default_value = "X_harmony")]
    output_key: String,

    /// If set, write the benchmark JSON here instead of `output_dir/<md5>.txt`.
    #[arg(long)]
    bench_json: Option<PathBuf>,

    /// If set, write the convergence history as JSON here.
    #[arg(long)]
    emit_history: Option<PathBuf>,

    /// Verbose progress.
    #[arg(long)]
    verbose: bool,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn default_harmony_version() -> String {
    format!("tikhonov-{}", env!("CARGO_PKG_VERSION"))
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|e| e.into()))
        .collect()
}

/// Transpose an AnnData-style `(n, d)` matrix to harmony's internal `(d, n)` column-major layout.
fn to_dxn(nxd: ArrayView2<'_, f64>) -> Array2<f64> {
    nxd.t().to_owned()
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let covariates: Vec<&str> = args.covariate_factors.split(',').map(str::trim).collect();

    let z_nxd = h5ad::read_obsm_f64(&args.input, &args.pc_key)?;
    let n_cells = z_nxd.nrows();
    let d = z_nxd.ncols();
    if d < args.num_pcs {
        return Err(format!(
            "obsm/{} has {} cols, --num-pcs is {}",
            args.pc_key, d, args.num_pcs
        )
        .into());
    }
    let z = to_dxn(z_nxd.view());

    // Build labels matrix (n, n_cov).
    let mut labels = ndarray::Array2::<u32>::zeros((n_cells, covariates.len()));
    let mut level_names: Vec<Vec<String>> = Vec::with_capacity(covariates.len());
    let mut total_levels = 0usize;
    for (c, name) in covariates.iter().enumerate() {
        let (cats, codes) = h5ad::read_obs_categorical(&args.input, name)?;
        if codes.len() != n_cells {
            return Err(format!(
                "obs/{} has {} entries; expected {}",
                name,
                codes.len(),
                n_cells
            )
            .into());
        }
        for (i, &code) in codes.iter().enumerate() {
            labels[[i, c]] = code;
        }
        total_levels += cats.len();
        level_names.push(cats);
    }

    // Build config.
    let theta_vec = parse_f64_list(&args.theta)?;
    let lambda_vec = parse_f64_list(&args.lambda)?;

    let mut cfg = HarmonyConfig::new()
        .with_max_iter(args.iterations)
        .with_sigma(args.sigma)
        .with_theta(theta_vec.clone())
        .with_seed(args.seed)
        .with_verbose(args.verbose)
        .with_n_threads(args.ncpus);
    if args.num_clusters > 0 {
        cfg = cfg.with_nclust(args.num_clusters);
    }
    if !lambda_vec.is_empty() {
        cfg = cfg.with_lambda(lambda_vec);
    }

    let t_start = Instant::now();
    let out = run_harmony(z.view(), labels.view(), &cfg)?;
    let elapsed = t_start.elapsed().as_secs_f64();

    // Transpose back to (n, d) before writing.
    let z_corr_nxd = out.z_corr.t().to_owned();
    h5ad::write_obsm_f64(&args.input, &args.output_key, z_corr_nxd.view())?;

    // History.
    if let Some(hist_path) = &args.emit_history {
        let s = serde_json::to_string_pretty(&out.history)?;
        std::fs::write(hist_path, s)?;
    }

    // Bench JSON.
    let unique_batches: std::collections::BTreeSet<u32> = (0..covariates.len())
        .flat_map(|c| labels.column(c).iter().copied().collect::<Vec<_>>())
        .collect();
    let record = BenchRecord {
        covariate_factors: args.covariate_factors.clone(),
        dataset: args.dataset.clone(),
        harmony_clusters: cfg.resolved_nclust(n_cells),
        harmony_version: args.code.clone(),
        iterations: args.iterations.to_string(),
        magic: args.magic.clone(),
        ncpus: args.ncpus.to_string(),
        number_of_pcs: args.num_pcs,
        output_dir: args.output_dir.display().to_string(),
        replicate: args.replicate.clone(),
        sigma: args.sigma,
        theta: if theta_vec.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::json!(theta_vec)
        },
        batches_included: unique_batches.len().min(total_levels),
        covariates: covariates.len(),
        cells_included: n_cells,
        peak_memory: peak_rss_mib(),
        elapsed_secs: elapsed,
    };

    match &args.bench_json {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, serde_json::to_string_pretty(&record)?)?;
        }
        None => {
            write_bench(&args.output_dir, &record)?;
        }
    }

    if args.verbose {
        eprintln!(
            "tikhonov done: {} iters, converged={}, elapsed={:.2}s",
            out.n_iter, out.converged, elapsed
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tikhonov: {e}");
            ExitCode::FAILURE
        }
    }
}
