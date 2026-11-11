//! From an `.h5ad` file to the Hodge decomposition of its velocity flow.
//!
//! Reads `obsm/X_pca` and `obsm/velocity_pca`, keeps the first `n_pcs`
//! columns, zeroes NaN velocity entries, builds the exact `k`-nearest
//! neighbour graph and its flag triangles, and decomposes the edge flow at
//! each requested triangle fill. Fill 0 uses no triangles and fill 1 uses all
//! of them; a partial fill needs the indices chosen by the Python reference,
//! whose NumPy permutation this crate does not reproduce.

use std::path::Path;

use cartan_complex::{Complex, ComplexError, Decomposition, HodgeOptions, decompose, triangles_of};
use thiserror::Error;

use crate::io::{IoError, read_obsm_f64};
use crate::{edge_flow, knn_edges};

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Complex(#[from] ComplexError),
    #[error("fill {fill} is partial and no triangle choice was given for it")]
    NoTriangleChoice { fill: f64 },
    #[error("positions have {x} rows and velocities {v}")]
    RowMismatch { x: usize, v: usize },
    #[error("fixture: {0}")]
    Fixture(String),
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub k: usize,
    pub n_pcs: usize,
    pub fills: Vec<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            k: 15,
            n_pcs: 30,
            fills: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        }
    }
}

/// For each partial fill, the indices into the sorted list of all triangles.
#[derive(Debug, Clone, Default)]
pub struct FillTriangles {
    pub by_fill: Vec<(f64, Vec<u32>)>,
}

impl FillTriangles {
    fn get(&self, fill: f64) -> Option<&[u32]> {
        self.by_fill
            .iter()
            .find(|(f, _)| *f == fill)
            .map(|(_, v)| v.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct FillReport {
    pub fill: f64,
    pub n_triangles: usize,
    pub grad_norm_frac: f64,
    pub curl_norm_frac: f64,
    pub harm_norm_frac: f64,
    pub decomposition: Decomposition,
}

#[derive(Debug, Clone)]
pub struct RunReport {
    pub n_cells: usize,
    pub edges: Vec<[u32; 2]>,
    pub flow: Vec<f64>,
    pub n_triangles_total: usize,
    pub fills: Vec<FillReport>,
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn run(
    path: &Path,
    cfg: &RunConfig,
    fill_triangles: Option<&FillTriangles>,
) -> Result<RunReport, PipelineError> {
    let x_all = read_obsm_f64(path, "X_pca")?;
    let mut v_all = read_obsm_f64(path, "velocity_pca")?;
    if x_all.nrows() != v_all.nrows() {
        return Err(PipelineError::RowMismatch {
            x: x_all.nrows(),
            v: v_all.nrows(),
        });
    }
    v_all.mapv_inplace(|a| if a.is_nan() { 0.0 } else { a });
    let p = cfg.n_pcs.min(x_all.ncols()).min(v_all.ncols());
    let x = x_all.slice(ndarray::s![.., ..p]);
    let v = v_all.slice(ndarray::s![.., ..p]);
    let n = x.nrows();

    let edges = knn_edges(x, cfg.k);
    let flow = edge_flow(x, v, &edges);
    let base = Complex::new(n, edges.clone())?;
    let all_tri = triangles_of(&base);
    let fnorm = norm(&flow);
    let opts = HodgeOptions::default();

    let mut fills = Vec::with_capacity(cfg.fills.len());
    for &fill in &cfg.fills {
        let tri: Vec<[u32; 3]> = if fill == 0.0 {
            Vec::new()
        } else if fill == 1.0 {
            all_tri.clone()
        } else {
            let idx = fill_triangles
                .and_then(|t| t.get(fill))
                .ok_or(PipelineError::NoTriangleChoice { fill })?;
            idx.iter().map(|&i| all_tri[i as usize]).collect()
        };
        let n_triangles = tri.len();
        let complex = base.clone().with_triangles(tri)?;
        let d = decompose(&complex, &flow, &opts)?;
        let frac = |a: &[f64]| if fnorm > 0.0 { norm(a) / fnorm } else { 0.0 };
        fills.push(FillReport {
            fill,
            n_triangles,
            grad_norm_frac: frac(&d.grad),
            curl_norm_frac: frac(&d.curl),
            harm_norm_frac: frac(&d.harm),
            decomposition: d,
        });
    }
    Ok(RunReport {
        n_cells: n,
        edges,
        flow,
        n_triangles_total: all_tri.len(),
        fills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5_metno as hdf5;
    use ndarray::Array2;

    fn write_h5ad(path: &std::path::Path, x: &Array2<f64>, v: &Array2<f64>) {
        let file = hdf5::File::create(path).unwrap();
        let obsm = file.create_group("obsm").unwrap();
        obsm.new_dataset_builder()
            .with_data(x)
            .create("X_pca")
            .unwrap();
        obsm.new_dataset_builder()
            .with_data(v)
            .create("velocity_pca")
            .unwrap();
    }

    #[test]
    fn pipeline_runs_on_a_synthetic_h5ad() {
        let n = 60;
        let x = Array2::from_shape_fn((n, 3), |(i, c)| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            let wobble = 0.02 * ((i * 7 + c * 3) % 5) as f64;
            [t.cos(), t.sin(), wobble][c] + if c < 2 { wobble } else { 0.0 }
        });
        let mut v = Array2::from_shape_fn((n, 3), |(i, c)| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            [-t.sin(), t.cos(), 0.0][c]
        });
        v[[3, 0]] = f64::NAN;
        v[[17, 2]] = f64::NAN;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring.h5ad");
        write_h5ad(&path, &x, &v);

        let cfg = RunConfig {
            k: 5,
            n_pcs: 30,
            fills: vec![0.0],
        };
        let rep = run(&path, &cfg, None).unwrap();
        assert_eq!(rep.n_cells, n);
        assert_eq!(rep.fills.len(), 1);
        let f0 = &rep.fills[0];
        assert!(f0.harm_norm_frac > 0.5, "harm {}", f0.harm_norm_frac);
        assert!(rep.flow.iter().all(|x| x.is_finite()));
        let d = &f0.decomposition;
        assert!(
            d.grad
                .iter()
                .chain(&d.curl)
                .chain(&d.harm)
                .all(|x| x.is_finite())
        );
    }

    #[test]
    fn a_partial_fill_needs_a_triangle_choice() {
        let x = Array2::from_shape_fn((8, 2), |(i, c)| (i * (c + 1)) as f64);
        let v = Array2::from_elem((8, 2), 1.0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("line.h5ad");
        write_h5ad(&path, &x, &v);
        let cfg = RunConfig {
            k: 3,
            n_pcs: 2,
            fills: vec![0.5],
        };
        assert!(matches!(
            run(&path, &cfg, None),
            Err(PipelineError::NoTriangleChoice { .. })
        ));
    }
}
