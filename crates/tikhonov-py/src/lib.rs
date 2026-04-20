//! PyO3 bindings for the `tikhonov` crate.
//!
//! Exposes `tikhonov._core.run_harmony(z, labels, **kwargs)` returning a
//! `PyRunHarmonyResult` with `.Z_corr`, `.Y`, `.R`, `.history`, `.converged`,
//! and `.n_iter` attributes. The Python-level AnnData adapter lives in
//! `python/tikhonov/anndata.py`; the Rust FFI sees only numpy arrays.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tikhonov::{HarmonyConfig, run_harmony};

#[pyclass(name = "HarmonyResult", module = "tikhonov._core")]
struct PyRunHarmonyResult {
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    n_iter: usize,
    z_corr: Array2<f64>,
    y: Array2<f64>,
    r: Array2<f64>,
    history_entries: Vec<tikhonov::HistoryEntry>,
}

#[pymethods]
impl PyRunHarmonyResult {
    /// Corrected embedding as `(n_cells, d)` (scanpy / harmonypy convention).
    #[getter(Z_corr)]
    fn z_corr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.z_corr.t().to_owned().into_pyarray(py)
    }

    /// Cluster centroids as `(K, d)`.
    #[getter(Y)]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.y.t().to_owned().into_pyarray(py)
    }

    /// Soft-assignment matrix as `(K, n_cells)`.
    #[getter(R)]
    fn r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.r.clone().into_pyarray(py)
    }

    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for e in &self.history_entries {
            let d = PyDict::new(py);
            d.set_item("iter", e.iter)?;
            d.set_item("cluster_iters", e.cluster_iters)?;
            d.set_item("kmeans_cost", e.kmeans_cost)?;
            d.set_item("kl_cost", e.kl_cost)?;
            d.set_item("ridge_cost", e.ridge_cost)?;
            d.set_item("objective", e.objective)?;
            d.set_item("elapsed_ms", e.elapsed_ms)?;
            list.append(d)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "HarmonyResult(Z_corr={:?}, Y={:?}, R={:?}, converged={}, n_iter={})",
            self.z_corr.dim(),
            self.y.dim(),
            self.r.dim(),
            self.converged,
            self.n_iter,
        )
    }
}

fn build_config_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<HarmonyConfig> {
    let mut cfg = HarmonyConfig::new();
    let Some(kwargs) = kwargs else { return Ok(cfg) };

    if let Some(v) = kwargs.get_item("nclust")? {
        if !v.is_none() {
            cfg = cfg.with_nclust(v.extract::<usize>()?);
        }
    }
    if let Some(v) = kwargs.get_item("max_iter")? {
        cfg = cfg.with_max_iter(v.extract::<usize>()?);
    }
    if let Some(v) = kwargs.get_item("max_iter_cluster")? {
        cfg = cfg.with_max_iter_cluster(v.extract::<usize>()?);
    }
    if let Some(v) = kwargs.get_item("sigma")? {
        cfg = cfg.with_sigma(v.extract::<f64>()?);
    }
    if let Some(v) = kwargs.get_item("theta")? {
        let theta: Vec<f64> = if let Ok(scalar) = v.extract::<f64>() {
            vec![scalar]
        } else {
            v.extract::<Vec<f64>>()?
        };
        cfg = cfg.with_theta(theta);
    }
    if let Some(v) = kwargs.get_item("lambda_")? {
        if !v.is_none() {
            let lam: Vec<f64> = if let Ok(scalar) = v.extract::<f64>() {
                vec![scalar]
            } else {
                v.extract::<Vec<f64>>()?
            };
            cfg = cfg.with_lambda(lam);
        }
    }
    if let Some(v) = kwargs.get_item("epsilon_cluster")? {
        cfg = cfg.with_epsilon_cluster(v.extract::<f64>()?);
    }
    if let Some(v) = kwargs.get_item("epsilon_harmony")? {
        cfg = cfg.with_epsilon_harmony(v.extract::<f64>()?);
    }
    if let Some(v) = kwargs.get_item("block_size")? {
        cfg = cfg.with_block_size(v.extract::<f64>()?);
    }
    if let Some(v) = kwargs.get_item("random_state")? {
        cfg = cfg.with_seed(v.extract::<u64>()?);
    }
    if let Some(v) = kwargs.get_item("verbose")? {
        cfg = cfg.with_verbose(v.extract::<bool>()?);
    }
    if let Some(v) = kwargs.get_item("n_threads")? {
        if !v.is_none() {
            cfg = cfg.with_n_threads(v.extract::<usize>()?);
        }
    }
    Ok(cfg)
}

/// Run Harmony2 on a `(d, n)` embedding and `(n, n_cov)` u32 labels.
#[pyfunction(signature = (z, labels, **kwargs))]
fn py_run_harmony(
    z: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray2<'_, u32>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyRunHarmonyResult> {
    let cfg = build_config_from_kwargs(kwargs)?;
    let z_view = z.as_array();
    let labels_view = labels.as_array();

    let result =
        run_harmony(z_view, labels_view, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyRunHarmonyResult {
        converged: result.converged,
        n_iter: result.n_iter,
        z_corr: result.z_corr,
        y: result.y,
        r: result.r,
        history_entries: result.history.entries,
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRunHarmonyResult>()?;
    m.add_function(wrap_pyfunction!(py_run_harmony, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
