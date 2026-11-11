//! `.h5ad` reading for the pipeline.
//!
//! A copy of `read_obsm_f64` from `tikhonov-cli`'s shim. It is copied rather
//! than shared because the published `tikhonov-cli` cannot depend on this
//! unpublished crate; a later version replaces both with the `anndata` crate.

use std::path::Path;

use hdf5_metno as hdf5;
use ndarray::Array2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("hdf5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("unsupported encoding in {path}: {reason}")]
    Unsupported { path: String, reason: String },
    #[error("path not found in h5ad: {0}")]
    NotFound(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

/// Reads `obsm/<key>` as a dense `f64` matrix of shape `(n_obs, d)`,
/// widening `f32` storage.
pub fn read_obsm_f64(path: &Path, key: &str) -> Result<Array2<f64>, IoError> {
    let file = hdf5::File::open(path)?;
    let obsm = file
        .group("obsm")
        .map_err(|_| IoError::NotFound("obsm".into()))?;
    let ds = obsm
        .dataset(key)
        .map_err(|_| IoError::NotFound(format!("obsm/{key}")))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(IoError::Shape(format!(
            "obsm/{key} has shape {shape:?}; expected 2D"
        )));
    }
    let dtype = ds.dtype()?;
    if dtype.is::<f64>() {
        Ok(ds.read_2d::<f64>()?)
    } else if dtype.is::<f32>() {
        Ok(ds.read_2d::<f32>()?.mapv(f64::from))
    } else {
        Err(IoError::Unsupported {
            path: format!("obsm/{key}"),
            reason: format!("dtype {dtype:?}, expected f32 or f64"),
        })
    }
}
