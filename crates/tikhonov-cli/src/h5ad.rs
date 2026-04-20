//! Minimal AnnData `.h5ad` shim (obsm + obs categorical only).
//!
//! Covers exactly what Harmony2 needs from an AnnData file:
//!
//! - [`read_obsm_f64`]: read a `(n_obs, d)` dense float matrix from `obsm/<key>`.
//! - [`read_obs_categorical`]: read a categorical column from `obs/<key>`.
//! - [`write_obsm_f64`]: write a dense float matrix back into `obsm/<key>`.
//!
//! Does NOT support `X`, `layers`, `varm`, `obsp`, `varp`, `uns`, `.raw`, or
//! `.zarr`. Fails loudly on any encoding we don't recognise.

use hdf5_metno as hdf5;
use ndarray::{Array1, Array2, ArrayView2};
use std::path::Path;
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

/// Read `obsm/<key>` as a dense `f64` matrix of shape `(n_obs, d)`.
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
    // hdf5-metno reads into ndarray; dtype detection through the reader.
    let dtype = ds.dtype()?;
    if dtype.is::<f64>() {
        Ok(ds.read_2d::<f64>()?)
    } else if dtype.is::<f32>() {
        let f32_data = ds.read_2d::<f32>()?;
        Ok(f32_data.mapv(|v| v as f64))
    } else {
        Err(IoError::Unsupported {
            path: format!("obsm/{key}"),
            reason: format!("dtype {dtype:?}, expected f32 or f64"),
        })
    }
}

/// Read `obs/<key>` as a categorical: returns `(categories, codes_u32)`.
///
/// Supports two AnnData encodings:
/// 1. `encoding-type = "categorical"` group with `codes` + `categories` datasets.
/// 2. Plain string dataset (scan + dedup).
pub fn read_obs_categorical(path: &Path, key: &str) -> Result<(Vec<String>, Array1<u32>), IoError> {
    let file = hdf5::File::open(path)?;
    let obs = file
        .group("obs")
        .map_err(|_| IoError::NotFound("obs".into()))?;

    // Try group-with-codes first.
    if let Ok(grp) = obs.group(key) {
        let categories_ds = grp
            .dataset("categories")
            .map_err(|_| IoError::Unsupported {
                path: format!("obs/{key}"),
                reason: "group lacks `categories` dataset".into(),
            })?;
        let codes_ds = grp.dataset("codes").map_err(|_| IoError::Unsupported {
            path: format!("obs/{key}"),
            reason: "group lacks `codes` dataset".into(),
        })?;
        let cats: Vec<String> = categories_ds
            .read_1d::<hdf5::types::VarLenUnicode>()?
            .iter()
            .map(|s| s.as_str().to_owned())
            .collect();
        let code_dtype = codes_ds.dtype()?;
        let codes: Array1<u32> = if code_dtype.is::<i8>() {
            codes_ds.read_1d::<i8>()?.mapv(|v| v.max(0) as u32)
        } else if code_dtype.is::<i16>() {
            codes_ds.read_1d::<i16>()?.mapv(|v| v.max(0) as u32)
        } else if code_dtype.is::<i32>() {
            codes_ds.read_1d::<i32>()?.mapv(|v| v.max(0) as u32)
        } else if code_dtype.is::<i64>() {
            codes_ds.read_1d::<i64>()?.mapv(|v| v.max(0) as u32)
        } else {
            return Err(IoError::Unsupported {
                path: format!("obs/{key}/codes"),
                reason: format!("dtype {code_dtype:?}, expected i8/i16/i32/i64"),
            });
        };
        return Ok((cats, codes));
    }

    // Fall back to flat string dataset.
    if let Ok(ds) = obs.dataset(key) {
        let raw: Vec<String> = ds
            .read_1d::<hdf5::types::VarLenUnicode>()?
            .iter()
            .map(|s| s.as_str().to_owned())
            .collect();
        let mut cats: Vec<String> = Vec::new();
        let mut codes: Vec<u32> = Vec::with_capacity(raw.len());
        for s in &raw {
            if let Some(pos) = cats.iter().position(|c| c == s) {
                codes.push(pos as u32);
            } else {
                codes.push(cats.len() as u32);
                cats.push(s.clone());
            }
        }
        return Ok((cats, Array1::from(codes)));
    }

    Err(IoError::NotFound(format!("obs/{key}")))
}

/// Write or overwrite `obsm/<key>` as a dense `f64` matrix.
///
/// Adds the AnnData-standard `encoding-type = "array"` and
/// `encoding-version = "0.2.0"` attributes on the dataset.
pub fn write_obsm_f64(path: &Path, key: &str, arr: ArrayView2<'_, f64>) -> Result<(), IoError> {
    let file = hdf5::File::open_rw(path)?;
    let obsm = match file.group("obsm") {
        Ok(g) => g,
        Err(_) => file.create_group("obsm")?,
    };
    // Remove existing if present.
    if obsm.link_exists(key) {
        obsm.unlink(key)?;
    }
    let ds = obsm.new_dataset::<f64>().shape(arr.dim()).create(key)?;
    ds.write(&arr)?;
    let attr_type = ds
        .new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create("encoding-type")?;
    let attr_ver = ds
        .new_attr::<hdf5::types::VarLenUnicode>()
        .shape(())
        .create("encoding-version")?;
    attr_type.write_scalar(&"array".parse::<hdf5::types::VarLenUnicode>().unwrap())?;
    attr_ver.write_scalar(&"0.2.0".parse::<hdf5::types::VarLenUnicode>().unwrap())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::NamedTempFile;

    fn build_tiny_h5ad(path: &Path) {
        let file = hdf5::File::create(path).unwrap();
        let obsm = file.create_group("obsm").unwrap();
        let arr: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let ds = obsm
            .new_dataset::<f64>()
            .shape(arr.dim())
            .create("X_pca")
            .unwrap();
        ds.write(&arr.view()).unwrap();

        let obs = file.create_group("obs").unwrap();
        let sample = obs.create_group("sample").unwrap();
        let codes: Array1<i8> = Array1::from(vec![0, 1, 0]);
        sample
            .new_dataset::<i8>()
            .shape(codes.dim())
            .create("codes")
            .unwrap()
            .write(&codes.view())
            .unwrap();
        use hdf5::types::VarLenUnicode;
        let cats: Vec<VarLenUnicode> = vec!["a".parse().unwrap(), "b".parse().unwrap()];
        let cats_arr = Array1::from(cats);
        sample
            .new_dataset::<VarLenUnicode>()
            .shape(cats_arr.dim())
            .create("categories")
            .unwrap()
            .write(&cats_arr.view())
            .unwrap();
    }

    #[test]
    fn roundtrip_obsm_and_obs() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        build_tiny_h5ad(path);

        let z = read_obsm_f64(path, "X_pca").unwrap();
        assert_eq!(z.dim(), (2, 3));
        assert_eq!(z[[0, 0]], 1.0);
        assert_eq!(z[[1, 2]], 6.0);

        let (cats, codes) = read_obs_categorical(path, "sample").unwrap();
        assert_eq!(cats, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(codes.to_vec(), vec![0, 1, 0]);

        let new: Array2<f64> = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        write_obsm_f64(path, "X_harmony", new.view()).unwrap();
        let back = read_obsm_f64(path, "X_harmony").unwrap();
        assert_eq!(back, new);
    }

    #[test]
    fn f32_upcasts_to_f64() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        {
            let file = hdf5::File::create(path).unwrap();
            let obsm = file.create_group("obsm").unwrap();
            let arr: Array2<f32> = array![[1.5_f32, 2.5], [3.5, 4.5]];
            obsm.new_dataset::<f32>()
                .shape(arr.dim())
                .create("X_pca")
                .unwrap()
                .write(&arr.view())
                .unwrap();
        }
        let z = read_obsm_f64(path, "X_pca").unwrap();
        assert_eq!(z[[0, 0]], 1.5);
    }

    #[test]
    fn missing_obsm_errors() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        let _ = hdf5::File::create(path).unwrap();
        let err = read_obsm_f64(path, "X_pca").unwrap_err();
        assert!(matches!(err, IoError::NotFound(_)));
    }
}
