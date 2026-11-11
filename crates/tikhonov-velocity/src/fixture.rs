//! Reader for the tight Python reference fixture.
//!
//! The directory contains `meta.json` and raw little-endian arrays: `edges.u32`
//! (pairs), `flow.f64`, `grad_fill{0,1}.f64`, `curl_fill{0,1}.f64`,
//! `harm_fill{0,1}.f64`, and `tri_fill{025,050,075}.u32`, the indices into the
//! sorted triangle list chosen for each partial fill.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::pipeline::{FillTriangles, PipelineError};

#[derive(Debug, Clone, Deserialize)]
pub struct FillMeta {
    pub fill: f64,
    pub n_triangles: usize,
    pub grad_norm_frac: f64,
    pub curl_norm_frac: f64,
    pub harm_norm_frac: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub n_cells: usize,
    pub k: usize,
    pub n_pcs: usize,
    pub n_edges: usize,
    pub n_triangles: usize,
    pub fills: Vec<FillMeta>,
}

/// The reference vectors for fills 0 and 1, and the triangle choices for the
/// partial fills.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub meta: Meta,
    pub edges: Vec<[u32; 2]>,
    pub flow: Vec<f64>,
    pub grad_fill0: Vec<f64>,
    pub curl_fill0: Vec<f64>,
    pub harm_fill0: Vec<f64>,
    pub grad_fill1: Vec<f64>,
    pub curl_fill1: Vec<f64>,
    pub harm_fill1: Vec<f64>,
    pub triangles: FillTriangles,
}

fn read_f64(path: &Path) -> Result<Vec<f64>, PipelineError> {
    let b =
        fs::read(path).map_err(|e| PipelineError::Fixture(format!("{}: {e}", path.display())))?;
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, PipelineError> {
    let b =
        fs::read(path).map_err(|e| PipelineError::Fixture(format!("{}: {e}", path.display())))?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

pub fn load(dir: &Path) -> Result<Fixture, PipelineError> {
    let meta_text = fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| PipelineError::Fixture(format!("meta.json: {e}")))?;
    let meta: Meta = serde_json::from_str(&meta_text)
        .map_err(|e| PipelineError::Fixture(format!("meta.json: {e}")))?;
    let edges = read_u32(&dir.join("edges.u32"))?
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect();
    let mut by_fill = Vec::new();
    for (fill, name) in [
        (0.25, "tri_fill025.u32"),
        (0.5, "tri_fill050.u32"),
        (0.75, "tri_fill075.u32"),
    ] {
        by_fill.push((fill, read_u32(&dir.join(name))?));
    }
    Ok(Fixture {
        edges,
        flow: read_f64(&dir.join("flow.f64"))?,
        grad_fill0: read_f64(&dir.join("grad_fill0.f64"))?,
        curl_fill0: read_f64(&dir.join("curl_fill0.f64"))?,
        harm_fill0: read_f64(&dir.join("harm_fill0.f64"))?,
        grad_fill1: read_f64(&dir.join("grad_fill1.f64"))?,
        curl_fill1: read_f64(&dir.join("curl_fill1.f64"))?,
        harm_fill1: read_f64(&dir.join("harm_fill1.f64"))?,
        triangles: FillTriangles { by_fill },
        meta,
    })
}
