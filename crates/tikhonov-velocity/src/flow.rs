//! Velocity edge flow.

use ndarray::ArrayView2;

/// The velocity flow along each edge: the mean of the two endpoint velocities
/// projected on the unit edge direction,
/// `f = <(v_i + v_j) / 2, x_j - x_i> / |x_j - x_i|`, and 0 on an edge of zero
/// length. This is the construction ddHodge and PHLOWER use.
pub fn edge_flow(x: ArrayView2<f64>, v: ArrayView2<f64>, edges: &[[u32; 2]]) -> Vec<f64> {
    assert_eq!(x.dim(), v.dim(), "positions and velocities differ in shape");
    edges
        .iter()
        .map(|&[i, j]| {
            let (xi, xj) = (x.row(i as usize), x.row(j as usize));
            let (vi, vj) = (v.row(i as usize), v.row(j as usize));
            let mut dot = 0.0;
            let mut len2 = 0.0;
            for c in 0..x.ncols() {
                let dc = xj[c] - xi[c];
                dot += 0.5 * (vi[c] + vj[c]) * dc;
                len2 += dc * dc;
            }
            if len2 == 0.0 { 0.0 } else { dot / len2.sqrt() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn flow_on_one_edge() {
        let x = array![[0.0, 0.0], [3.0, 4.0]];
        let v = array![[1.0, 0.0], [1.0, 2.0]];
        let f = edge_flow(x.view(), v.view(), &[[0, 1]]);
        assert!((f[0] - 1.4).abs() < 1e-15);
    }

    #[test]
    fn zero_length_edge_has_zero_flow() {
        let x = array![[1.0, 2.0], [1.0, 2.0]];
        let v = array![[1.0, 0.0], [0.0, 1.0]];
        assert_eq!(edge_flow(x.view(), v.view(), &[[0, 1]]), vec![0.0]);
    }
}
