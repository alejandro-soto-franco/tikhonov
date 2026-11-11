//! Exact nearest neighbour edges.
//!
//! Distances are sums of squared coordinate differences, computed directly.
//! The Gram identity `|x|^2 + |y|^2 - 2 x.y` is faster but rounds differently,
//! which reorders exact ties and so changes the edge set relative to a tree
//! search such as scikit-learn's.

use std::cmp::Ordering;

use ndarray::ArrayView2;
use rayon::prelude::*;

fn by_distance_then_index(a: &(f64, u32), b: &(f64, u32)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Edges joining each row of `x` to its `k` nearest other rows.
///
/// Distance ties go to the lower index. The result is the symmetrised union,
/// each edge `[i, j]` with `i < j`, deduplicated and sorted. With fewer than
/// `k + 1` rows every row is joined to all the others.
pub fn knn_edges(x: ArrayView2<f64>, k: usize) -> Vec<[u32; 2]> {
    let n = x.nrows();
    let d = x.ncols();
    let owned = x.as_standard_layout();
    let rows = owned.as_slice().expect("standard layout is contiguous");
    let kk = k.min(n.saturating_sub(1));
    let mut edges: Vec<[u32; 2]> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let xi = &rows[i * d..(i + 1) * d];
            let mut cand: Vec<(f64, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let xj = &rows[j * d..(j + 1) * d];
                    let s: f64 = xi.iter().zip(xj).map(|(a, b)| (a - b) * (a - b)).sum();
                    (s, j as u32)
                })
                .collect();
            if kk < cand.len() && kk > 0 {
                cand.select_nth_unstable_by(kk - 1, by_distance_then_index);
            }
            cand.truncate(kk);
            let i = i as u32;
            cand.into_iter().map(move |(_, j)| [i.min(j), i.max(j)])
        })
        .collect();
    edges.par_sort_unstable();
    edges.dedup();
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn line_of_points() {
        let x = array![[0.0], [1.0], [2.0], [10.0]];
        assert_eq!(knn_edges(x.view(), 1), vec![[0, 1], [1, 2], [2, 3]]);
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        let x = array![[0.0], [1.0], [-1.0]];
        assert_eq!(knn_edges(x.view(), 1), vec![[0, 1], [0, 2]]);
    }

    #[test]
    fn duplicate_rows_are_deterministic() {
        let x = array![[0.0], [0.0], [5.0]];
        let a = knn_edges(x.view(), 1);
        assert_eq!(a, vec![[0, 1], [0, 2]]);
        assert_eq!(a, knn_edges(x.view(), 1));
    }

    #[test]
    fn matches_brute_force_in_several_dimensions() {
        // Points on a deterministic lattice-like cloud, compared with a plain
        // double loop over every pair.
        let n = 300;
        let x = ndarray::Array2::from_shape_fn((n, 5), |(i, j)| {
            ((i * 7 + j * 13) % 29) as f64 * 0.37 + ((i * j) % 11) as f64 * 0.11
        });
        let k = 6;
        let mut expect = std::collections::BTreeSet::new();
        for i in 0..n {
            let mut d: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let s: f64 = (0..5).map(|c| (x[[i, c]] - x[[j, c]]).powi(2)).sum();
                    (s, j)
                })
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for &(_, j) in &d[..k] {
                expect.insert([i.min(j) as u32, i.max(j) as u32]);
            }
        }
        assert_eq!(
            knn_edges(x.view(), k),
            expect.into_iter().collect::<Vec<_>>()
        );
    }
}
