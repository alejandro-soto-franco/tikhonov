//! Top-level [`run_harmony`] driver.

use crate::cluster::{dist_mat, normalise_cols_l1, update_r_block, update_y};
use crate::config::HarmonyConfig;
use crate::correct::{apply_moe_ridge, default_lambda};
use crate::embed::{kmeans_pp_init, l2_normalize_cols};
use crate::error::HarmonyError;
use crate::history::{HarmonyHistory, HistoryEntry};
use crate::objective;
use crate::phi::Phi;
use ndarray::{Array1, Array2, ArrayView2};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::time::Instant;

pub struct HarmonyResult {
    pub z_corr: Array2<f64>,
    pub y: Array2<f64>,
    pub r: Array2<f64>,
    pub history: HarmonyHistory,
    pub converged: bool,
    pub n_iter: usize,
}

/// Run Harmony2 integration on a PC embedding `z` with batch `labels`.
///
/// - `z`: `d × n` (PCs are rows, cells are columns).
/// - `labels`: `n × n_cov` u32 codes per covariate.
pub fn run_harmony(
    z: ArrayView2<'_, f64>,
    labels: ArrayView2<'_, u32>,
    config: &HarmonyConfig,
) -> Result<HarmonyResult, HarmonyError> {
    let (_d, n) = z.dim();
    if labels.nrows() != n {
        return Err(HarmonyError::ShapeMismatch(format!(
            "labels has {} rows; expected {} (one per cell)",
            labels.nrows(),
            n
        )));
    }

    if let Some(nt) = config.n_threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build_global();
    }

    let phi = Phi::from_codes(labels)?;
    let k = config.resolved_nclust(n);
    let theta = Array1::from(config.resolved_theta(phi.n_cov));
    let sigma = Array1::from(vec![config.sigma; k]);
    let pr_b = phi.pr_b();
    let lambda = match &config.lambda {
        None => default_lambda(phi.b),
        Some(lam) if lam.len() == 1 => {
            let mut v = vec![lam[0]; phi.b + 1];
            v[0] = 1e-8;
            v
        }
        Some(lam) if lam.len() == phi.b + 1 => lam.clone(),
        Some(lam) => {
            return Err(HarmonyError::InvalidConfig(format!(
                "lambda must have length 1 or B+1 = {}; got {}",
                phi.b + 1,
                lam.len()
            )));
        }
    };

    let z_orig = z.to_owned();
    let mut z_cos = l2_normalize_cols(z.view());
    let mut y = kmeans_pp_init(z_cos.view(), k, config.seed);

    let mut dist = dist_mat(y.view(), z_cos.view());
    let mut r = Array2::<f64>::zeros((k, n));
    for kk in 0..k {
        for i in 0..n {
            r[[kk, i]] = (-dist[[kk, i]] / sigma[kk]).exp();
        }
    }
    normalise_cols_l1(&mut r);

    let mut e = Array2::<f64>::zeros((k, phi.b));
    let mut o = Array2::<f64>::zeros((k, phi.b));
    let row_sums_r: Vec<f64> = (0..k).map(|kk| r.row(kk).sum()).collect();
    for kk in 0..k {
        for b in 0..phi.b {
            e[[kk, b]] = row_sums_r[kk] * pr_b[b];
        }
    }
    for c in 0..phi.n_cov {
        for i in 0..n {
            let b = phi.row_of_cell[c * phi.n + i] as usize;
            for kk in 0..k {
                o[[kk, b]] += r[[kk, i]];
            }
        }
    }

    let mut history = HarmonyHistory::new();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let t_total = Instant::now();

    let (km, ent, cross, tot) = objective::compute(
        r.view(),
        dist.view(),
        o.view(),
        e.view(),
        sigma.view(),
        theta.view(),
        &phi,
    );
    history.push(HistoryEntry {
        iter: 0,
        cluster_iters: 0,
        kmeans_cost: km,
        kl_cost: ent,
        ridge_cost: cross,
        objective: tot,
        elapsed_ms: t_total.elapsed().as_millis() as u64,
    });

    let mut converged = false;
    let mut n_iter_done = 0usize;

    for iter in 1..=config.max_iter {
        n_iter_done = iter;

        let mut inner_iters = 0usize;
        let mut inner_obj: Vec<f64> = vec![history.last().unwrap().objective];
        for j in 1..=config.max_iter_cluster {
            inner_iters = j;
            y = update_y(z_cos.view(), r.view());
            dist = dist_mat(y.view(), z_cos.view());
            let _scale = update_r_block(
                &mut r,
                &mut o,
                &mut e,
                dist.view(),
                &phi,
                pr_b.view(),
                sigma.view(),
                theta.view(),
                config.block_size,
                &mut rng,
            );
            let (_km, _ent, _cross, tot2) = objective::compute(
                r.view(),
                dist.view(),
                o.view(),
                e.view(),
                sigma.view(),
                theta.view(),
                &phi,
            );
            inner_obj.push(tot2);
            if j > 3 {
                let n_obj = inner_obj.len();
                let old_w: f64 = inner_obj[n_obj - 4..n_obj - 1].iter().sum();
                let new_w: f64 = inner_obj[n_obj - 3..n_obj].iter().sum();
                if old_w.abs() > 0.0 && ((old_w - new_w) / old_w.abs()) < config.epsilon_cluster {
                    break;
                }
            }
        }

        let mut z_corr = z_orig.clone();
        apply_moe_ridge(z_orig.view(), &mut z_corr, r.view(), &phi, &lambda)?;
        z_cos = l2_normalize_cols(z_corr.view());
        dist = dist_mat(y.view(), z_cos.view());

        let (km, ent, cross, tot2) = objective::compute(
            r.view(),
            dist.view(),
            o.view(),
            e.view(),
            sigma.view(),
            theta.view(),
            &phi,
        );
        history.push(HistoryEntry {
            iter,
            cluster_iters: inner_iters,
            kmeans_cost: km,
            kl_cost: ent,
            ridge_cost: cross,
            objective: tot2,
            elapsed_ms: t_total.elapsed().as_millis() as u64,
        });

        if history.last_rel_change() < config.epsilon_harmony {
            converged = true;
            break;
        }
    }

    let z_corr_final = {
        let mut zc = z_orig.clone();
        apply_moe_ridge(z_orig.view(), &mut zc, r.view(), &phi, &lambda)?;
        zc
    };

    Ok(HarmonyResult {
        z_corr: z_corr_final,
        y,
        r,
        history,
        converged,
        n_iter: n_iter_done,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn runs_end_to_end_on_toy() {
        let mut z = Array2::<f64>::zeros((4, 6));
        for i in 0..3 {
            z[[0, i]] = 1.0 + 0.01 * i as f64;
            z[[1, i]] = 0.5;
        }
        for i in 3..6 {
            z[[0, i]] = -1.0 + 0.01 * (i - 3) as f64;
            z[[1, i]] = -0.5;
        }
        let labels = array![[0u32], [0], [0], [1], [1], [1]];
        let cfg = HarmonyConfig::new()
            .with_nclust(2)
            .with_max_iter(5)
            .with_max_iter_cluster(20)
            .with_block_size(0.2);
        let out = run_harmony(z.view(), labels.view(), &cfg).unwrap();
        assert_eq!(out.z_corr.dim(), (4, 6));
        assert!(!out.history.entries.is_empty());
    }
}
