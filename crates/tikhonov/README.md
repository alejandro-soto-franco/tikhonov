# tikhonov

Pure-Rust implementation of the Harmony2 single-cell integration algorithm ([Patikas et al. 2026](https://github.com/immunogenomics/harmony2-ms)), targeting numerical parity with [harmony-R](https://github.com/immunogenomics/harmony) 1.2.4.

## Quickstart

```rust
use ndarray::array;
use tikhonov::{HarmonyConfig, run_harmony};

// Z: d x n PC embedding (d PCs, n cells). labels: n x n_cov batch codes (u32).
let z = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
let labels = array![[0], [1], [0]];
let config = HarmonyConfig::new().with_nclust(2).with_max_iter(10);
let out = run_harmony(z.view(), labels.view(), &config)?;
println!("converged in {} iters", out.n_iter);
```

## Numerical parity

- Tier 1 (CI): `1e-6` max relative error vs harmony-R 1.2.4 on three synthetic fixtures.
- Tier 2 (nightly): `1e-3` max relative error + cluster majority ≥ 99% on Tirosh 2016 oligodendroglioma.
- See the project README for Tier 3 (Patikas tahoe scale) results.

## License

Apache-2.0.
