# Changelog

## Unreleased

- Phase 5 Patikas scale benchmark has not yet been executed (needs tahoe dataset + 16-cpu machine).
- Tier-1 goldens (harmony-R 1.2.4 parity at `1e-6`) deferred pending an R + harmony-1.2.4 environment.

## 0.1.0 (target)

- Core `tikhonov` crate: full harmony-R 1.2.4 algorithm, `f64` throughout, `faer 0.22` Cholesky, rayon-ready kernels, deterministic ChaCha8 rng.
- `tikhonov-cli` binary with Patikas `RunHarmony.R` flag parity and Patikas-schema JSON emitter.
- `tikhonov-py` PyO3 bindings + Python package: `tikhonov.integrate(adata, key=...)` and `tikhonov.run_harmony(Z, meta, vars_use, ...)`.
- CI matrix: fmt, clippy, core test on ubuntu/macos/windows, Python 3.10 + 3.12.
- Tirosh 2016 oligodendroglioma functional integration test (`@pytest.mark.integration`).
- Patikas sweep harness (`benchmarks/patikas/`) + `compare.py` plotter.
