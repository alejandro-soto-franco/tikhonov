# tikhonov

Pure-Rust Harmony2 for single-cell data integration, with Python bindings and a CLI.

- **Rust crate:** [`tikhonov`](crates/tikhonov) on crates.io.
- **CLI:** `cargo install tikhonov-cli` installs the `tikhonov` binary.
- **Python:** `pip install tikhonov` and `import tikhonov`.

## Status

v0.1.0 in development. Targets numerical parity with [harmony-R 1.2.4](https://github.com/immunogenomics/harmony) on small fixtures (`1e-6`) and convergence-metric parity on mid-size fixtures (`1e-3`). See the individual crate READMEs for quickstarts.

## License

Apache-2.0. See `LICENSE`.
