# tikhonov-cli

Standalone CLI for Harmony2 single-cell integration. Reads an `.h5ad` file, runs Harmony2 on a PC embedding, writes the corrected embedding back, and emits Patikas-schema benchmark JSON.

## Install

```bash
cargo install tikhonov-cli
```

This installs the `tikhonov` binary.

## Quickstart

```bash
tikhonov \
  --input data.h5ad \
  --pc-key X_pca \
  --covariate-factors sample \
  --output-key X_harmony \
  --iterations 10 \
  --num-pcs 20 \
  --num-clusters 50 \
  --sigma 0.1 \
  --theta 2.0 \
  --ncpus 16 \
  --bench-json results/run.json
```

The corrected embedding lands in `obsm["X_harmony"]` of the input file.

## Patikas benchmark reproduction

The flag set matches [Patikas et al. 2026](https://github.com/immunogenomics/harmony2-ms)'s `RunHarmony.R` with one addition (`--input <path>`). Emitted JSON matches the schema at `harmony2-ms/tahoe/benchmark/results/`, so our results drop into their notebook unchanged.

## HDF5 dependency

This crate links against `libhdf5`. On Fedora: `dnf install hdf5-devel`. On macOS: `brew install hdf5`. On Ubuntu: `apt install libhdf5-dev`. Static builds are available via `hdf5-metno`'s `static` feature at the cost of longer compile times.

## License

Apache-2.0.
