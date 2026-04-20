# Patikas 2026 benchmark reproduction

Reproduces the scale benchmark from [Patikas et al. 2026](https://github.com/immunogenomics/harmony2-ms) using the `tikhonov` CLI in place of `RunHarmony.R`.

## Schema compatibility

`tikhonov-cli` emits JSON with the exact key set the Patikas harness produces (see `tikhonov-research/harmony2-ms/tahoe/benchmark/results/*.txt`), so our output drops into their comparison notebook without transformation.

## Layout

```
benchmarks/patikas/
├── README.md           this file
├── run_tikhonov.sh     wrapper: translates Patikas-style env + magic tag to CLI flags
├── cmd-tikhonov.txt    sweep list mirroring cmd-new.txt
├── scripts/            supporting scripts (slurm, env)
├── results/            emitted JSON lands here (gitignored output)
└── compare.py          plots our results vs harmony-R 1.2.4 results
```

## Running the sweep

Requires an `.h5ad` file with `obsm["X_pca"]` and an `obs` batch column. On a 16-CPU machine with the tahoe dataset staged as `tahoe-profile.h5ad`:

```bash
bash run_tikhonov.sh tahoe-profile.h5ad
```

The script iterates over `cmd-tikhonov.txt`, invoking `tikhonov-cli` with the Patikas-compatible flag set and writing JSON to `results/`.

## Comparing to harmony-R 1.2.4

```bash
python compare.py \
  --ours benchmarks/patikas/results \
  --theirs ~/tikhonov-research/harmony2-ms/tahoe/benchmark/results \
  --out benchmarks/patikas/fig_scaling.png
```

`compare.py` loads both JSON directories, groups by `(batches_included, cells_included)`, and plots median ± IQR runtime and peak memory (tikhonov vs harmony-R 1.2.4).

## Skipped

Tail of the Patikas sweep (4M, 8M, 16M cells) is omitted until a 32-64 GB RAM machine is available. 1M and 2M runs are the default.
