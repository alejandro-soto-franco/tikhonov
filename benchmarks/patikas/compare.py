#!/usr/bin/env python3
"""
Compare tikhonov vs harmony-R 1.2.4 Patikas-benchmark results.

Inputs: two directories containing Patikas-schema JSONs. Groups records by
`(batches_included, cells_included)`, plots median ± IQR of `elapsed_secs`
and `peak_memory` side by side, saves a 2-panel figure.

Usage:
    python compare.py \
        --ours benchmarks/patikas/results \
        --theirs ~/tikhonov-research/harmony2-ms/tahoe/benchmark/results \
        --out benchmarks/patikas/fig_scaling.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dir(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.txt")) + sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def group(records: list[dict], by: tuple[str, ...] = ("batches_included", "cells_included")):
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in by)
        buckets[key].append(r)
    return buckets


def median_iqr(records: list[dict], field: str) -> tuple[float, float, float]:
    vals = np.array([r[field] for r in records if field in r], dtype=np.float64)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    return med, q25, q75


def plot_axis(ax, ours, theirs, field: str, ylabel: str, *, logy: bool = False):
    buckets_ours = group(ours)
    buckets_theirs = group(theirs)
    keys = sorted(set(buckets_ours) | set(buckets_theirs), key=lambda k: (k[0] or 0, k[1] or 0))
    x = np.arange(len(keys))
    labels = [f"{k[0]}B\n{k[1]//1_000_000}M" if k[1] else str(k) for k in keys]

    def series(buckets):
        med, lo, hi = [], [], []
        for k in keys:
            m, l, h = median_iqr(buckets.get(k, []), field)
            med.append(m); lo.append(l); hi.append(h)
        return np.array(med), np.array(lo), np.array(hi)

    mo, lo, ho = series(buckets_ours)
    mt, lt, ht = series(buckets_theirs)

    ax.errorbar(x - 0.08, mo, yerr=[mo - lo, ho - mo], fmt="o-", label="tikhonov", capsize=3)
    ax.errorbar(x + 0.08, mt, yerr=[mt - lt, ht - mt], fmt="s-", label="harmony-R 1.2.4", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--theirs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ours = load_dir(args.ours)
    theirs = load_dir(args.theirs)

    if not ours and not theirs:
        print("No records found in either directory.")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_axis(axes[0], ours, theirs, "elapsed_secs", "Elapsed (s)", logy=True)
    plot_axis(axes[1], ours, theirs, "peak_memory", "Peak RSS (MiB)", logy=True)
    axes[0].set_title("Runtime")
    axes[1].set_title("Peak memory")
    fig.suptitle(f"Patikas benchmark: tikhonov ({len(ours)} runs) vs harmony-R 1.2.4 ({len(theirs)} runs)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
