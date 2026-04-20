#!/usr/bin/env bash
# Patikas-style sweep driver for `tikhonov-cli`.
#
# Usage: ./run_tikhonov.sh <input.h5ad> [cmd_list]
#
# Reads `cmd-tikhonov.txt` (or the path in $2), invoking `tikhonov` with the
# Patikas flag set. Each run emits a Patikas-schema JSON under `results/`.

set -euo pipefail

INPUT=${1:?"first arg: path to .h5ad with obsm/X_pca and the batch obs column"}
CMD_LIST=${2:-$(dirname "$0")/cmd-tikhonov.txt}
OUT_DIR=$(dirname "$0")/results
mkdir -p "$OUT_DIR"

# Ensure tikhonov is on PATH.
if ! command -v tikhonov >/dev/null; then
    echo "tikhonov binary not found on PATH; did you 'cargo install --path crates/tikhonov-cli'?" >&2
    exit 1
fi

i=0
while IFS= read -r line; do
    # Skip comments and blanks.
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    i=$((i + 1))
    echo "[$i] tikhonov --input $INPUT $line"
    # shellcheck disable=SC2086
    tikhonov --input "$INPUT" -o "$OUT_DIR" $line
done < "$CMD_LIST"

echo "Done. $i runs written to $OUT_DIR/"
