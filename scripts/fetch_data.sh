#!/usr/bin/env bash
# Download the two benchmark tables and the published embeddings (about 190 MB, CC BY 4.0)
# from the authors' repository into ./data. Nothing else from that repository is used.
set -euo pipefail
dest="${1:-data}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git clone --depth 1 --filter=blob:none --sparse https://github.com/nmdl-mizo/decoupled-gnn-transfer-learning.git "$tmp/src"
git -C "$tmp/src" sparse-checkout set data
mkdir -p "$dest"
cp -r "$tmp/src/data/lic" "$tmp/src/data/aoi" "$dest/"
echo "data ready in $dest/"
