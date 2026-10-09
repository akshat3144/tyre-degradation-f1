#!/usr/bin/env bash
# Fetch reference material. Not tracked in git -- downloaded, not written by us.
#
#   bash scripts/fetch_reference.sh
#
# Gets the benchmark authors' code repo into benchmark/upstream/, and copies
# their Stan models into benchmark/stan/ with array syntax migrated.
#
# The benchmark paper PDFs are fetched separately by paper/refs/fetch.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p benchmark/stan

echo "==> upstream repo"
rm -rf benchmark/upstream
git clone --depth 1 -q https://github.com/colecappello12/F1_SSM_Paper.git benchmark/upstream
rm -rf benchmark/upstream/.git   # vendored for reference, not a submodule

echo "==> migrating their Stan models to 2.32+ array syntax"
# Their code predates Stan 2.32, which removed `int name[N];`. This is a
# mechanical declaration change only -- the model, priors and data blocks are
# untouched. upstream/ is left pristine so the diff stays auditable.
SRC="benchmark/upstream/Cross_Validation_Scripts_and_Stan_Code"
cp "$SRC"/*.stan "$SRC"/*.stanfunctions benchmark/stan/
sed -i -E 's/^([[:space:]]*)int ([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z0-9_]+)\];/\1array[\3] int \2;/' \
  benchmark/stan/*.stan

echo
echo "done:"
echo "  benchmark/upstream/  ($(find benchmark/upstream -type f | wc -l) files)"
echo "  benchmark/stan/      ($(ls -1 benchmark/stan | wc -l) files, array syntax migrated)"
