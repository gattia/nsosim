#!/usr/bin/env bash
# Download transform chain test fixtures from GitHub Releases.
#
# Usage:
#   cd tests/fixtures/transforms && ./download_fixtures.sh
#
# Or just run pytest — the conftest auto-downloads if fixtures are missing.

set -euo pipefail

RELEASE_TAG="test-fixtures-v1"
REPO="gattia/nsosim"
ASSET="transform-fixtures.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if fixtures already exist
if [ -f "nsm_recon_ref_femur.vtk" ] && [ -f "subject_9003316/femur_nsm_recon_mm.vtk" ]; then
    echo "Fixtures already present, skipping download."
    exit 0
fi

echo "Downloading transform test fixtures from ${REPO} release ${RELEASE_TAG}..."

if command -v gh &> /dev/null; then
    gh release download "$RELEASE_TAG" --repo "$REPO" --pattern "$ASSET" --dir /tmp --clobber
else
    URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET}"
    curl -sL -o "/tmp/${ASSET}" "$URL"
fi

echo "Extracting..."
tar xzf "/tmp/${ASSET}" -C "$SCRIPT_DIR"
rm -f "/tmp/${ASSET}"

echo "Done. Fixtures extracted to ${SCRIPT_DIR}"
