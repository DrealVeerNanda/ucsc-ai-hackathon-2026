#!/usr/bin/env bash
# Publishes deck/ (and only deck/) to the gh-pages branch, which GitHub Pages serves at
# https://drealveernanda.github.io/ucsc-ai-hackathon-2026/. The rest of the repo stays private.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -R deck/. "$tmp"
rm -f "$tmp/deploy.sh"
touch "$tmp/.nojekyll"
cd "$tmp"
git init -q -b gh-pages
git add -A
git commit -q -m "Deck from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q -f "$(git -C "$OLDPWD" remote get-url origin)" gh-pages
echo "Published: https://drealveernanda.github.io/ucsc-ai-hackathon-2026/"
