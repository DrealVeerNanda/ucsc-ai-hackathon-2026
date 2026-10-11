#!/usr/bin/env bash
# Publishes deck/ (and only deck/) to the gh-pages branch, which GitHub Pages serves at
# https://drealveernanda.github.io/ucsc-ai-hackathon-2026/.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -R deck/. "$tmp"
rm -f "$tmp/deploy.sh"
touch "$tmp/.nojekyll"
# The README as a page at readme/, rendered by GitHub's own Markdown renderer. Email addresses are
# left out of the page.
mkdir -p "$tmp/readme"
body=$(sed -E 's/ \([^()]*@[^()]*\.[a-z]+\)//' README.md | gh api markdown -F mode=gfm -F context=DrealVeerNanda/ucsc-ai-hackathon-2026 -F text=@-)
cat > "$tmp/readme/index.html" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0a0f17">
  <title>Proprio README</title>
  <meta name="description" content="How Proprio is built and run: the pipeline, the GX10 setup, and the options.">
  <link rel="icon" href="../icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="../readme.css">
</head>
<body>
  <nav class="top" aria-label="Site"><a href="../">Presentation</a><span></span></nav>
  <main class="doc">
$body
  </main>
</body>
</html>
HTML
cd "$tmp"
git init -q -b gh-pages
git add -A
git commit -q -m "Deck from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q -f "$(git -C "$OLDPWD" remote get-url origin)" gh-pages
echo "Published: https://drealveernanda.github.io/ucsc-ai-hackathon-2026/ and .../readme/"
