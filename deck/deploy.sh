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
# A Markdown file as a page, rendered by GitHub's own Markdown renderer: md_page DIR TITLE DESCRIPTION < file
md_page() {
  mkdir -p "$tmp/$1"
  local body
  body=$(gh api markdown -F mode=gfm -F context=DrealVeerNanda/ucsc-ai-hackathon-2026 -F text=@-)
  cat > "$tmp/$1/index.html" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0a0f17">
  <title>$2</title>
  <meta name="description" content="$3">
  <link rel="icon" href="../icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="../readme.css">
</head>
<body>
  <nav class="top" aria-label="Site"><a href="../">Presentation</a><span><a href="../script/">Script</a> &nbsp; <a href="../writeup/">Write-up</a> &nbsp; <a href="../readme/">README</a></span></nav>
  <main class="doc">
$body
  </main>
</body>
</html>
HTML
}
# The README, without the email addresses.
sed -E 's/ \([^()]*@[^()]*\.[a-z]+\)//' README.md | md_page readme "Proprio README" "How Proprio is built and run: the pipeline, the GX10 setup, and the options."
md_page script "Proprio pitch script" "The 2-minute pitch, the live demo run-of-show, Q&amp;A prep and where every number comes from." < docs/script.md
python3 tools/writeup_page.py "$tmp"     # the Devpost write-up, from docs/devpost.md, at writeup/
cd "$tmp"
git init -q -b gh-pages
git add -A
git commit -q -m "Deck from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push -q -f "$(git -C "$OLDPWD" remote get-url origin)" gh-pages
echo "Published: https://drealveernanda.github.io/ucsc-ai-hackathon-2026/ and .../script/, .../writeup/, .../readme/"
