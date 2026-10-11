"""Turns docs/devpost.md (the Devpost write-up) into the writeup/ page of the Pages site.

    python tools/writeup_page.py OUT_DIR      # deck/deploy.sh runs it

Handles the Markdown that file uses: ## headings, - lists, **bold**, _italic_ and $$ display math,
which KaTeX (deck/vendor/katex) draws in the browser. A # line is the elevator pitch."""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PITCH = ("For blind people, finding a dropped or misplaced item is the most common everyday challenge. "
         "Proprio guides their hand onto anything they ask for, one spoken word at a time.")


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", t)


def body(md: str) -> str:
    out = []
    for block in md.strip().split("\n\n"):
        items = []
        for line in block.strip().split("\n"):
            if line.startswith("# "):
                out.append(f'<p class="pitch">{inline(line[2:])}</p>')
            elif line.startswith("## "):
                out.append(f"<h2>{inline(line[3:])}</h2>")
            elif line.startswith("- "):
                items.append(f"<li>{inline(line[2:])}</li>")
            elif line.startswith("$$"):
                # Devpost's Markdown eats a bare _ inside $$, so the file writes \_; KaTeX wants _
                out.append(f'<div class="math">{html.escape(line.strip("$").replace(chr(92) + "_", "_"))}</div>')
            else:
                out.append(f"<p>{inline(line)}</p>")
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
    return "\n".join(out)


def main(out_dir: str) -> None:
    md = (ROOT / "docs" / "devpost.md").read_text()
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0a0f17">
  <title>Proprio</title>
  <meta name="description" content="{html.escape(PITCH)}">
  <link rel="icon" href="../icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="../vendor/katex/katex.min.css">
  <link rel="stylesheet" href="../readme.css">
</head>
<body>
  <nav class="top" aria-label="Site"><a href="../">Presentation</a><a href="../readme/">README</a></nav>
  <main class="doc">
    <h1>Proprio</h1>
{body(md)}
  </main>
  <script src="../vendor/katex/katex.min.js"></script>
  <script>
    document.querySelectorAll(".math").forEach((e) => katex.render(e.textContent, e, {{ displayMode: true, throwOnError: false }}));
  </script>
</body>
</html>
"""
    out = Path(out_dir) / "writeup"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)


if __name__ == "__main__":
    main(sys.argv[1])
