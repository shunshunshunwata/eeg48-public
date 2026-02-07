from __future__ import annotations

import argparse
import csv
import html
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
TEXT_EXT = {".txt", ".md"}
DATA_EXT = {".csv", ".tsv", ".json"}
OTHER_EXT = {".pdf"}

@dataclass
class Artifact:
    relpath: str
    ext: str
    size_bytes: int

def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def _load_metrics(run_dir: Path) -> dict:
    candidates = [
        run_dir / "metrics.json",
        run_dir / "summary.json",
        run_dir / "results.json",
        run_dir / "metrics" / "metrics.json",
        run_dir / "report" / "metrics.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}

def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f}{u}" if u != "B" else f"{int(size)}{u}"
        size /= 1024
    return f"{int(n)}B"

def generate_results_card(run_dir: Path, out_dir: Optional[Path] = None, title: Optional[str] = None) -> Tuple[Path, Path]:
    run_dir = run_dir.resolve()
    if out_dir is None:
        out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[Artifact] = []
    images: List[str] = []
    tables: List[str] = []
    texts: List[str] = []
    others: List[str] = []

    for p in _iter_files(run_dir):
        if out_dir in p.parents:
            continue
        rel = str(p.relative_to(run_dir)).replace(os.sep, "/")
        ext = p.suffix.lower()
        artifacts.append(Artifact(relpath=rel, ext=ext, size_bytes=p.stat().st_size))
        if ext in IMAGE_EXT:
            images.append(rel)
        elif ext in DATA_EXT:
            tables.append(rel)
        elif ext in TEXT_EXT:
            texts.append(rel)
        elif ext in OTHER_EXT:
            others.append(rel)

    index_csv = out_dir / "artifacts_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "ext", "size_bytes", "size_human"])
        for a in sorted(artifacts, key=lambda x: x.relpath):
            w.writerow([a.relpath, a.ext, a.size_bytes, _human_size(a.size_bytes)])

    metrics = _load_metrics(run_dir)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    page_title = title or f"EEG48 Results Card — {run_dir.name}"

    images_show = images[:24]

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    if isinstance(metrics, dict) and metrics:
        metrics_rows = []
        for k, v in metrics.items():
            try:
                vv = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            except Exception:
                vv = str(v)
            metrics_rows.append(f"<tr><td><code>{esc(str(k))}</code></td><td><code>{esc(vv)}</code></td></tr>")
        metrics_html = "<table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>" + "".join(metrics_rows) + "</tbody></table>"
    else:
        metrics_html = "<p class='muted'>No metrics.json detected. (Optional) Add one to display key scores here.</p>"

    def list_block(items: List[str], label: str) -> str:
        if not items:
            return f"<p class='muted'>No {label} found.</p>"
        li = "\n".join([f"<li><a href='../{esc(x)}'>{esc(x)}</a></li>" for x in items[:50]])
        more = f"<p class='muted'>Showing 50 / {len(items)}. See artifacts_index.csv for the full list.</p>" if len(items) > 50 else ""
        return f"<ul>{li}</ul>{more}"

    html_out = out_dir / "results_card.html"
    html_out.write_text(f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{esc(page_title)}</title>
<style>
:root {{
  --bg: #0b0f14;
  --card: #121a23;
  --text: #e8eef6;
  --muted: #9bb0c6;
  --accent: #8bd5ff;
  --border: rgba(255,255,255,0.08);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 24px;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}}
h1 {{ font-size: 22px; margin: 0 0 6px 0; }}
h2 {{ font-size: 16px; margin: 18px 0 8px 0; color: var(--accent); }}
p {{ margin: 8px 0; line-height: 1.5; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  padding: 2px 6px;
  border-radius: 8px;
}}
.container {{ max-width: 1100px; margin: 0 auto; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}}
.muted {{ color: var(--muted); }}
img {{
  width: 100%;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
}}
.small {{ font-size: 12px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 14px;
  border: 1px solid var(--border);
}}
th, td {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
th {{ text-align: left; background: rgba(255,255,255,0.05); }}
tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>{esc(page_title)}</h1>
    <p class="muted small">Generated: {esc(now)} · Run dir: <code>{esc(str(run_dir))}</code></p>
    <p class="muted">This page aggregates artifacts for quick review: key metrics, figures, and file index. Share it as a single “evidence card”.</p>
  </div>

  <h2>Key metrics</h2>
  <div class="card">
    {metrics_html}
    <p class="muted small">Tip: put <code>metrics.json</code> in the run directory to show scores (e.g., mean AUC, p-values, N subjects).</p>
  </div>

  <h2>Key figures</h2>
  <div class="card">
    {("<p class='muted'>No images found in run directory.</p>" if not images_show else "")}
    <div class="grid">
      {("".join([f"<div><a href='../{esc(img)}'><img src='../{esc(img)}' alt='{esc(img)}'/></a><p class='small muted'>{esc(img)}</p></div>" for img in images_show]))}
    </div>
    {("<p class='muted small'>Showing first 24 images. Full list in artifacts_index.csv.</p>" if len(images) > 24 else "")}
  </div>

  <h2>Tables & data files</h2>
  <div class="card">{list_block(tables, "tables")}</div>

  <h2>Text notes</h2>
  <div class="card">{list_block(texts, "texts")}</div>

  <h2>Other files</h2>
  <div class="card">{list_block(others, "other files")}</div>

  <h2>Artifact index</h2>
  <div class="card">
    <p>Full file list: <a href="./artifacts_index.csv"><code>artifacts_index.csv</code></a></p>
  </div>

  <p class="muted small">Made with <code>eeg48 results-card</code>.</p>
</div>
</body>
</html>
''', encoding="utf-8")

    return html_out, index_csv

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eeg48 results-card", description="Generate a single-page HTML Results Card for a run directory.")
    p.add_argument("--run-dir", required=True, type=str, help="Path to directory containing outputs (figures/tables/metrics).")
    p.add_argument("--out-dir", default=None, type=str, help="Optional output directory (default: <run-dir>/report).")
    p.add_argument("--title", default=None, type=str, help="Optional page title.")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    html_path, index_path = generate_results_card(run_dir=run_dir, out_dir=out_dir, title=args.title)
    print(f"[OK] Results Card: {html_path}")
    print(f"[OK] Artifact index: {index_path}")
    return 0
