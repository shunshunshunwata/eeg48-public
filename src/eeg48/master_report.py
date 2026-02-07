from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rel(from_dir: Path, to_path: Path) -> str:
    try:
        return os.path.relpath(to_path, start=from_dir)
    except Exception:
        return str(to_path)


def generate_master_report(
    root_dir: Path,
    module_run_dirs: Dict[str, Path],
    out_dir: Optional[Path] = None,
    title: str = "EEG48 — Master Report",
) -> Tuple[Path, Path]:
    """Generate an index HTML that links module Results Cards + summaries.

    Writes:
      - <out_dir>/index.html
      - <out_dir>/summary.json
    """
    root_dir = root_dir.expanduser().resolve()
    out_dir = (out_dir.expanduser().resolve() if out_dir else (root_dir / "eeg48_reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "modules": {},
    }

    rows = []
    for mod, run_dir in module_run_dirs.items():
        run_dir = Path(run_dir).expanduser().resolve()
        metrics_path = run_dir / "metrics.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        card = run_dir / "report" / "results_card.html"
        card_link = _rel(out_dir, card) if card.exists() else None

        # Small metric snippet that is safe to show
        snippet = {}
        for k in ("mean_score", "best_task", "n_tasks_scored", "cv", "primary_model"):
            if k in metrics:
                snippet[k] = metrics.get(k)

        summary["modules"][mod] = {
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "results_card": str(card) if card.exists() else None,
            "snippet": snippet,
        }

        rows.append((mod, run_dir, card_link, snippet))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML
    def fmt_snippet(s: dict) -> str:
        if not s:
            return "<em>—</em>"
        parts = []
        if "mean_score" in s and s["mean_score"] is not None:
            parts.append(f"mean_score={html.escape(str(s['mean_score']))}")
        if "n_tasks_scored" in s:
            parts.append(f"n_tasks={html.escape(str(s['n_tasks_scored']))}")
        if "cv" in s:
            parts.append(f"cv={html.escape(str(s['cv']))}")
        if "primary_model" in s:
            parts.append(f"model={html.escape(str(s['primary_model']))}")
        return "<br/>".join(parts) if parts else "<em>—</em>"

    tr = []
    for mod, run_dir, card_link, snippet in rows:
        mod_h = html.escape(mod)
        run_h = html.escape(str(run_dir))
        if card_link:
            card_html = f'<a href="{html.escape(card_link)}">results_card.html</a>'
        else:
            card_html = "<em>not found</em>"
        tr.append(
            f"<tr>"
            f"<td><b>{mod_h}</b></td>"
            f"<td><code>{run_h}</code></td>"
            f"<td>{card_html}</td>"
            f"<td>{fmt_snippet(snippet)}</td>"
            f"</tr>"
        )

    html_out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color: #555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="sub">Generated: {html.escape(summary["generated_at"])}</div>

  <h2>Modules</h2>
  <table>
    <thead>
      <tr>
        <th>Module</th>
        <th>Run directory</th>
        <th>Results Card</th>
        <th>Key metrics</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tr)}
    </tbody>
  </table>

  <p style="margin-top:18px;">
    JSON summary: <a href="{html.escape(_rel(out_dir, summary_path))}">summary.json</a>
  </p>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(html_out, encoding="utf-8")
    return index_path, summary_path
