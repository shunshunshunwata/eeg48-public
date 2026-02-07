from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_EXTS = {".py",".ipynb",".md",".txt",".sh",".yml",".yaml",".toml",".json",".csv"}

PATTERNS = {
    "mac_user_path": re.compile(r"/Users/([^/\s]+)/"),
    "win_user_path": re.compile(r"([A-Z]:\\Users\\([^\\\s]+)\\)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}

def scan_file(path: Path) -> List[Dict]:
    hits: List[Dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return hits
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for name, pat in PATTERNS.items():
            for m in pat.finditer(line):
                val = m.group(0)
                hits.append({
                    "pattern": name,
                    "file": str(path.as_posix()),
                    "line": i,
                    "match": val,
                })
    return hits

def scan_repo(root: Path, exts: Optional[set[str]] = None, exclude_dirs: Optional[List[str]] = None) -> List[Dict]:
    exts = exts or DEFAULT_EXTS
    exclude_dirs = exclude_dirs or [".git", ".venv", "venv", "__pycache__", "dist", "build"]
    hits: List[Dict] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in exts:
            continue
        hits.extend(scan_file(p))
    return hits

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eeg48 privacy-scan", description="Scan repository for common private identifiers (user paths, emails).")
    p.add_argument("--root", default=".", help="Repository root to scan.")
    p.add_argument("--out", default=None, help="Optional JSON output file.")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    hits = scan_repo(root)
    print(f"[OK] scanned: {root}")
    print(f"[OK] hits: {len(hits)}")
    if hits:
        # Print compact summary
        for h in hits[:50]:
            print(f"- {h['pattern']}: {h['file']}:{h['line']}  {h['match']}")
        if len(hits) > 50:
            print(f"... (showing 50 / {len(hits)})")
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote: {outp}")
    return 0
