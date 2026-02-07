from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class ExecResult:
    ok: bool
    command: list[str]
    stdout_path: Path
    stderr_path: Path
    executed_notebook: Path


def _find_nbconvert_cmd() -> list[str]:
    # Prefer `jupyter nbconvert` then `python -m nbconvert` fallback.
    if shutil.which("jupyter"):
        return ["jupyter", "nbconvert"]
    return ["python", "-m", "nbconvert"]


def execute_notebook(
    notebook_path: Path,
    out_dir: Path,
    executed_name: Optional[str] = None,
    timeout: int = 600,
) -> ExecResult:
    """Execute a notebook and save an executed copy into out_dir.

    - Copies the notebook into out_dir first.
    - Runs nbconvert in out_dir so relative paths resolve there.
    - Saves stdout/stderr logs for debugging.

    Note: if the notebook uses absolute local paths, execution may still fail.
    """
    notebook_path = notebook_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    executed_name = executed_name or (notebook_path.stem + "_executed.ipynb")
    executed_nb = out_dir / executed_name

    shutil.copy2(notebook_path, executed_nb)

    nbconvert = _find_nbconvert_cmd()

    stdout_path = out_dir / "notebook_exec_stdout.log"
    stderr_path = out_dir / "notebook_exec_stderr.log"

    cmd = nbconvert + [
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout", str(timeout),
        executed_nb.name,
    ]

    p = subprocess.run(
        cmd,
        cwd=str(out_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_path.write_text(p.stdout or "", encoding="utf-8")
    stderr_path.write_text(p.stderr or "", encoding="utf-8")

    return ExecResult(
        ok=(p.returncode == 0),
        command=cmd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        executed_notebook=executed_nb,
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="eeg48 notebook-run",
        description="Execute a Jupyter notebook into an output directory (logs included)."
    )
    ap.add_argument("--notebook", required=True, help="Path to .ipynb")
    ap.add_argument("--out-dir", required=True, help="Directory to execute notebook in (and save executed copy).")
    ap.add_argument("--timeout", type=int, default=600, help="Execution timeout (seconds).")
    ap.add_argument("--executed-name", default=None, help="Optional executed notebook filename.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    res = execute_notebook(Path(args.notebook), Path(args.out_dir), executed_name=args.executed_name, timeout=args.timeout)
    if res.ok:
        print(f"[OK] executed: {res.executed_notebook}")
    else:
        print("[FAIL] execution returned non-zero exit code. See logs:")
        print(f"  stdout: {res.stdout_path}")
        print(f"  stderr: {res.stderr_path}")
    print(f"[CMD] {' '.join(res.command)}")
    return 0 if res.ok else 1
