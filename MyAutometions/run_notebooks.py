#!/usr/bin/env python3
"""
Run all .ipynb notebooks under a root directory, save executed notebooks,
and write a per-notebook detailed log that includes every cell's source and outputs.

Behavior:
- Finds notebooks recursively under --root (default '.').
- Executes each notebook using nbconvert.ExecutePreprocessor.
- Saves executed notebooks to --outdir preserving relative paths with suffix '_executed.ipynb'.
- Writes a timestamped per-notebook log that contains, for each cell:
    - cell index and cell type
    - source code / markdown
    - all outputs (stream, execute_result, display_data, error)
- If a notebook fails, saves the partial executed notebook if possible and logs the exception,
  then continues to the next notebook.
"""

import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from datetime import datetime
import traceback
import sys
import logging
import json
import html
from typing import List, Any

# ---------- Helpers for logging cell outputs ----------

def render_output(output: dict) -> str:
    """
    Convert a single nbformat output object into a readable string for logs.
    Handles 'stream', 'execute_result', 'display_data', 'error'.
    """
    out_type = output.get('output_type', '<unknown>')
    if out_type == 'stream':
        # stream: usually stdout or stderr
        name = output.get('name', 'stdout')
        text = output.get('text', '')
        return f"[stream:{name}]\n{text}"
    elif out_type in ('execute_result', 'display_data'):
        # execute_result: python repr / display_data: images, rich data
        data = output.get('data', {})
        # Prefer text/plain if present, otherwise summarize keys
        if 'text/plain' in data:
            return f"[{out_type}]\n{data['text/plain']}"
        else:
            # For rich outputs (images, html, json, etc.) provide a short summary
            keys = ', '.join(sorted(data.keys()))
            # if image/png present we note it's binary image data
            summary_lines = [f"[{out_type}] available mime types: {keys}"]
            # try to include small text/html or application/json if present
            if 'text/html' in data:
                # escape HTML to keep logs safe/readable
                html_text = html.unescape(str(data['text/html']))[:2000]
                summary_lines.append("text/html (truncated):")
                summary_lines.append(html_text)
            if 'application/json' in data:
                try:
                    json_text = json.dumps(data['application/json'], indent=2)
                    summary_lines.append("application/json (truncated):")
                    summary_lines.append(json_text[:2000])
                except Exception:
                    summary_lines.append("application/json (non-serializable)")
            if 'image/png' in data or 'image/jpeg' in data:
                summary_lines.append("image data present (image/png or image/jpeg) - saved in executed notebook but not embedded in log.")
            return "\n".join(summary_lines)
    elif out_type == 'error':
        # error holds ename, evalue, traceback (list of lines)
        ename = output.get('ename', '')
        evalue = output.get('evalue', '')
        tb = output.get('traceback', [])
        tb_text = "\n".join(tb) if isinstance(tb, list) else str(tb)
        return f"[error] {ename}: {evalue}\nTraceback:\n{tb_text}"
    else:
        # fallback: pretty-print the whole object
        try:
            return f"[{out_type}] {json.dumps(output, indent=2, default=str)[:2000]}"
        except Exception:
            return f"[{out_type}] (unserializable output object)"

def notebook_to_log_text(nb: nbformat.NotebookNode, nb_path: Path, start_time: datetime, end_time: datetime, include_cell_outputs: bool = True) -> str:
    """
    Produce a multi-line string summarizing executed notebook including cell sources and outputs.
    """
    lines: List[str] = []
    header = f"Notebook: {nb_path}\nStarted: {start_time.isoformat()}Z\nEnded:   {end_time.isoformat()}Z\n"
    lines.append(header)
    lines.append("=" * 80)
    for idx, cell in enumerate(nb.cells):
        cell_type = cell.get('cell_type', '<unknown>')
        lines.append(f"\nCELL {idx} [{cell_type}]\n" + "-" * 60)
        # Source (truncate if extremely long)
        source = cell.get('source', '')
        src_preview = source if len(source) < 5000 else source[:5000] + "\n...[truncated]..."
        lines.append("SOURCE:\n" + src_preview)
        if include_cell_outputs:
            outputs = cell.get('outputs', [])
            if not outputs:
                lines.append("OUTPUTS: (none)")
            else:
                lines.append("OUTPUTS:")
                for out_i, out in enumerate(outputs):
                    lines.append(f"\n--- output {out_i} ---")
                    try:
                        rendered = render_output(out)
                    except Exception as e:
                        rendered = f"(failed to render output: {e})"
                    lines.append(rendered)
    return "\n".join(lines)


# ---------- Logging and execution logic ----------

def setup_logger(path: Path):
    """Return a logger writing to the specified file and to stdout (INFO)."""
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.DEBUG)
    # prevent multiple handlers if called repeatedly
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def execute_and_log_single_notebook(nb_path: Path, out_nb_path: Path, nb_log_path: Path,
                                    timeout: int, kernel_name: str, include_cell_outputs: bool):
    """
    Execute one notebook and write detailed log including each cell's outputs.
    """
    # Ensure directories exist
    nb_log_path.parent.mkdir(parents=True, exist_ok=True)
    out_nb_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(nb_log_path)
    logger.info(f"STARTING notebook: {nb_path}")
    start_time = datetime.utcnow()
    nb = None
    try:
        nb = nbformat.read(str(nb_path), as_version=4)
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

        # Execute in the notebook's directory so relative paths work
        exec_dir = str(nb_path.parent.resolve())
        logger.debug(f"Executing in {exec_dir} with kernel={kernel_name} timeout={timeout}")

        ep.preprocess(nb, {'metadata': {'path': exec_dir}})

        # Write executed notebook to out_nb_path
        nbformat.write(nb, str(out_nb_path))
        end_time = datetime.utcnow()
        # Build a detailed text log including cell outputs
        log_text = notebook_to_log_text(nb, nb_path, start_time, end_time, include_cell_outputs=include_cell_outputs)
        # Save human-readable log too (in addition to structured logger)
        with nb_log_path.open('a', encoding='utf-8') as f:
            f.write("\n\n" + ("#" * 20) + " DETAILED NOTEBOOK OUTPUTS " + ("#" * 20) + "\n")
            f.write(log_text)
        logger.info(f"SUCCESS: executed and saved -> {out_nb_path}")
        return True

    except Exception as e:
        # Log exception and traceback
        tb = traceback.format_exc()
        logger.error(f"EXCEPTION while executing {nb_path}:\n{e}\nTraceback:\n{tb}")
        # Try to save partial executed notebook if available
        if nb is not None:
            try:
                nbformat.write(nb, str(out_nb_path))
                logger.info(f"Partial executed notebook saved to {out_nb_path}")
                end_time = datetime.utcnow()
                # still dump whatever outputs we have into the log
                log_text = notebook_to_log_text(nb, nb_path, start_time, end_time, include_cell_outputs=include_cell_outputs)
                with nb_log_path.open('a', encoding='utf-8') as f:
                    f.write("\n\n" + ("#" * 20) + " PARTIAL NOTEBOOK OUTPUTS (failure occurred) " + ("#" * 20) + "\n")
                    f.write(log_text)
            except Exception as save_err:
                logger.error(f"Failed to save partial notebook or logs: {save_err}")
        return False


# ---------- Notebook discovery ----------

def find_notebooks(root: Path, recursive: bool = True):
    if recursive:
        return sorted(root.rglob("*.ipynb"))
    else:
        return sorted(root.glob("*.ipynb"))


# ---------- CLI entrypoint ----------

def main():
    parser = argparse.ArgumentParser(description="Execute notebooks and save per-cell outputs to logs")
    parser.add_argument('--root', '-r', default='.', help='Root folder to search for notebooks (default .)')
    parser.add_argument('--outdir', '-o', default='./executed_notebooks', help='Where to save executed notebooks')
    parser.add_argument('--logs', '-l', default='./notebook_logs', help='Where to save per-notebook logs')
    parser.add_argument('--timeout', '-t', default=600, type=int, help='Timeout in seconds per cell (default 600)')
    parser.add_argument('--kernel', '-k', default=None, help='Kernel name to use (default: from notebook or python3)')
    parser.add_argument('--no-cell-outputs', action='store_true', help='Do NOT include cell outputs in logs (faster)')
    parser.add_argument('--recursive', action='store_true', help='Search recursively (default yes)')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    logs_dir = Path(args.logs).resolve()
    timeout = args.timeout
    kernel = args.kernel or 'python3'
    include_cell_outputs = not args.no_cell_outputs

    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    notebooks = find_notebooks(root, recursive=False)

    top_logger = setup_logger(logs_dir / "summary.log")
    top_logger.info(f"Found {len(notebooks)} notebooks under {root}")
    top_logger.debug(f"Those notebooks were found {notebooks}")

    if not notebooks:
        top_logger.warning("No notebooks found. Exiting.")
        return

    for nb_path in notebooks:
        rel = nb_path.relative_to(root)
        out_nb_path = outdir / rel.parent / (rel.stem + "_executed.ipynb")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        nb_log_path = logs_dir / rel.parent / f"{rel.stem}_{ts}.log"

        top_logger.info(f"Processing: {nb_path} -> log: {nb_log_path}")
        # Each notebook handled individually; failures logged but do not stop the loop
        success = execute_and_log_single_notebook(nb_path=nb_path, out_nb_path=out_nb_path,
                                                 nb_log_path=nb_log_path, timeout=timeout,
                                                 kernel_name=kernel, include_cell_outputs=include_cell_outputs)
        if not success:
            top_logger.error(f"Notebook failed: {nb_path} (see {nb_log_path})")
        else:
            top_logger.info(f"Notebook finished: {nb_path} (see {nb_log_path})")

    top_logger.info("ALL DONE.")

if __name__ == "__main__":
    main()

    #run exemple
    #python3 run_notebooks.py --root . --outdir ./executed --logs ./logs --timeout 90000000000
