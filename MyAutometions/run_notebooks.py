#!/usr/bin/env python3
"""
Run all .ipynb notebooks under a root directory, save executed notebooks
and write per-notebook logs. If a notebook fails the script will continue
with the next notebook (try/except per-notebook).
"""

import argparse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback
from pathlib import Path
from datetime import datetime
import logging
import sys

def setup_logger(log_file_path: Path):
    """Create a logger that writes to `log_file_path` and stdout."""
    logger = logging.getLogger(str(log_file_path))
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def run_notebook(nb_path: Path, out_nb_path: Path, log_path: Path, timeout: int, kernel_name: str):
    """
    Execute a single notebook with ExecutePreprocessor, save executed notebook and log results.
    """
    logger = setup_logger(log_path)
    logger.info(f"STARTING notebook: {nb_path}")
    start_time = datetime.utcnow()

    try:
        nb = nbformat.read(nb_path, as_version=4)
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

        # Execute notebook in its own working directory to allow relative file paths
        exec_dir = nb_path.parent.resolve()
        logger.debug(f"Executing with timeout={timeout}s kernel='{kernel_name}' in {exec_dir}")

        # Execute
        ep.preprocess(nb, {'metadata': {'path': str(exec_dir)}})

        # Write executed notebook
        out_nb_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_nb_path)
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"SUCCESS: {nb_path} executed in {elapsed:.1f}s -> {out_nb_path}")

    except Exception as e:
        # Save partial notebook (if nb exists) for debugging
        err_tb = traceback.format_exc()
        logger.error(f"FAILED: {nb_path}\nException: {e}\nTraceback:\n{err_tb}")

        try:
            # If nb exists in local scope, attempt to write partial output
            if 'nb' in locals():
                out_nb_path.parent.mkdir(parents=True, exist_ok=True)
                nbformat.write(nb, out_nb_path)
                logger.info(f"Partial/failed notebook saved to: {out_nb_path}")
        except Exception as save_err:
            logger.error(f"Failed to save partial notebook: {save_err}")

def find_notebooks(root: Path, recursive: bool = False):
    """Yield .ipynb file paths under root. Sorted for deterministic order."""
    if recursive:
        notebooks = sorted(root.rglob('*.ipynb'))
    else:
        notebooks = sorted(root.glob('*.ipynb'))
    return notebooks

def main():
    p = argparse.ArgumentParser(description="Execute .ipynb files and keep logs.")
    p.add_argument('--root', '-r', type=str, default='.', help='Root directory to search for notebooks.')
    p.add_argument('--outdir', '-o', type=str, default='./executed_notebooks',
                   help='Directory to write executed notebooks to (preserves relative paths).')
    p.add_argument('--logs', '-l', type=str, default='./notebook_logs',
                   help='Directory to write per-notebook logs.')
    p.add_argument('--timeout', '-t', type=int, default=600,
                   help='Timeout in seconds per notebook cell (default 600).')
    p.add_argument('--kernel', '-k', type=str, default=None,
                   help='Kernel name to use (default: notebook-specified or python3).')
    p.add_argument('--recursive', action='store_true', help='Search recursively (default yes).')
    args = p.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    logs_dir = Path(args.logs).resolve()
    timeout = args.timeout
    kernel_name = args.kernel

    # Create directories
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    notebooks = find_notebooks(root, recursive=False)
    print(notebooks)

    if not notebooks:
        print(f"No notebooks found under {root}")
        return

    # Top-level summary logger
    top_log_path = logs_dir / 'summary.log'
    top_logger = setup_logger(top_log_path)
    top_logger.info(f"Found {len(notebooks)} notebooks under {root}")

    for nb_path in notebooks:
        # Create a relative path for outputs to mirror input tree
        rel = nb_path.relative_to(root)
        out_nb_path = outdir / rel.parent / (rel.stem + '_executed.ipynb')

        # Log file per notebook (timestamped)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        nb_log_path = logs_dir / rel.parent / f"{rel.stem}_{ts}.log"
        nb_log_path.parent.mkdir(parents=True, exist_ok=True)

        top_logger.info(f"Processing: {nb_path} -> log: {nb_log_path.name}")
        # Run with try/catch internally; failures are handled inside run_notebook
        run_notebook(nb_path=nb_path, out_nb_path=out_nb_path, log_path=nb_log_path,
                     timeout=timeout, kernel_name=kernel_name or 'python3')

    top_logger.info("ALL DONE.")

if __name__ == '__main__':
    main()
    #run exemple
    #python3 run_notebooks.py --root . --outdir ./executed --logs ./logs --timeout 90000000000
