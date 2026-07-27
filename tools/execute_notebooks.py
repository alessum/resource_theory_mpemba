"""Execute and validate publishable notebooks in repository order.

Pass one or more notebook filenames to execute only those files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "atherm_ex1.ipynb",
    "atherm_ex2.ipynb",
    "asymmetry_modes_example.ipynb",
    "asymm_ex1.ipynb",
    "asymm_ex2.ipynb",
    "asymm_ex3.ipynb",
    "asymm_ex4.ipynb",
    "asymm_ex4.1.a.ipynb",
    "asymm_ex5.ipynb",
    "asymm_ex6.ipynb",
    "Mpemba_nonstatioinarity.ipynb",
    "Mpemba_ETH.ipynb",
    "Mpemba_QFI_monotone.ipynb",
]


def main() -> None:
    requested = sys.argv[1:]
    unknown = set(requested) - set(NOTEBOOKS)
    if unknown:
        raise ValueError(f"Unknown notebook filename(s): {sorted(unknown)}")
    selected = requested or NOTEBOOKS
    for filename in selected:
        path = ROOT / filename
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        elapsed = time.perf_counter() - started
        print(f"{filename}: {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
