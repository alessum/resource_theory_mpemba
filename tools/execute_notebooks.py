"""Execute and validate publishable notebooks without rewriting them.

Pass ``--write`` to refresh stored outputs intentionally.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
# Paper reading order (see NOTEBOOKS.md). Each trailing comment gives the
# manuscript location so the sequence maps 1:1 onto the paper.
NOTEBOOKS = [
    "atherm_ex1.ipynb",           # 1  Sec. II B,    Fig. 2
    "atherm_ex2.ipynb",           # 2  Sec. II C,    Fig. 3
    "asymm_modes_example.ipynb",  # 3  Sec. III C,   Fig. 4
    "asymm_ex1.ipynb",            # 4  Sec. III D,   Fig. 5
    "asymm_ex2.ipynb",            # 5  Sec. III E,   Fig. 6
    "asymm_ex3.ipynb",            # 6  Sec. III F,   Fig. 7
    "asymm_ex4.ipynb",            # 7  Sec. III G 1, Fig. 9
    "asymm_ex4.1.a.ipynb",        # 8  Sec. III G 1, Fig. 10
    "asymm_ex5.ipynb",            # 9  Sec. III G 2, Fig. 11
    "asymm_ex6.ipynb",            # 10 Sec. IV,      Figs. 12-13
    "non-stationarity.ipynb",     # 11 Appendix A,   Fig. S1  / pub. Fig. 14
    "atherm_ETH.ipynb",           # 12 Appendix B,   Fig. S2  / pub. Fig. 15
    "quantum-fisher-info.ipynb",  # 13 Appendix D,   Fig. S3  / pub. Fig. 17
]


def main() -> None:
    requested = sys.argv[1:]
    write = "--write" in requested
    requested = [name for name in requested if name != "--write"]
    unknown = set(requested) - set(NOTEBOOKS)
    if unknown:
        raise ValueError(f"Unknown notebook filename(s): {sorted(unknown)}")
    selected = requested or NOTEBOOKS
    for filename in selected:
        path = NOTEBOOKS_DIR / filename
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
        if write:
            nbformat.write(notebook, path)
        elapsed = time.perf_counter() - started
        print(f"{filename}: {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
