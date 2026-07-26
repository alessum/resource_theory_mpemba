"""Generate the raw data consumed by the three circuit notebooks."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpemba-matplotlib")
)

import circuit_data as cd


DATA_DIR = ROOT / "data" / "circuit_examples"
REFERENCE_REPOSITORY = "alessum/mpemba_circuits"
REFERENCE_COMMIT = "5042f3600a5b14c93b515e0bd0dab0e8fa4d5509"
REFERENCE_PARAMETER_FILE = "data/U1_rnd_parameters.npy"
REFERENCE_PARAMETER_SHA256 = (
    "9aba292fc2e4f83d8ec7a19fa910a543956053f48b5dd5b31cff7f1fe476d228"
)
REFERENCE_CIRCUITS = np.array([40, 43, 46], dtype=int)
REFERENCE_SAMPLE_TIMES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100],
    dtype=int,
)


def _progress(message: str) -> None:
    print(message, flush=True)


def _save(name: str, dataset: dict) -> None:
    path = cd.save_dataset(DATA_DIR / name, dataset)
    shape = dataset["asymmetry"].shape
    print(f"saved {path.relative_to(ROOT)} with asymmetry{shape}", flush=True)


def _load_reference_parameter_table(checkout: Path) -> np.ndarray:
    path = checkout / REFERENCE_PARAMETER_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Point --reference-checkout at a clone of "
            f"https://github.com/{REFERENCE_REPOSITORY}."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_PARAMETER_SHA256:
        raise ValueError(
            "The reference parameter file does not match the pinned file at "
            f"commit {REFERENCE_COMMIT}: got SHA-256 {digest}."
        )
    table = np.load(path, allow_pickle=False)
    if table.shape != (100, 20, 5):
        raise ValueError(
            f"Expected the reference parameter table to have shape "
            f"(100, 20, 5), got {table.shape}."
        )
    return table


def _reference_dataset(
    checkout: Path | None,
    *,
    paper_scale: bool,
) -> dict:
    stored_path = DATA_DIR / "u1_nonmarkovian_reference.npz"
    if checkout is not None:
        table = _load_reference_parameter_table(checkout)
    elif stored_path.exists() and not paper_scale:
        with np.load(stored_path, allow_pickle=False) as stored:
            selected = stored["gate_parameters"]
        table = np.empty((100, 20, 5))
        table[REFERENCE_CIRCUITS] = selected
    else:
        raise FileNotFoundError(
            "A reference checkout is required. Clone "
            f"https://github.com/{REFERENCE_REPOSITORY}, then pass "
            "--reference-checkout /path/to/mpemba_circuits."
        )

    indices = (
        np.arange(100, dtype=int) if paper_scale else REFERENCE_CIRCUITS
    )
    return cd.generate_u1_nonmarkovian(
        n_realizations=len(indices),
        steps=1_000 if paper_scale else 100,
        n_system=8,
        n_environment=12,
        parameter_divisor=1,
        gate_parameters=table[indices],
        source_indices=indices,
        source_repository=REFERENCE_REPOSITORY,
        source_commit=REFERENCE_COMMIT,
        source_parameter_file=REFERENCE_PARAMETER_FILE,
        source_parameter_sha256=REFERENCE_PARAMETER_SHA256,
        data_level=(
            "full manuscript/reference-repository production run"
            if paper_scale
            else "manuscript system size and reference parameters; "
            "three circuits, first 100 layers at 19 sampled times"
        ),
        sample_times=None if paper_scale else REFERENCE_SAMPLE_TIMES,
        progress=_progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the manuscript circuit protocols and save per-realization "
            "data for the notebooks."
        )
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "u1-markovian",
            "u1-nonmarkovian",
            "u1-reference",
            "su2",
        ),
        default="all",
    )
    parser.add_argument(
        "--reference-checkout",
        type=Path,
        help=(
            "Path to a checkout of alessum/mpemba_circuits at the pinned "
            "reference commit. Required to create the reference U(1) data "
            "from the complete archived parameter table."
        ),
    )
    parser.add_argument(
        "--paper-scale",
        action="store_true",
        help=(
            "Use every paper-scale size and ensemble count. This is an HPC run; "
            "the default already uses the exact Fig. 9 size and exact Fig. 11 "
            "system size."
        ),
    )
    args = parser.parse_args()

    if args.only in ("all", "u1-markovian"):
        _save(
            "u1_markovian.npz",
            cd.generate_u1_markovian(progress=_progress),
        )

    if args.only in ("all", "u1-nonmarkovian"):
        kwargs = {}
        if args.paper_scale:
            _save(
                "u1_nonmarkovian_reference.npz",
                _reference_dataset(
                    args.reference_checkout, paper_scale=True
                ),
            )
        else:
            _save(
                "u1_nonmarkovian.npz",
                cd.generate_u1_nonmarkovian(
                    progress=_progress, **kwargs
                ),
            )

    if args.only == "u1-reference":
        _save(
            "u1_nonmarkovian_reference.npz",
            _reference_dataset(
                args.reference_checkout, paper_scale=args.paper_scale
            ),
        )

    if args.only in ("all", "su2"):
        kwargs = (
            {"n_realizations": 100, "coefficient_order": "figure"}
            if args.paper_scale
            else {}
        )
        _save(
            "su2_nonmarkovian.npz",
            cd.generate_su2_nonmarkovian(progress=_progress, **kwargs),
        )


if __name__ == "__main__":
    main()
