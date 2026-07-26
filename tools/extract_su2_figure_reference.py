#!/usr/bin/env python3
"""Extract the five vector curves from the published Fig. 11 source PDF.

The archived figure was saved by Adobe Illustrator with each curve expanded
into circular vector elements.  This script converts that PDF to SVG, identifies
the five 1,001-point colour groups, and maps their vertical coordinates back to
the published y-axis.  The result is a *figure reference*, not raw simulation
data, and is labelled as such in the CSV metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


THETA_BY_DARKNESS = (0.50, 0.45, 0.40, 0.35, 0.30)
N_POINTS = 1_001

# Page coordinates of the plotting box in the Illustrator PDF.  The y limits
# are visible axis limits, not fitted to the curves.
PLOT_TOP = 0.5117646570099623
PLOT_BOTTOM = 270.343753489702
Y_MAX = 2.3
Y_MIN = 1.3

PATH_RE = re.compile(
    r'<path[^>]*fill="rgb\(([^"]+)\)"[^>]*d="([^"]+)"'
)
NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colour_luminance(colour: str) -> float:
    components = [
        float(value.rstrip("%")) / 100
        for value in colour.split(",")
    ]
    return (
        0.2126 * components[0]
        + 0.7152 * components[1]
        + 0.0722 * components[2]
    )


def path_centres(svg_text: str) -> dict[str, list[float]]:
    centres: dict[str, list[float]] = defaultdict(list)
    for colour, path_data in PATH_RE.findall(svg_text):
        if not path_data.startswith("M "):
            continue
        numbers = [float(value) for value in NUMBER_RE.findall(path_data)]
        coordinates = list(zip(numbers[0::2], numbers[1::2]))
        if not coordinates:
            continue
        y_values = [coordinate[1] for coordinate in coordinates]
        centres[colour].append((min(y_values) + max(y_values)) / 2)
    return {
        colour: values
        for colour, values in centres.items()
        if len(values) == N_POINTS
    }


def y_value(page_y: float) -> float:
    fraction = (page_y - PLOT_TOP) / (PLOT_BOTTOM - PLOT_TOP)
    return Y_MAX - fraction * (Y_MAX - Y_MIN)


def find_pdftocairo() -> str:
    """Locate Poppler without assuming Homebrew is on the shell ``PATH``."""
    discovered = shutil.which("pdftocairo")
    if discovered is not None:
        return discovered
    for candidate in (
        Path("/opt/homebrew/bin/pdftocairo"),
        Path("/usr/local/bin/pdftocairo"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "pdftocairo is required. Install Poppler and make pdftocairo "
        "available on PATH."
    )


def main() -> None:
    args = parse_args()
    if not args.input_pdf.exists():
        raise FileNotFoundError(args.input_pdf)

    with tempfile.TemporaryDirectory(prefix="su2-figure-") as directory:
        svg_path = Path(directory) / "figure.svg"
        subprocess.run(
            [
                find_pdftocairo(),
                "-svg",
                str(args.input_pdf),
                str(svg_path),
            ],
            check=True,
        )
        groups = path_centres(svg_path.read_text(encoding="utf-8"))

    if len(groups) != len(THETA_BY_DARKNESS):
        counts = sorted(len(values) for values in groups.values())
        raise RuntimeError(
            "Expected five 1,001-point colour groups; "
            f"found {len(groups)} with counts {counts}."
        )

    ordered_groups = sorted(
        groups.items(), key=lambda item: colour_luminance(item[0])
    )
    curves = {
        theta: [y_value(value) for value in values]
        for theta, (_, values) in zip(
            THETA_BY_DARKNESS, ordered_groups
        )
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# data_kind=vectorized_published_figure_reference\n")
        handle.write("# manuscript_figure=Fig. 11\n")
        handle.write(f"# source_sha256={sha256(args.input_pdf)}\n")
        handle.write("# y_axis_calibration=[1.3,2.3]\n")
        handle.write(
            "# displayed_time_grid=0.1,0.2,...,100.1; "
            "the manuscript does not document its conversion to Floquet layers\n"
        )
        writer = csv.writer(handle)
        theta_columns = [f"theta_{theta:.2f}_pi" for theta in sorted(curves)]
        writer.writerow(["displayed_time", *theta_columns])
        for index in range(N_POINTS):
            writer.writerow(
                [
                    f"{(index + 1) / 10:.1f}",
                    *[
                        f"{curves[theta][index]:.12f}"
                        for theta in sorted(curves)
                    ],
                ]
            )

    print(f"Wrote {args.output_csv} from {args.input_pdf}")


if __name__ == "__main__":
    main()
