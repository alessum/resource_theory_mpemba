"""Build the data-backed notebooks for manuscript Figs. 9--11."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def _source(text: str) -> str:
    return dedent(text).strip() + "\n"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def write_notebook(filename: str, cells: list) -> None:
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:03d}")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / filename).write_text(
        json.dumps(notebook, indent=1) + "\n",
        encoding="utf-8",
    )


COMMON_IMPORTS = r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_TOOLS_DIR = (Path.cwd() / "tools").resolve()
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import circuit_data as cd
import notebook_utils as nu

np.set_printoptions(precision=5, suppress=True)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (7.2, 4.2),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 11,
})

BLUE = "#394F87"
ORANGE = "#E8792E"
GOLD = "#F4C95D"
PALETTE_3 = [BLUE, ORANGE, GOLD]
PALETTE_5 = plt.cm.Oranges(np.linspace(0.35, 0.95, 5))


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}
"""


def build_u1_markovian() -> None:
    write_notebook(
        "asymm_ex4.ipynb",
        [
            md(
                r"""
                # Random circuits I: Markovian U(1) symmetry restoration

                **Manuscript map.** Sec. III G 1, Fig. 9, Eqs. (68)-(77).

                This notebook is a data analysis, not a schematic imitation. The
                stored file contains 100 independently generated brickwork circuits,
                all three initial states, all 201 time samples, every
                charge-resolved channel eigenvalue, the slow decay rates, and the
                initial slow-mode overlaps. The calculation uses the gate
                implementation in `functions.py` and the reset channel in
                `circuit_data.py`.

                We first inspect raw realizations and only then form the four
                ensemble diagnostics shown in Fig. 9.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Generate or load the raw data

                A complete rerun takes only a few seconds on this system. Leave
                `REGENERATE_DATA=False` when reading the notebook with its stored,
                already executed results; switch it to `True` to overwrite the NPZ
                with the deterministic 100-circuit run.
                """
            ),
            code(
                r"""
                DATA_PATH = Path("data/circuit_examples/u1_markovian.npz")
                REGENERATE_DATA = False

                if REGENERATE_DATA:
                    generated = cd.generate_u1_markovian()
                    cd.save_dataset(DATA_PATH, generated)

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"{DATA_PATH} is missing. Run "
                        "`python tools/generate_circuit_data.py "
                        "--only u1-markovian`."
                    )
                data = load_npz(DATA_PATH)

                asymmetry = data["asymmetry"]
                times = data["times"]
                theta_over_pi = data["theta_over_pi"]
                block_sizes = data["block_sizes"]
                labels = [
                    fr"$\theta={theta:.1f}\pi,\ b={block}$"
                    for theta, block in zip(theta_over_pi, block_sizes)
                ]

                print(data["protocol"].item())
                print(data["data_level"].item())
                print(
                    f"Ns={data['n_system'].item()}, "
                    f"Ne={data['n_environment'].item()}, "
                    f"realizations={data['n_realizations'].item()}"
                )
                print("raw asymmetry array:", asymmetry.shape)
                """
            ),
            md(
                r"""
                ## 2. Protocol and selected U(1) modes

                The system begins in the block-rotated states

                $$
                |\varphi(\theta,b)\rangle =
                \bigotimes_n e^{-i\theta Y^{\otimes b}/2}|0\rangle^{\otimes N_s},
                $$

                with $(\theta/\pi,b)=(0.1,1),(0.2,2),(0.5,3)$. A block of
                size $b$ creates coherences only in charge-difference sectors that
                are multiples of $b$. Each circuit realization defines one fixed
                U(1)-covariant channel. Its two-qubit environment is maximally mixed
                and reset after every Floquet layer, so the same channel is applied
                repeatedly.
                """
            ),
            code(
                r"""
                occupied_modes = []
                for theta, block_size in zip(theta_over_pi, block_sizes):
                    ket = nu.multi_spin_tilted_state(
                        int(data["n_system"]), theta * np.pi, int(block_size)
                    )
                    rho = np.outer(ket, ket.conj())
                    norms = nu.mode_trace_norms_u1(
                        rho, int(data["n_system"])
                    )
                    support = [
                        charge for charge, norm in norms.items()
                        if norm > 1e-10
                    ]
                    occupied_modes.append(support)
                    print(
                        f"b={block_size}: occupied charge differences {support}"
                    )

                assert np.isfinite(asymmetry).all()
                assert asymmetry.shape == (
                    int(data["n_realizations"]), 3, len(times)
                )
                # A reset covariant channel must make the resource monotone.
                assert np.diff(asymmetry, axis=-1).max() < 1e-10
                """
            ),
            md(
                r"""
                ## 3. Look at individual runs before averaging

                Thin lines below are actual circuit realizations. Their different
                decay speeds are the statistical object being averaged; they are not
                generated from a fitted exponential.
                """
            ),
            code(
                r"""
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
                for state_index, (ax, label, color) in enumerate(
                    zip(axes, labels, PALETTE_3)
                ):
                    for realization in range(12):
                        ax.semilogy(
                            times,
                            np.maximum(
                                asymmetry[realization, state_index], 1e-14
                            ),
                            color=color,
                            alpha=0.20,
                            lw=0.8,
                        )
                    ax.semilogy(
                        times,
                        np.maximum(
                            asymmetry[:, state_index].mean(axis=0), 1e-14
                        ),
                        color=color,
                        lw=2.2,
                    )
                    ax.set(title=label, xlabel="Floquet layer")
                axes[0].set_ylabel(r"$M[\rho_s(t)]$")
                fig.suptitle("Twelve raw channels and their ensemble mean")
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 4. Ensemble reduction and Mpemba times

                Bands use the 16th and 84th percentiles of the 100 raw values at
                each time. A crossing time is linearly interpolated between the two
                neighboring sampled layers.
                """
            ),
            code(
                r"""
                mean_curves = asymmetry.mean(axis=0)
                lower_curves, upper_curves = np.quantile(
                    asymmetry, [0.16, 0.84], axis=0
                )
                mean_rates = data["slowest_log_modulus"].mean(axis=0)
                std_rates = data["slowest_log_modulus"].std(axis=0)
                mean_overlaps = data["slow_mode_overlap"].mean(axis=0)
                std_overlaps = data["slow_mode_overlap"].std(axis=0)

                for faster in (1, 2):
                    tau = nu.crossing_time(
                        times, mean_curves[faster], mean_curves[0]
                    )
                    print(
                        f"{labels[faster]} crosses {labels[0]} at "
                        f"t={tau:.2f}"
                    )
                    assert np.isfinite(tau)

                assert np.all(np.diff(mean_rates) < 0)
                """
            ),
            code(
                r"""
                fig, axes = plt.subplots(2, 2, figsize=(11, 7.7))

                for mean, lower, upper, label, color in zip(
                    mean_curves,
                    lower_curves,
                    upper_curves,
                    labels,
                    PALETTE_3,
                ):
                    axes[0, 0].semilogy(
                        times, np.maximum(mean, 1e-14),
                        color=color, lw=2, label=label
                    )
                    axes[0, 0].fill_between(
                        times,
                        np.maximum(lower, 1e-14),
                        np.maximum(upper, 1e-14),
                        color=color,
                        alpha=0.16,
                    )
                axes[0, 0].set(
                    xlabel="Floquet layer",
                    ylabel=r"$M[\rho_s(t)]$",
                    title="(a) asymmetry: mean and 16-84% interval",
                )
                axes[0, 0].legend(fontsize=8)

                spectrum = data["spectrum"]
                spectrum_charge = data["spectrum_charge"]
                for charge in range(-int(data["n_system"]),
                                    int(data["n_system"]) + 1):
                    values = spectrum[:, spectrum_charge == charge].ravel()
                    axes[0, 1].scatter(
                        values.real,
                        values.imag,
                        s=3,
                        alpha=0.10,
                        color=plt.cm.Oranges(
                            abs(charge) / int(data["n_system"])
                        ),
                    )
                axes[0, 1].set(
                    xlabel=r"$\operatorname{Re}\eta$",
                    ylabel=r"$\operatorname{Im}\eta$",
                    title="(b) all channel eigenvalues from 100 runs",
                )
                axes[0, 1].axhline(0, color="0.65", lw=0.7)

                charges = np.arange(int(data["n_system"]) + 1)
                axes[1, 0].errorbar(
                    charges,
                    mean_rates,
                    yerr=std_rates,
                    color="0.25",
                    marker="o",
                    capsize=3,
                )
                axes[1, 0].set(
                    xlabel=r"$|\mu|$",
                    ylabel=r"$\langle\log|\eta_\mu|\rangle$",
                    title="(c) slowest nonstationary decay rate",
                )

                for state_index, (label, color) in enumerate(
                    zip(labels, PALETTE_3)
                ):
                    axes[1, 1].errorbar(
                        charges,
                        mean_overlaps[state_index],
                        yerr=std_overlaps[state_index],
                        color=color,
                        marker="o",
                        capsize=2,
                        label=label,
                    )
                axes[1, 1].set(
                    xlabel=r"$|\mu|$",
                    ylabel=r"$|\operatorname{Tr}(\ell_\mu^\dagger\rho_0)|$",
                    title="(d) overlap with the slowest sector mode",
                )
                axes[1, 1].legend(fontsize=8)

                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## Takeaway

                The four panels are different views of the same simulated
                ensemble. Block preparation removes entire low-charge sectors from
                the initial state. Since the measured slow decay exponent becomes
                more negative with $|\mu|$, the initially more asymmetric
                high-block states overtake the $b=1$ state. No fitted or illustrative
                curve enters this conclusion.
                """
            ),
        ],
    )


def build_u1_nonmarkovian() -> None:
    write_notebook(
        "asymm_ex4.1.a.ipynb",
        [
            md(
                r"""
                # Random circuits II: non-Markovian U(1) data

                **Manuscript map.** Sec. III G 1, Fig. 10, Eq. (78).

                Here the environment is a finite quantum memory. It starts in
                $|0\rangle^{\otimes N_e}$ and is never reset, so information and
                asymmetry can return to the system.

                This walkthrough uses two complementary **simulated** datasets:

                1. a reference run at the manuscript Hilbert-space size
                   $N_s=8,N_e=12$, driven by archived gate parameters from
                   [`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits);
                2. a longer $N_s=4,N_e=8$ ensemble used to estimate crossings and
                   fluctuations locally.

                The first checks fidelity to the production code. The second gives
                enough realizations and time samples for honest ensemble
                statistics. Neither is presented as the complete 100-circuit,
                1000-layer Fig. 10 array.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Load the raw arrays and audit their provenance

                Both files store every monotone value and every gate parameter;
                the notebook performs the reductions itself. The exact-size file
                embeds circuits 40, 43, and 46 selected by the reference
                repository's launcher.

                Set `REGENERATE_SCALED=True` to rerun the local 24-circuit
                ensemble. Recreating the exact-size file from its upstream
                parameter table is an explicit command shown below the audit.
                """
            ),
            code(
                r"""
                REFERENCE_PATH = Path(
                    "data/circuit_examples/u1_nonmarkovian_reference.npz"
                )
                SCALED_PATH = Path(
                    "data/circuit_examples/u1_nonmarkovian.npz"
                )
                REGENERATE_SCALED = False

                if REGENERATE_SCALED:
                    generated = cd.generate_u1_nonmarkovian()
                    cd.save_dataset(SCALED_PATH, generated)

                for path in (REFERENCE_PATH, SCALED_PATH):
                    if not path.exists():
                        raise FileNotFoundError(
                            f"{path} is missing; see "
                            "data/circuit_examples/README.md."
                        )

                reference = load_npz(REFERENCE_PATH)
                scaled = load_npz(SCALED_PATH)

                reference_asymmetry = reference["asymmetry"]
                reference_times = reference["times"]
                asymmetry = scaled["asymmetry"]
                times = scaled["times"]
                theta_over_pi = scaled["theta_over_pi"]
                labels = [fr"$\theta={theta:.2f}\pi$" for theta in theta_over_pi]
                # Fig. 10 uses dark orange for the smallest tilt and light
                # orange for the largest one.
                theta_colors = PALETTE_5[::-1]

                print(
                    "reference source:",
                    reference["source_repository"].item(),
                )
                print(
                    "commit:", reference["source_commit"].item()
                )
                print(
                    "parameter SHA-256:",
                    reference["source_parameter_sha256"].item(),
                )
                print(
                    "reference circuits:",
                    reference["source_indices"].tolist(),
                )
                print(
                    "exact-size raw array:", reference_asymmetry.shape,
                    f"(Ns={reference['n_system'].item()}, "
                    f"Ne={reference['n_environment'].item()})",
                )
                print(
                    "scaled raw array:", asymmetry.shape,
                    f"(Ns={scaled['n_system'].item()}, "
                    f"Ne={scaled['n_environment'].item()})",
                )

                assert asymmetry.shape == (
                    int(scaled["n_realizations"]), 5, len(times)
                )
                assert reference_asymmetry.shape == (
                    int(reference["n_realizations"]),
                    5,
                    len(reference_times),
                )
                assert np.isfinite(asymmetry).all()
                assert np.isfinite(reference_asymmetry).all()
                assert int(reference["n_system"]) == 8
                assert int(reference["n_environment"]) == 12
                assert reference["source_commit"].item() == (
                    "5042f3600a5b14c93b515e0bd0dab0e8fa4d5509"
                )
                """
            ),
            code(
                r"""
                # Reproduction command for the exact-size reference file:
                #
                # python tools/generate_circuit_data.py \
                #   --only u1-reference \
                #   --reference-checkout /path/to/mpemba_circuits
                #
                # The generator verifies the upstream parameter file's SHA-256
                # before starting the expensive state-vector evolution.
                """
            ),
            md(
                r"""
                ## 2. What is evolved?

                The system state is the homogeneous product tilt

                $$
                |\varphi(\theta)\rangle =
                \bigotimes_{n=1}^{N_s} e^{-i\theta s_y^{(n)}}|0\rangle_n.
                $$

                Each realization samples one U(1)-symmetric brickwork layer from
                Eqs. (71)-(72) and repeatedly applies it to the *joint* pure state.
                The reduced density matrix is constructed from the resulting
                state-vector amplitudes at every time. Thus the plotted values are
                $S(\mathcal G_{\rm U(1)}[\rho_s])-S(\rho_s)$ evaluated on actual
                reduced states.

                For the exact-size file, the gate parameters are not resampled:
                they are rows of `data/U1_rnd_parameters.npy` in the linked
                reference repository. The optimized layer implementation is
                checked against the repository's `functions.apply_U` convention
                before every generation run.
                """
            ),
            md(
                r"""
                ## 3. Exact-size reference trajectories

                Each panel below is one complete $2^{20}$-dimensional
                state-vector evolution. These are direct diagnostics of the
                production setup, not fitted or illustrative curves. Three
                circuits are too few to estimate the Fig. 10 ensemble mean, so no
                uncertainty band or crossing claim is extracted from them.
                """
            ),
            code(
                r"""
                fig, axes = plt.subplots(
                    1,
                    reference_asymmetry.shape[0],
                    figsize=(14, 3.8),
                    sharex=True,
                    sharey=True,
                )
                for realization, ax in enumerate(np.atleast_1d(axes)):
                    for state_index, (label, color) in enumerate(
                        zip(labels, theta_colors)
                    ):
                        ax.plot(
                            reference_times[1:],
                            reference_asymmetry[
                                realization, state_index, 1:
                            ],
                            color=color,
                            lw=1.3,
                            label=label,
                        )
                    ax.set(
                        xscale="log",
                        xlabel="Floquet layer",
                        title=(
                            "reference circuit "
                            f"{reference['source_indices'][realization]}"
                        ),
                    )
                axes[0].set_ylabel(r"$M_{\rm U(1)}[\rho_s(t)]$")
                axes[-1].legend(fontsize=7)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 4. Reproducible ensemble statistics

                We now switch to the longer local ensemble. The left panel exposes
                six individual circuits; the right panel computes the mean and
                16th–84th percentile interval from all 24 realizations. The reduced
                Hilbert space changes finite-size details, so this block validates
                the mechanism and analysis—not the exact numerical curve in
                Fig. 10.
                """
            ),
            code(
                r"""
                mean_curves = asymmetry.mean(axis=0)
                lower_curves, upper_curves = np.quantile(
                    asymmetry, [0.16, 0.84], axis=0
                )

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
                for state_index, (label, color) in enumerate(
                    zip(labels, theta_colors)
                ):
                    for realization in range(6):
                        axes[0].plot(
                            times[1:],
                            asymmetry[realization, state_index, 1:],
                            color=color,
                            alpha=0.18,
                            lw=0.8,
                        )
                    axes[1].plot(
                        times[1:],
                        mean_curves[state_index, 1:],
                        color=color,
                        lw=2,
                        label=label,
                    )
                    axes[1].fill_between(
                        times[1:],
                        lower_curves[state_index, 1:],
                        upper_curves[state_index, 1:],
                        color=color,
                        alpha=0.15,
                    )

                for ax in axes:
                    ax.set_xscale("log")
                    ax.set_xlabel("Floquet layer")
                axes[0].set(
                    ylabel=r"$M_{\rm U(1)}[\rho_s(t)]$",
                    title="six raw realizations per tilt",
                )
                axes[1].set(title="mean and 16-84% interval")
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 5. Crossings and information backflow

                A positive one-step increment is impossible in the reset Markovian
                protocol, but allowed here because the environment retains memory.
                We report crossings from the mean curves and visualize how often a
                positive increment occurs in the raw ensemble.
                """
            ),
            code(
                r"""
                crossing_times = []
                for state_index in range(1, len(theta_over_pi)):
                    tau = nu.crossing_time(
                        times,
                        mean_curves[state_index],
                        mean_curves[0],
                    )
                    crossing_times.append(tau)
                    print(
                        f"{labels[state_index]} crosses {labels[0]} "
                        f"at t={tau:.2f}"
                    )
                assert np.isfinite(crossing_times).all()

                increments = np.diff(asymmetry, axis=-1)
                revival_fraction = (increments > 1e-10).mean(axis=(0, 1))
                print(
                    "largest positive one-step change in the raw data:",
                    increments.max(),
                )
                print(
                    "fraction of all raw steps with a revival:",
                    (increments > 1e-10).mean(),
                )
                assert increments.max() > 0
                """
            ),
            code(
                r"""
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

                for state_index, (label, color) in enumerate(
                    zip(labels, theta_colors)
                ):
                    axes[0].plot(
                        times[1:],
                        np.diff(mean_curves[state_index]),
                        color=color,
                        lw=1.2,
                        label=label,
                    )
                axes[0].axhline(0, color="0.35", lw=0.8)
                axes[0].set(
                    xscale="log",
                    xlabel="Floquet layer",
                    ylabel=r"$\Delta M$",
                    title="increments of the mean curves",
                )

                axes[1].plot(
                    times[1:], revival_fraction, color=ORANGE, lw=1.6
                )
                axes[1].set(
                    xscale="log",
                    ylim=(-0.02, 1.02),
                    xlabel="Floquet layer",
                    ylabel="fraction with positive increment",
                    title="raw-data evidence of backflow",
                )
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## Takeaway

                The reference trajectories verify the manuscript-size circuit and
                upstream parameter conventions. The independent 24-circuit run
                then supplies enough raw data to expose the Mpemba crossings and
                non-Markovian revivals. Unlike the reset channel, a finite
                environment can return asymmetry to the system.

                Reproducing the full Fig. 10 average requires all 100 archived
                circuits for 1000 layers:
                `tools/generate_circuit_data.py --only u1-reference
                --paper-scale --reference-checkout ...`. That is an HPC-scale
                calculation and is deliberately not disguised by interpolation or
                synthetic curves here.
                """
            ),
        ],
    )


def build_su2_nonmarkovian() -> None:
    write_notebook(
        "asymm_ex5.ipynb",
        [
            md(
                r"""
                # Example 5 audit: SU(2)-covariant open-system dynamics

                **Manuscript map.** Sec. III G 2, Fig. 11, Eq. (79).

                This is a physics audit of the non-Abelian random-circuit example,
                not a cosmetic reproduction of the published plot. It checks four
                logically separate ingredients:

                1. the partial-SWAP gates and singlet environment are SU(2)
                   invariant;
                2. the Haar twirl is the exact Schur-Weyl conditional expectation;
                3. the stored $N_s=8,N_e=12$ trajectories obey the analytic
                   $t=0$ asymmetry and the global covariance bound;
                4. the independent vector data in Fig. 11 are compatible with the
                   printed state and monotone.

                They are not. The earlier notebook incorrectly treated the mismatch
                as a complementary-angle convention. The figure itself does not
                expose $t=0$, its displayed time is not documented in Floquet-layer
                units, and its values cannot be identified with the printed
                Eq. (79) without extra, unavailable provenance. This notebook keeps
                the equation-consistent simulation and the digitized figure
                reference separate.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Load raw trajectories and their provenance

                Two different data products are loaded:

                - `su2_nonmarkovian.npz` contains three raw, equation-consistent
                  trajectories at the manuscript Hilbert-space size. It validates
                  the implementation but is not a Fig. 11 reproduction.
                - `su2_fig11_vector_reference.csv` is extracted from the original
                  vector figure. It is useful for a consistency audit, but it is not
                  raw simulation output and contains no circuit parameters.

                `REGENERATE_DATA=True` recreates the reduced Eq. (79) validation
                run. A 100-realization run is expensive and still cannot recover
                undocumented choices in the published figure.
                """
            ),
            code(
                r"""
                DATA_PATH = Path("data/circuit_examples/su2_nonmarkovian.npz")
                FIGURE_REFERENCE_PATH = Path(
                    "data/circuit_examples/su2_fig11_vector_reference.csv"
                )
                REGENERATE_DATA = False

                if REGENERATE_DATA:
                    generated = cd.generate_su2_nonmarkovian()
                    cd.save_dataset(DATA_PATH, generated)

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"{DATA_PATH} is missing. Run "
                        "`python tools/generate_circuit_data.py --only su2`."
                    )
                if not FIGURE_REFERENCE_PATH.exists():
                    raise FileNotFoundError(FIGURE_REFERENCE_PATH)
                data = load_npz(DATA_PATH)
                figure_reference = np.genfromtxt(
                    FIGURE_REFERENCE_PATH,
                    delimiter=",",
                    names=True,
                    comments="#",
                    skip_header=5,
                )
                asymmetry = data["asymmetry"]
                times = data["times"]
                theta_over_pi = data["theta_over_pi"]
                labels = [fr"$\theta={theta:.2f}\pi$" for theta in theta_over_pi]
                n_system = int(data["n_system"])
                n_environment = int(data["n_environment"])
                n_realizations = int(data["n_realizations"])
                stored_convention = data.get(
                    "coefficient_order", np.array("legacy/unspecified")
                ).item()

                print(data["protocol"].item())
                print(data["data_level"].item())
                print(
                    f"Ns={n_system}, Ne={n_environment}, "
                    f"R={n_realizations}, "
                    f"T={times[-1]}"
                )
                print("raw asymmetry array:", asymmetry.shape)
                print("stored convention metadata:", stored_convention)
                print("environment:", data["environment"].item())
                print(
                    "published vector samples:",
                    len(figure_reference),
                    "(reference only)",
                )

                assert asymmetry.shape == (
                    n_realizations, len(theta_over_pi), len(times)
                )
                assert np.isfinite(asymmetry).all()
                assert np.array_equal(times, np.arange(times[-1] + 1))
                """
            ),
            md(
                r"""
                ## 2. Gate symmetry and the open-system structure

                With spin operators $s^\alpha=\sigma^\alpha/2$, setting
                $J=J_z$ in Eq. (72) gives the isotropic exchange

                $$
                H_{12}=J\,\mathbf s_1\!\cdot\!\mathbf s_2,\qquad
                e^{-iH_{12}}
                =e^{iJ/4}\left[
                  \cos(J/2)I-i\sin(J/2)\operatorname{SWAP}
                \right].
                $$

                The irrelevant global phase leaves the partial-SWAP gate used in
                the code. Such a gate commutes with $R\otimes R$ for every
                $R\in{\rm SU}(2)$. Equivalently it commutes with all three
                collective two-spin generators
                $s^\alpha\otimes I+I\otimes s^\alpha$.

                The caption's randomness is **quenched Floquet disorder**: for
                each realization, one coupling
                $J_b\sim{\rm Uniform}[-\pi/5,\pi/5]$ is drawn for every bond gate
                in $\hat U$; that same $\hat U$ is repeated at every circuit
                layer. The gates are not redrawn in time. Modulo a global phase,
                the singlet/triplet decomposition has only one relative phase, so
                the partial-SWAP family is the general two-qubit
                SU(2)-invariant unitary family. The narrow coupling interval is a
                particular random distribution on that family, not Haar measure
                over the full relative-phase circle.

                A product of singlets is invariant, so tracing out the environment
                defines an SU(2)-covariant map from the initial time to every $t$.

                The same finite environment is reused. Consequently the family of
                reduced maps need not be CP divisible: $M(t+\Delta t)$ may exceed
                $M(t)$ even though covariance still requires $M(t)\leq M(0)$.
                """
            ),
            code(
                r"""
                from scipy.linalg import expm

                coupling = 0.371
                exchange_hamiltonian = coupling * sum(
                    np.kron(pauli / 2, pauli / 2)
                    for pauli in (nu.X, nu.Y, nu.Z)
                )
                exchange_gate = expm(-1j * exchange_hamiltonian)
                partial_swap = nu.su2_gate(coupling)
                assert np.allclose(
                    exchange_gate,
                    np.exp(1j * coupling / 4) * partial_swap,
                    atol=1e-12,
                )

                axis = np.array([0.2, -0.3, 0.7])
                axis /= np.linalg.norm(axis)
                rotation = expm(
                    -0.5j
                    * 0.83
                    * sum(component * pauli for component, pauli in zip(
                        axis, (nu.X, nu.Y, nu.Z)
                    ))
                )
                collective_two = np.kron(rotation, rotation)
                gate_commutator = (
                    partial_swap @ collective_two
                    - collective_two @ partial_swap
                )
                collective_generators = [
                    np.kron(pauli / 2, nu.I2)
                    + np.kron(nu.I2, pauli / 2)
                    for pauli in (nu.X, nu.Y, nu.Z)
                ]
                generator_commutators = [
                    np.linalg.norm(
                        partial_swap @ generator
                        - generator @ partial_swap
                    )
                    for generator in collective_generators
                ]
                singlet = nu.singlet_product(2)

                print("||[u, R x R]|| =", np.linalg.norm(gate_commutator))
                print(
                    "||[u, Sx]||, ||[u, Sy]||, ||[u, Sz]|| =",
                    generator_commutators,
                )
                print(
                    "singlet invariance error =",
                    np.linalg.norm(collective_two @ singlet - singlet),
                )
                assert np.linalg.norm(gate_commutator) < 1e-12
                assert max(generator_commutators) < 1e-12
                assert np.linalg.norm(collective_two @ singlet - singlet) < 1e-12
                """
            ),
            md(
                r"""
                ### Archived execution path

                The public `alessum/mpemba_circuits` history verifies the
                partial-SWAP constructor, but it is not a runnable provenance
                record for Fig. 11:

                - the committed runner selects `U1`, not `SU2`;
                - its dormant SU(2) branch samples $J\in[-\pi,\pi]$, not the
                  caption's $[-\pi/5,\pi/5]$;
                - `Circuit.run` does not construct an SU(2)-twirled eight-qubit
                  reduced state; an earlier revision called the four-qubit twirl;
                - it prepares a homogeneous spin-coherent system and a polarized
                  $|0\rangle^{\otimes12}$ environment, not Eq. (79) and a singlet
                  environment;
                - that history contains no SU(2) coupling table or raw SU(2)
                  trajectories.

                Thus the answer is **yes at the two-qubit gate level**, but the
                published curves cannot be regenerated by merely changing the
                archived symmetry flag.
                """
            ),
            md(
                r"""
                ## 3. Exact non-Abelian twirl

                The printed Eq. (79) writes the initial state and environment as

                $$
                |\varphi(\theta)\rangle =
                \cos(\theta/2)|\xi\rangle^{\otimes N_s/2}
                +\sin(\theta/2)|0\rangle^{\otimes N_s},\qquad
                |\pi_e\rangle=|\xi\rangle^{\otimes N_e/2},
                $$

                where $|\xi\rangle=(|01\rangle-|10\rangle)/\sqrt2$. Schur-Weyl
                decomposition gives
                $\mathcal H_s=\bigoplus_j\mathcal M_j\otimes\mathcal V_j$.
                Haar twirling erases coherences between inequivalent $j$ sectors,
                preserves the multiplicity state, and replaces each spin-$j$
                representation by $I_{2j+1}/(2j+1)$:

                $$
                \mathcal G_{\rm SU(2)}(\rho)
                =\bigoplus_j
                \operatorname{Tr}_{\mathcal V_j}(\Pi_j\rho\Pi_j)
                \otimes\frac{I_{\mathcal V_j}}{2j+1}.
                $$

                The resource monotone is evaluated in natural-log units,

                $$
                M_{\rm SU(2)}(\rho_s)
                =S(\mathcal G_{\rm SU(2)}[\rho_s])-S(\rho_s)
                =S(\rho_s\Vert\mathcal G_{\rm SU(2)}[\rho_s]).
                """
            ),
            code(
                r"""
                print("Schur multiplicities (2j -> multiplicity):")
                print(
                    dict(
                        zip(
                            data["spin_twice"].tolist(),
                            data["multiplicity"].tolist(),
                        )
                    )
                )

                # Independent structural checks on a generic four-qubit state.
                rng = np.random.default_rng(7)
                test_ket = rng.normal(size=16) + 1j * rng.normal(size=16)
                test_ket /= np.linalg.norm(test_ket)
                test_rho = np.outer(test_ket, test_ket.conj())
                test_basis, test_paths = nu.su2_schur_basis(4)
                test_twirl = nu.su2_twirl_exact(
                    test_rho, test_basis, test_paths
                )
                test_twirl_twice = nu.su2_twirl_exact(
                    test_twirl, test_basis, test_paths
                )

                collective_four = nu.kron_all([rotation] * 4)
                with np.errstate(
                    over="ignore", invalid="ignore", divide="ignore"
                ):
                    rotated_twirl = (
                        collective_four
                        @ test_twirl
                        @ collective_four.conj().T
                    )
                assert np.isfinite(rotated_twirl).all()
                invariance_error = np.linalg.norm(rotated_twirl - test_twirl)
                idempotence_error = np.linalg.norm(
                    test_twirl_twice - test_twirl
                )
                print("twirl trace:", np.trace(test_twirl).real)
                print("twirl minimum eigenvalue:", np.linalg.eigvalsh(test_twirl).min())
                print("twirl idempotence error:", idempotence_error)
                print("twirl invariance error:", invariance_error)

                assert np.allclose(np.trace(test_twirl), 1, atol=1e-12)
                assert np.linalg.eigvalsh(test_twirl).min() > -1e-12
                assert idempotence_error < 1e-11
                assert invariance_error < 1e-11
                """
            ),
            md(
                r"""
                ## 4. A no-go test for the published figure

                At $t=0$ the system is pure and has support only in a singlet
                ($j=0$) and in $|j=N_s/2,m=j\rangle$. If $p$ is the polarized
                weight, the twirled state has entropy

                $$
                M(0)=h_2(p)+p\ln(N_s+1).
                $$

                Literal Eq. (79) has
                $p_{\rm Eq.79}=\sin^2(\theta/2)$. Because the environment is
                invariant and the joint unitary is SU(2)-invariant, covariance
                gives the global-in-time bound

                $$
                M[\rho_\theta(t)]\leq M[\rho_\theta(0)].
                $$

                The vector reference is tested against this bound in both nats and
                bits. We also show the exchanged-coefficient hypothesis, but do not
                relabel it as Eq. (79). Passing a necessary bound does not establish
                data provenance.
                """
            ),
            code(
                r"""
                def binary_entropy(probability):
                    probability = np.asarray(probability, dtype=float)
                    return -(
                        probability * np.log(probability)
                        + (1 - probability) * np.log(1 - probability)
                    )


                def analytic_initial_asymmetry(polarized_weight):
                    return (
                        binary_entropy(polarized_weight)
                        + polarized_weight * np.log(n_system + 1)
                    )


                angles = np.pi * theta_over_pi
                p_equation = np.sin(angles / 2) ** 2
                p_swapped = np.cos(angles / 2) ** 2
                initial_equation = analytic_initial_asymmetry(p_equation)
                initial_swapped = analytic_initial_asymmetry(p_swapped)
                measured_initial = asymmetry[:, :, 0]

                equation_error = np.max(
                    np.abs(measured_initial - initial_equation)
                )
                assert equation_error < 1e-11

                reference_columns = [
                    f"theta_0{int(round(theta * 100)):02d}_pi"
                    for theta in theta_over_pi
                ]
                reference_curves = np.column_stack(
                    [figure_reference[name] for name in reference_columns]
                )
                reference_maximum = reference_curves.max(axis=0)

                print("literal Eq. (79) M(0):", initial_equation)
                print("stored M(0):          ", measured_initial.mean(axis=0))
                print("published-vector maxima:", reference_maximum)
                print(
                    "Eq. (79) violations in nats:",
                    reference_maximum > initial_equation + 1e-3,
                )
                print(
                    "Eq. (79) violations even if read as bits:",
                    reference_maximum
                    > initial_equation / np.log(2) + 1e-3,
                )
                assert np.all(
                    reference_maximum > initial_equation + 1e-3
                )
                assert np.any(
                    reference_maximum
                    > initial_equation / np.log(2) + 1e-3
                )

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
                axes[0].plot(
                    figure_reference["displayed_time"],
                    reference_curves,
                )
                for state_index, (label, color) in enumerate(
                    zip(labels, PALETTE_5)
                ):
                    axes[0].lines[state_index].set(
                        color=color, label=label, lw=1.8
                    )
                axes[0].set(
                    xscale="log",
                    xlabel="displayed time in Fig. 11",
                    ylabel="published vertical coordinate",
                    title="Vectorized Fig. 11 reference",
                )
                axes[0].legend(fontsize=8)

                axes[1].plot(
                    theta_over_pi,
                    reference_maximum,
                    "ko-",
                    label="maximum visible in Fig. 11",
                )
                axes[1].plot(
                    theta_over_pi,
                    initial_equation,
                    "o-",
                    color=BLUE,
                    label="Eq. (79) bound, nats",
                )
                axes[1].plot(
                    theta_over_pi,
                    initial_equation / np.log(2),
                    "s--",
                    color=ORANGE,
                    label="Eq. (79) bound, bits",
                )
                axes[1].plot(
                    theta_over_pi,
                    initial_swapped / np.log(2),
                    "^:",
                    color=GOLD,
                    label="swapped coefficients, bits",
                )
                axes[1].set(
                    xlabel=r"reported $\theta/\pi$",
                    ylabel="asymmetry",
                    title="Necessary covariance-bound check",
                )
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ### A plausible observable bug, tested rather than adopted

                Complete dephasing in the eigenvectors returned by a numerical
                diagonalization of $J^2$ is **not** the SU(2) Haar twirl. The
                eigenvalue of $J^2$ is highly degenerate, so an eigensolver may
                choose an arbitrary basis inside each multiplicity space.
                Destroying coherence in that arbitrary basis produces a
                platform-dependent number.

                This error is worth testing because, for literal Eq. (79), it
                yields values near the high left edge of Fig. 11 and reverses the
                $\theta$ ordering. The following cell applies complete dephasing
                in two equally valid $J^2$ eigenbases. Their disagreement proves
                that this quantity cannot be the symmetry twirl.
                """
            ),
            code(
                r"""
                def pure_state_diagonal_entropy(state, basis):
                    with np.errstate(
                        over="ignore", invalid="ignore", divide="ignore"
                    ):
                        probabilities = np.abs(basis.conj().T @ state) ** 2
                    probabilities = probabilities[probabilities > 1e-14]
                    return -np.sum(probabilities * np.log(probabilities))


                coupled_basis, _ = nu.su2_schur_basis(n_system)
                collective_spin = nu.collective_spin(n_system)
                with np.errstate(
                    over="ignore", invalid="ignore", divide="ignore"
                ):
                    j_squared = sum(
                        component @ component
                        for component in collective_spin
                    )
                _, numerical_j2_basis = np.linalg.eigh(j_squared)

                diagonal_entropy_coupled = []
                diagonal_entropy_numerical = []
                for angle in angles:
                    state = nu.su2_tilted_state(n_system, angle)
                    diagonal_entropy_coupled.append(
                        pure_state_diagonal_entropy(state, coupled_basis)
                    )
                    diagonal_entropy_numerical.append(
                        pure_state_diagonal_entropy(
                            state, numerical_j2_basis
                        )
                    )
                diagonal_entropy_coupled = np.array(
                    diagonal_entropy_coupled
                )
                diagonal_entropy_numerical = np.array(
                    diagonal_entropy_numerical
                )

                print(
                    "complete dephasing, coupled-spin basis:",
                    diagonal_entropy_coupled,
                )
                print(
                    "complete dephasing, numerical J^2 basis:",
                    diagonal_entropy_numerical,
                )
                print(
                    "first visible Fig. 11 coordinates:",
                    reference_curves[0],
                )
                assert not np.allclose(
                    diagonal_entropy_coupled,
                    diagonal_entropy_numerical,
                    atol=1e-6,
                )
                """
            ),
            md(
                r"""
                `tools/analyze_su2_curve_hypotheses.py` evolves this bug
                hypothesis with the printed gates. It improves the sparse
                figure-comparison RMSE relative to the archived polarized-bath
                runner, but it does not reproduce the time dependence: the five
                candidate trajectories remain much farther apart than the
                published curves. It is therefore a diagnostic clue, not a
                replacement protocol.
                """
            ),
            md(
                r"""
                The SU(2) random-circuit literature does not repair this by itself.
                Liu *et al.* use a different staggered tilted ferromagnet,

                $$
                |\psi_0(\theta)\rangle=
                e^{-i\frac{\theta}{2}\sum_j(-1)^j\sigma_j^y}
                |0\cdots0\rangle,
                $$

                in a closed symmetric circuit. Replacing Eq. (79) by that state
                would be a different model, not a convention fix. The archived
                helper code also mixes natural-log relative entropy with a
                base-two entropy helper, so entropy units must be explicit in any
                future raw-data recovery.
                """
            ),
            md(
                r"""
                ## 5. Raw trajectories before ensemble reduction

                The left panel exposes every stored realization when the ensemble
                is small (otherwise the first ten). The right panel shows the mean.
                For fewer than 20 circuits the band is the full observed range;
                only larger ensembles use a 16th-84th percentile band.
                """
            ),
            code(
                r"""
                mean_curves = asymmetry.mean(axis=0)
                if n_realizations < 20:
                    lower_curves = asymmetry.min(axis=0)
                    upper_curves = asymmetry.max(axis=0)
                    band_label = "full observed range"
                else:
                    lower_curves, upper_curves = np.quantile(
                        asymmetry, [0.16, 0.84], axis=0
                    )
                    band_label = "16th-84th percentile"
                n_show = min(10, n_realizations)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
                for state_index, (label, color) in enumerate(
                    zip(labels, PALETTE_5)
                ):
                    for realization in range(n_show):
                        axes[0].plot(
                            times[1:],
                            asymmetry[realization, state_index, 1:],
                            color=color,
                            alpha=0.22,
                            lw=0.9,
                        )
                    axes[1].plot(
                        times[1:],
                        mean_curves[state_index, 1:],
                        color=color,
                        lw=2,
                        label=label,
                    )
                    axes[1].fill_between(
                        times[1:],
                        lower_curves[state_index, 1:],
                        upper_curves[state_index, 1:],
                        color=color,
                        alpha=0.12,
                    )

                for ax in axes:
                    ax.set_xscale("log")
                    ax.set_xlabel("Floquet layer")
                axes[0].set(
                    ylabel=r"$M_{\rm SU(2)}[\rho_s(t)]$",
                    title=f"{n_show} raw exact-size realizations",
                )
                axes[1].set(
                    title=f"{n_realizations}-circuit mean; {band_label}"
                )
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 6. Crossings, global monotonicity, and memory revivals

                A Mpemba crossing is directional: the initially more asymmetric
                mean curve must later fall below the initially less asymmetric one.
                We test that direction for every pair.

                Reusing the environment can produce positive one-step increments.
                We call these *resource revivals*. They are compatible with memory
                effects and failure of CP divisibility, but a positive increment of
                one monotone alone is not a complete BLP distinguishability
                backflow calculation.
                """
            ),
            code(
                r"""
                finite_crossings = []
                for first in range(len(theta_over_pi)):
                    for second in range(first):
                        if mean_curves[first, 0] >= mean_curves[second, 0]:
                            initially_more, initially_less = first, second
                        else:
                            initially_more, initially_less = second, first
                        tau = nu.crossing_time(
                            times,
                            mean_curves[initially_more],
                            mean_curves[initially_less],
                        )
                        if np.isfinite(tau):
                            finite_crossings.append(
                                (
                                    theta_over_pi[initially_more],
                                    theta_over_pi[initially_less],
                                    tau,
                                )
                            )

                if finite_crossings:
                    for more, less, tau in finite_crossings:
                        print(
                            f"initially more asymmetric theta={more:.2f}pi "
                            f"crosses theta={less:.2f}pi at t={tau:.2f}"
                        )
                else:
                    print(
                        "No directional Mpemba crossing in this stored mean."
                    )

                increments = np.diff(asymmetry, axis=-1)
                overshoot_above_initial = np.max(
                    asymmetry - asymmetry[:, :, [0]]
                )
                print(
                    "largest raw one-step resource revival:",
                    increments.max(),
                )
                print(
                    "fraction of raw steps with positive increment:",
                    (increments > 1e-10).mean(),
                )
                print(
                    "largest violation of M(t) <= M(0):",
                    overshoot_above_initial,
                )

                assert increments.max() > 0
                assert overshoot_above_initial < 1e-10
                """
            ),
            code(
                r"""
                fig, ax = plt.subplots(figsize=(7.2, 3.8))
                for state_index, (label, color) in enumerate(
                    zip(labels, PALETTE_5)
                ):
                    ax.plot(
                        times[1:],
                        np.diff(mean_curves[state_index]),
                        color=color,
                        lw=1.2,
                        label=label,
                    )
                ax.axhline(0, color="0.35", lw=0.8)
                ax.set(
                    xscale="log",
                    xlabel="Floquet layer",
                    ylabel=r"$\Delta M_{\rm SU(2)}$",
                    title="mean one-step resource increments",
                )
                ax.legend(ncol=3, fontsize=8)
                plt.show()
                """
            ),
            md(
                r"""
                ## 7. What is established

                - The local gates, environment preparation, and exact Schur twirl
                  implement an SU(2)-covariant open-system dynamics.
                  Each realization uses random SU(2)-invariant two-qubit
                  partial-SWAP gates with fixed-in-time bond couplings; this is a
                  quenched random Floquet circuit.
                - The bundled raw data obey the analytic initial asymmetry and the
                  covariance bound $M(t)\le M(0)$, while displaying intermediate
                  resource revivals from environment reuse.
                - The three-realization NPZ is an implementation check, not a
                  statistically converged figure reproduction.
                - The public runner contains the correct SU(2) gate constructor,
                  but its committed execution path, initial state, environment,
                  coupling law, and missing raw SU(2) data do not establish Fig. 11
                  provenance.
                - The published vector curves fail the covariance bound for literal
                  Eq. (79), even if their vertical values are read as bits rather
                  than nats. Exchanged coefficients in bit units pass this necessary
                  test, but the missing raw parameters prevent a provenance claim.
                - Complete dephasing in a numerical $J^2$ eigenbasis partly
                  explains the anomalous left-edge scale, but it is basis dependent
                  and fails to reproduce the curve separation in time.
                - The staggered state of Liu *et al.* is physically relevant
                  literature context but defines a different initial-state family.

                **Primary references:** [published manuscript and Fig.
                11](https://doi.org/10.1103/rbt4-psfd);
                [Marvian and Spekkens, modes of
                asymmetry](https://doi.org/10.1103/PhysRevA.90.062110);
                [Breuer *et al.*, non-Markovian open
                dynamics](https://doi.org/10.1103/RevModPhys.88.021002);
                [Liu *et al.*, SU(2)-symmetric random
                circuits](https://doi.org/10.1103/PhysRevLett.133.140405).
                """
            ),
        ],
    )


def build_all() -> None:
    build_u1_markovian()
    build_u1_nonmarkovian()
    build_su2_nonmarkovian()


if __name__ == "__main__":
    build_all()
    print("Wrote 3 data-backed circuit notebooks.")
