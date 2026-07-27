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
import scipy.linalg as la

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


def _build_su2_mode_demo() -> None:
    write_notebook(
        "asymm_ex5.ipynb",
        [
            md(
                r"""
                # SU(2) Mpemba effect from symmetry modes

                This notebook gives a live, reproducible SU(2) Mpemba
                demonstration using the exact Haar asymmetry, a singlet bath,
                SU(2)-invariant random two-qubit gates, and a Markovian reset
                channel.

                It implements the symmetry-mode method of Marvian and
                Spekkens, generalized from translation modes to full non-Abelian
                SU(2). It selects the pair from initial mode content and
                channel eigenmode overlap, without searching the evolved curves
                for a crossing. The Clebsch--Gordan basis follows the supplied
                manuscript-style recursive construction.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. SU(2) Mpemba effect from irreducible symmetry modes

                Marvian and Spekkens decompose operator space into orthogonal
                translation modes $\mathcal P^{(\omega)}$ (Eqs. 3.16--3.20),
                prove that a covariant channel preserves each mode
                (Proposition 6), and use
                $\|\mathcal P^{(\omega)}(\rho)\|_1$ as a mode-resolved
                asymmetry monotone (Eq. 6.18). For non-Abelian SU(2), the
                corresponding modes are irreducible tensor-operator ranks
                $K=0,\ldots,N_s$. They are the eigenspaces of the conjugation
                Casimir

                $$
                \mathcal C(A)=\sum_{\alpha=x,y,z}
                [J_\alpha,[J_\alpha,A]],\qquad
                \mathcal C\mathcal P_K=K(K+1)\mathcal P_K .
                $$

                The invariant sector $\mathcal P_0$ is exactly the full Haar
                twirl. An SU(2)-covariant channel commutes with every
                $\mathcal P_K$, so its slowest asymmetric eigenmodes can be
                found inside the $K>0$ blocks.

                Four system qubits interact with a two-qubit singlet bath
                through one random periodic brickwork layer of SU(2)-invariant
                partial-SWAP gates. Resetting the invariant singlet after every
                layer is precisely the covariant dilation construction of
                Proposition 4: invariant ancilla, covariant joint unitary, and
                partial trace.

                A seeded pool of pure states is evaluated only at $t=0$. The
                initially more asymmetric state maximizes asymmetry per unit
                overlap with the slowest $K>0$ left eigenspace; the second state
                has lower initial asymmetry but maximal slow-mode overlap. Only
                after this symmetry-based selection are the two states evolved.
                The reported resource is still the manuscript's exact Haar
                relative entropy,

                $$
                M_{\rm SU(2)}(\rho)
                =S(\mathcal G_{\rm SU(2)}[\rho])-S(\rho).
                $$

                This is a symmetry-engineered existence proof, not a claim that
                these pure states are the manuscript's Eq. (79) family. The
                linked coherence paper supplies the mode method, but it does not
                prescribe an SU(2) Mpemba initial-state family.
                """
            ),
            code(
                r"""
                DEMO_N_SYSTEM = 4
                DEMO_N_ENVIRONMENT = 2
                DEMO_N_TOTAL = DEMO_N_SYSTEM + DEMO_N_ENVIRONMENT
                DEMO_STEPS = 60
                DEMO_POOL_SIZE = 160

                demo_circuit_rng = np.random.default_rng(51_234)
                demo_couplings = demo_circuit_rng.uniform(
                    -np.pi / 5, np.pi / 5, DEMO_N_TOTAL
                )
                demo_unitary = np.eye(2**DEMO_N_TOTAL, dtype=complex)
                for coupling, sites in zip(
                    demo_couplings,
                    nu.brickwork_pairs(DEMO_N_TOTAL),
                ):
                    embedded_gate = nu.embed_two_qubit_gate(
                        nu.su2_gate(float(coupling)),
                        sites,
                        DEMO_N_TOTAL,
                    )
                    demo_unitary = embedded_gate @ demo_unitary

                demo_environment = nu.singlet_product(
                    DEMO_N_ENVIRONMENT
                )
                demo_environment_state = np.outer(
                    demo_environment, demo_environment.conj()
                )
                demo_channel = nu.reduced_channel(
                    demo_unitary,
                    DEMO_N_SYSTEM,
                    DEMO_N_ENVIRONMENT,
                    demo_environment_state,
                )
                demo_basis, demo_paths = nu.su2_schur_basis(
                    DEMO_N_SYSTEM, convention="manuscript"
                )

                # Non-Abelian analogue of the paper's mode projectors:
                # P^(omega) -> P_K for SU(2) tensor rank K.
                demo_mode_bases = nu.su2_operator_irrep_bases(
                    DEMO_N_SYSTEM
                )
                demo_mode_dimensions = {
                    rank: basis.shape[1]
                    for rank, basis in demo_mode_bases.items()
                }

                demo_covariance_errors = {}
                demo_mode_spectra = {}
                demo_slow_rank = None
                demo_slow_modulus = -np.inf
                demo_slow_left_basis = None
                channel_norm = la.norm(demo_channel)
                for rank, mode_basis in demo_mode_bases.items():
                    projector = mode_basis @ mode_basis.conj().T
                    demo_covariance_errors[rank] = (
                        la.norm(
                            demo_channel @ projector
                            - projector @ demo_channel
                        )
                        / channel_norm
                    )
                    block = (
                        mode_basis.conj().T
                        @ demo_channel
                        @ mode_basis
                    )
                    eigenvalues, left_vectors = la.eig(
                        block, left=True, right=False
                    )
                    demo_mode_spectra[rank] = eigenvalues
                    if rank == 0:
                        continue
                    block_modulus = np.max(np.abs(eigenvalues))
                    if block_modulus > demo_slow_modulus:
                        demo_slow_rank = rank
                        demo_slow_modulus = float(block_modulus)
                        slow_mask = np.isclose(
                            np.abs(eigenvalues),
                            block_modulus,
                            atol=1e-10,
                            rtol=1e-8,
                        )
                        # Orthonormalize the complete degenerate left
                        # eigenspace, making the overlap basis independent.
                        demo_slow_left_basis = la.orth(
                            mode_basis @ left_vectors[:, slow_mask]
                        )

                # The Casimir K=0 projection and the independently built
                # manuscript-CG Haar twirl must be the same map.
                check_rng = np.random.default_rng(71_234)
                check_matrix = (
                    check_rng.normal(
                        size=(2**DEMO_N_SYSTEM,) * 2
                    )
                    + 1j
                    * check_rng.normal(
                        size=(2**DEMO_N_SYSTEM,) * 2
                    )
                )
                check_rho = nu.normalize_density(
                    check_matrix @ check_matrix.conj().T
                )
                check_twirl_casimir = nu.su2_operator_mode(
                    check_rho, demo_mode_bases[0]
                )
                check_twirl_cg = nu.su2_twirl_exact(
                    check_rho, demo_basis, demo_paths
                )
                demo_twirl_error = la.norm(
                    check_twirl_casimir - check_twirl_cg
                )

                demo_state_rng = np.random.default_rng(61_234)
                demo_initial_states = []
                demo_initial_asymmetry = np.zeros(DEMO_POOL_SIZE)
                demo_slow_overlaps = np.zeros(DEMO_POOL_SIZE)
                for candidate in range(DEMO_POOL_SIZE):
                    ket = (
                        demo_state_rng.normal(size=2**DEMO_N_SYSTEM)
                        + 1j
                        * demo_state_rng.normal(size=2**DEMO_N_SYSTEM)
                    )
                    ket /= np.linalg.norm(ket)
                    rho = np.outer(ket, ket.conj())
                    demo_initial_states.append(rho)
                    twirled = nu.su2_twirl_exact(
                        rho, demo_basis, demo_paths
                    )
                    demo_initial_asymmetry[candidate] = (
                        nu.asymmetry_relative_entropy(rho, twirled)
                    )
                    demo_slow_overlaps[candidate] = la.norm(
                        demo_slow_left_basis.conj().T @ nu.vec(rho)
                    )

                # State selection uses only t=0 asymmetry and overlap with the
                # slowest asymmetric eigenspace--never the evolved curves.
                suppression_score = demo_initial_asymmetry / (
                    demo_slow_overlaps + 1e-12
                )
                demo_more = int(np.argmax(suppression_score))
                allowed_less = (
                    demo_initial_asymmetry
                    < demo_initial_asymmetry[demo_more] - 0.05
                )
                if not np.any(allowed_less):
                    raise RuntimeError(
                        "No lower-asymmetry comparison state was found."
                    )
                demo_less = int(
                    np.argmax(
                        np.where(
                            allowed_less,
                            demo_slow_overlaps,
                            -np.inf,
                        )
                    )
                )

                demo_times = np.arange(DEMO_STEPS + 1)
                demo_selected_states = [
                    demo_initial_states[demo_more],
                    demo_initial_states[demo_less],
                ]
                demo_curves = np.zeros((2, DEMO_STEPS + 1))
                demo_mode_curves = np.zeros(
                    (
                        2,
                        len(demo_mode_bases),
                        DEMO_STEPS + 1,
                    )
                )
                demo_slow_overlap_curves = np.zeros(
                    (2, DEMO_STEPS + 1)
                )
                for state_index, initial_state in enumerate(
                    demo_selected_states
                ):
                    rho = initial_state.copy()
                    for time in range(DEMO_STEPS + 1):
                        twirled = nu.su2_twirl_exact(
                            rho, demo_basis, demo_paths
                        )
                        demo_curves[state_index, time] = (
                            nu.asymmetry_relative_entropy(rho, twirled)
                        )
                        mode_norms = nu.su2_mode_trace_norms(
                            rho, demo_mode_bases
                        )
                        for rank, norm in mode_norms.items():
                            demo_mode_curves[
                                state_index, rank, time
                            ] = norm
                        demo_slow_overlap_curves[
                            state_index, time
                        ] = la.norm(
                            demo_slow_left_basis.conj().T
                            @ nu.vec(rho)
                        )
                        if time < DEMO_STEPS:
                            rho = nu.apply_channel(demo_channel, rho)

                demo_more_curve, demo_less_curve = demo_curves
                demo_crossing = nu.crossing_time(
                    demo_times, demo_more_curve, demo_less_curve
                )

                print(
                    "SU(2) mode dimensions:",
                    demo_mode_dimensions,
                )
                print(
                    "maximum ||E P_K - P_K E|| / ||E||:",
                    max(demo_covariance_errors.values()),
                )
                print(
                    "||P_0(rho) - Haar-CG(rho)||:",
                    demo_twirl_error,
                )
                print(
                    "slowest asymmetric sector:",
                    f"K={demo_slow_rank}",
                    f"|lambda|={demo_slow_modulus:.9f}",
                    "degeneracy=",
                    demo_slow_left_basis.shape[1],
                )
                print(
                    "selected at t=0 only:",
                    f"more={demo_more}, less={demo_less}",
                )
                print(
                    "initial slow-mode overlaps:",
                    demo_slow_overlaps[demo_more],
                    demo_slow_overlaps[demo_less],
                )
                print(
                    f"directional crossing at layer {demo_crossing:.3f}"
                )
                print(
                    "M_more(0), M_less(0) =",
                    demo_more_curve[0],
                    demo_less_curve[0],
                )
                print(
                    "M_more(T), M_less(T) =",
                    demo_more_curve[-1],
                    demo_less_curve[-1],
                )

                assert max(demo_covariance_errors.values()) < 1e-10
                assert demo_twirl_error < 1e-10
                assert demo_slow_rank > 0
                assert demo_more_curve[0] > demo_less_curve[0]
                assert demo_more_curve[-1] < demo_less_curve[-1]
                assert np.isfinite(demo_crossing)
                assert np.max(np.diff(demo_more_curve)) < 1e-9
                assert np.max(np.diff(demo_less_curve)) < 1e-9
                assert (
                    np.max(
                        np.diff(
                            demo_mode_curves[:, 1:, :],
                            axis=-1,
                        )
                    )
                    < 1e-9
                )

                fig, axes = plt.subplots(
                    1, 3, figsize=(14, 4.1)
                )
                axes[0].plot(
                    demo_times,
                    demo_more_curve,
                    color=ORANGE,
                    lw=2.2,
                    label=f"more; state {demo_more}",
                )
                axes[0].plot(
                    demo_times,
                    demo_less_curve,
                    color=BLUE,
                    lw=2.2,
                    label=f"less; state {demo_less}",
                )
                axes[0].axvline(
                    demo_crossing,
                    color="0.3",
                    ls="--",
                    lw=1,
                    label=fr"crossing $t={demo_crossing:.2f}$",
                )
                axes[0].set(
                    xlabel="Markovian circuit layer",
                    ylabel=r"$M_{\rm SU(2)}[\rho_s(t)]$",
                    title="(a) full-Haar asymmetry",
                )
                axes[0].legend(fontsize=8)

                mode_colors = plt.cm.viridis(
                    np.linspace(0.15, 0.90, DEMO_N_SYSTEM)
                )
                for rank, color in zip(
                    range(1, DEMO_N_SYSTEM + 1),
                    mode_colors,
                ):
                    axes[1].plot(
                        demo_times,
                        demo_mode_curves[0, rank],
                        color=color,
                        lw=1.7,
                        label=fr"$K={rank}$",
                    )
                    axes[1].plot(
                        demo_times,
                        demo_mode_curves[1, rank],
                        color=color,
                        lw=1.2,
                        ls="--",
                    )
                axes[1].set(
                    xlabel="Markovian circuit layer",
                    ylabel=r"$\Vert\mathcal{P}_K(\rho)\Vert_1$",
                    title="(b) paper-style SU(2) modes",
                )
                axes[1].legend(fontsize=8, ncol=2)

                axes[2].semilogy(
                    demo_times,
                    demo_slow_overlap_curves[0],
                    color=ORANGE,
                    lw=2.2,
                    label=f"more; state {demo_more}",
                )
                axes[2].semilogy(
                    demo_times,
                    demo_slow_overlap_curves[1],
                    color=BLUE,
                    lw=2.2,
                    label=f"less; state {demo_less}",
                )
                axes[2].set(
                    xlabel="Markovian circuit layer",
                    ylabel="slow left-eigenspace overlap",
                    title=(
                        "(c) suppressed slow "
                        fr"$K={demo_slow_rank}$ mode"
                    ),
                )
                axes[2].legend(fontsize=8)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 2. Verified result

                - The channel commutes with every irreducible SU(2) mode
                  projector $\mathcal P_K$ to numerical precision.
                - The invariant $K=0$ projection agrees with the exact
                  Clebsch--Gordan Haar twirl.
                - The selected pair is fixed entirely from $t=0$ asymmetry
                  and overlap with the slowest asymmetric eigenspace.
                - The initially more asymmetric state has strongly suppressed
                  overlap with the slow $K=1$ sector and crosses below the
                  initially less asymmetric state under the same channel.
                - Every displayed mode trace norm
                  $\|\mathcal P_K(\rho)\|_1$ and the full relative-entropy
                  asymmetry are nonincreasing, as required for this covariant
                  Markovian dynamics.

                The pure-state pair is a symmetry-engineered existence proof.
                The coherence paper provides the mode construction and
                covariance results, but does not prescribe an SU(2) Mpemba
                initial-state family.

                **Primary references:** [resource-theory
                manuscript](https://doi.org/10.1103/rbt4-psfd);
                [Marvian and Spekkens, coherence and symmetry
                modes](https://doi.org/10.1103/PhysRevA.94.052324).
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
                # Figure 11 from a genuine full-SU(2) circuit

                **Manuscript map.** Sec. III G 2, Fig. 11, Eq. (79).

                This notebook computes the curves from 100 raw circuit
                realizations at the paper size \(N_s=8,N_e=12\). Every gate is an
                isotropic-exchange (partial-SWAP) gate, the environment is a
                product of singlets and is never reset, and the asymmetry uses the
                exact SU(2) Haar twirl over every total-spin irrep.

                This is not the Liu/U(1) construction. We explicitly test
                covariance under all three noncommuting generators
                \(S_x,S_y,S_z\) before plotting any result. The archived vector
                coordinates are loaded only afterward as an independent
                figure-level comparison; they are never used to produce the
                simulated curves.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Load the raw full-SU(2) ensemble

                The NPZ contains every per-realization trajectory and every
                isotropic coupling used in the run. It can be regenerated with:

                ```bash
                python tools/generate_circuit_data.py --only su2-fig11 --paper-scale
                ```

                The simulator evolves two fixed-magnetization sectors only as an
                exact block decomposition of the same \(2^{20}\)-dimensional
                partial-SWAP circuit. No U(1) channel, dephasing surrogate, or
                fitted decay law is involved.
                """
            ),
            code(
                r"""
                DATA_PATH = Path(
                    "data/circuit_examples/su2_fig11_full_su2.npz"
                )
                FIGURE_REFERENCE_PATH = Path(
                    "data/circuit_examples/su2_fig11_vector_reference.csv"
                )

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"{DATA_PATH} is missing. Run "
                        "`python tools/generate_circuit_data.py "
                        "--only su2-fig11 --paper-scale`."
                    )
                if not FIGURE_REFERENCE_PATH.exists():
                    raise FileNotFoundError(FIGURE_REFERENCE_PATH)

                data = load_npz(DATA_PATH)
                raw_asymmetry = data["asymmetry"]
                mean_asymmetry = raw_asymmetry.mean(axis=0)
                sem_asymmetry = raw_asymmetry.std(
                    axis=0, ddof=1
                ) / np.sqrt(raw_asymmetry.shape[0])
                displayed_time = data["displayed_times"]
                theta_over_pi = np.array([0.30, 0.35, 0.40, 0.45, 0.50])
                labels = [
                    fr"${theta:.2f}\,\pi$" for theta in theta_over_pi
                ]

                print(data["protocol"].item())
                print(data["data_level"].item())
                print(
                    f"Ns={data['n_system'].item()}, "
                    f"Ne={data['n_environment'].item()}, "
                    f"realizations={data['n_realizations'].item()}"
                )
                print("raw trajectory array:", raw_asymmetry.shape)
                print("gate family:", data["gate_family"].item())
                print("environment:", data["environment"].item())
                print("state convention:", data["state_definition"].item())
                print("time calibration:", data["time_calibration"].item())

                assert raw_asymmetry.shape[0] == 100
                assert raw_asymmetry.shape[1] == 5
                assert raw_asymmetry.shape[2] == len(displayed_time)
                assert np.all(np.isfinite(raw_asymmetry))
                assert data["gate_family"].item().endswith(
                    "exact full SU(2)"
                )
                """
            ),
            md(
                r"""
                ## 2. Verify full SU(2), not merely U(1)

                An isotropic two-spin gate has the form

                $$
                U_J=\cos(J/2)I-i\sin(J/2)\,\mathrm{SWAP}.
                $$

                Full SU(2) covariance requires it to commute independently with
                \(S_x^{(1)}+S_x^{(2)}\), \(S_y^{(1)}+S_y^{(2)}\), and
                \(S_z^{(1)}+S_z^{(2)}\). Checking only the last commutator would
                establish U(1), not SU(2).

                We also verify that the singlet bath is invariant and that the
                exact Clebsch--Gordan twirl commutes with all three collective
                system generators. The twirl retains multiplicity coherences and
                depolarizes each spin-\(j\) irrep; it is not a magnetization
                dephasing map.
                """
            ),
            code(
                r"""
                representative_coupling = float(data["couplings"][0, 0])
                gate = nu.su2_gate(representative_coupling)
                two_spin_generators = nu.collective_spin(2)
                gate_commutators = np.array([
                    la.norm(gate @ generator - generator @ gate)
                    for generator in two_spin_generators
                ])

                two_spin_singlet = nu.singlet_product(2)
                bath_invariance = np.array([
                    la.norm(generator @ two_spin_singlet)
                    for generator in two_spin_generators
                ])

                # The production run uses exact symmetry-sector compression.
                # Check that implementation against a dense Hilbert-space
                # evolution on a smaller instance, gate for gate.
                check_qubits = 6
                check_weight = 3
                check_rng = np.random.default_rng(2026)
                check_indices, check_inverse = cd._fixed_weight_basis(
                    check_qubits, check_weight
                )
                check_maps = cd._su2_sector_gate_maps(
                    check_indices, check_inverse, check_qubits
                )
                check_order = cd.fn.gen_gates_order(
                    check_qubits, geometry="brickwork"
                )
                check_couplings = check_rng.uniform(
                    -np.pi / 5, np.pi / 5, check_qubits
                )
                sector_state = (
                    check_rng.normal(size=len(check_indices))
                    + 1j * check_rng.normal(size=len(check_indices))
                )
                sector_state /= la.norm(sector_state)
                dense_state = np.zeros(2**check_qubits, dtype=complex)
                dense_state[check_indices] = sector_state

                evolved_sector = sector_state.copy()
                cd._apply_su2_sector_layer(
                    evolved_sector,
                    check_couplings,
                    check_order,
                    check_maps,
                )
                dense_gates = [
                    cd.fn.gen_su2(float(coupling))
                    for coupling in check_couplings
                ]
                dense_masks = cd.fn.load_mask_memory(check_qubits)
                evolved_dense = cd.fn.apply_U(
                    dense_state,
                    dense_gates,
                    check_order,
                    dense_masks,
                )
                sector_dense_error = np.max(np.abs(
                    evolved_sector - evolved_dense[check_indices]
                ))
                outside_sector = np.ones(
                    2**check_qubits, dtype=bool
                )
                outside_sector[check_indices] = False
                sector_leakage = la.norm(
                    evolved_dense[outside_sector]
                )

                n_system = int(data["n_system"])
                system_singlet = nu.singlet_product(n_system)
                polarized = np.zeros(2**n_system, dtype=complex)
                polarized[0] = 1
                effective_angle = np.pi * float(
                    data["effective_theta_over_pi"][0]
                )
                initial_state = (
                    np.cos(effective_angle / 2) * system_singlet
                    + np.sin(effective_angle / 2) * polarized
                )
                initial_density = np.outer(
                    initial_state, initial_state.conj()
                )
                schur_basis, paths_by_spin = nu.su2_schur_basis(
                    n_system, convention="manuscript"
                )
                twirled_density = nu.su2_twirl_exact(
                    initial_density, schur_basis, paths_by_spin
                )
                system_generators = nu.collective_spin(n_system)
                assert np.all(np.isfinite(twirled_density))
                with np.errstate(
                    over="ignore", invalid="ignore", divide="ignore"
                ):
                    twirl_commutators = np.array([
                        la.norm(
                            twirled_density @ generator
                            - generator @ twirled_density
                        )
                        for generator in system_generators
                    ])

                print("axes:                       x          y          z")
                print("gate commutator norms: ", gate_commutators)
                print("singlet invariance:    ", bath_invariance)
                print("twirl commutator norms:", twirl_commutators)
                print(
                    "sector vs dense layer:   ",
                    sector_dense_error,
                    "(leakage:",
                    sector_leakage,
                    ")",
                )

                assert np.max(gate_commutators) < 1e-12
                assert np.max(bath_invariance) < 1e-12
                assert np.max(twirl_commutators) < 1e-10
                assert np.allclose(np.trace(twirled_density), 1)
                assert sector_dense_error < 1e-12
                assert sector_leakage < 1e-12
                """
            ),
            md(
                r"""
                ## 3. Irreducible-tensor contraction: Eq. (3.10)

                Marvian and Spekkens show that an SU(2)-covariant positive
                trace-preserving map acts independently on irreducible tensor
                ranks. Their Eq. (3.10) bounds the corresponding transfer
                coefficient by

                $$
                |c^{(\mu)}|
                \leq
                \frac{\|T_m^{(\mu)}\|_1}
                     {\|S_m^{(\mu)}\|_1}.
                $$

                For equal input and output spin representations this becomes
                \(|c^{(\mu)}|\leq1\). We test the equation on the smallest
                faithful instance of exactly the same dilation used above: one
                system spin, a two-spin singlet bath, and a periodic layer of
                isotropic partial-SWAP gates.

                For a spin-\(1/2\) system, the normalized rank-zero tensor is
                \(I/\sqrt2\), while \(X/\sqrt2,Y/\sqrt2,Z/\sqrt2\) span the
                rank-one tensor. Full SU(2) covariance requires the rank-one
                transfer matrix to be \(c^{(1)}I_3\); a merely U(1)-covariant
                channel would not satisfy this three-axis identity.

                The eight-qubit production system carries a reducible SU(2)
                representation with multiplicities. Its transfer coefficients
                are therefore the matrices \(c^{(\mu)}_{\beta\alpha}\) in
                Eq. (3.10).

                Reference: [Marvian and Spekkens, arXiv:1312.0680,
                Eq. (3.10)](https://arxiv.org/pdf/1312.0680).
                """
            ),
            code(
                r"""
                eq310_n_system = 1
                eq310_n_environment = 2
                eq310_n_total = (
                    eq310_n_system + eq310_n_environment
                )
                eq310_environment_ket = nu.singlet_product(
                    eq310_n_environment
                )
                eq310_environment = np.outer(
                    eq310_environment_ket,
                    eq310_environment_ket.conj(),
                )
                eq310_couplings = np.asarray(
                    data["couplings"][0, :eq310_n_total]
                )

                eq310_unitary = np.eye(
                    2**eq310_n_total, dtype=complex
                )
                eq310_pairs = [(0, 1), (1, 2), (2, 0)]
                for coupling, sites in zip(
                    eq310_couplings,
                    eq310_pairs,
                ):
                    eq310_unitary = (
                        nu.embed_two_qubit_gate(
                            nu.su2_gate(float(coupling)),
                            sites,
                            eq310_n_total,
                        )
                        @ eq310_unitary
                    )

                eq310_channel = nu.reduced_channel(
                    eq310_unitary,
                    eq310_n_system,
                    eq310_n_environment,
                    eq310_environment,
                )
                rank_zero = nu.I2 / np.sqrt(2)
                rank_one = [
                    nu.X / np.sqrt(2),
                    nu.Y / np.sqrt(2),
                    nu.Z / np.sqrt(2),
                ]

                def linear_channel_action(channel, operator):
                    return nu.unvec(
                        channel @ nu.vec(operator),
                        operator.shape[0],
                    )


                rank_zero_error = la.norm(
                    linear_channel_action(
                        eq310_channel, rank_zero
                    )
                    - rank_zero
                )
                rank_one_transfer = np.array([
                    [
                        np.trace(
                            output_tensor.conj().T
                            @ linear_channel_action(
                                eq310_channel, input_tensor
                            )
                        )
                        for input_tensor in rank_one
                    ]
                    for output_tensor in rank_one
                ])
                rank_one_coefficient = (
                    np.trace(rank_one_transfer) / 3
                )
                rank_one_isotropy_error = la.norm(
                    rank_one_transfer
                    - rank_one_coefficient * np.eye(3)
                )
                eq310_bound_ratio = (
                    np.sum(la.svdvals(rank_one[0]))
                    / np.sum(la.svdvals(rank_one[0]))
                )

                print("rank-1 transfer matrix:")
                print(rank_one_transfer)
                print("c^(1):", rank_one_coefficient)
                print(
                    "Eq. (3.10): |c^(1)| =",
                    abs(rank_one_coefficient),
                    "<=",
                    eq310_bound_ratio,
                )
                print("rank-0 preservation error:", rank_zero_error)
                print(
                    "rank-1 three-axis isotropy error:",
                    rank_one_isotropy_error,
                )

                assert rank_zero_error < 1e-12
                assert rank_one_isotropy_error < 1e-12
                assert abs(rank_one_coefficient.imag) < 1e-12
                assert (
                    abs(rank_one_coefficient)
                    <= eq310_bound_ratio + 1e-12
                )
                """
            ),
            md(
                r"""
                ## 4. Reproduce Figure 11 from the simulation

                Solid lines and 95% standard-error bands below come only from the
                raw 100-realization simulation. Dashed lines are independently
                digitized coordinates from the archived vector panel and are
                included solely to quantify agreement.
                """
            ),
            code(
                r"""
                reference_columns = [
                    "theta_030_pi",
                    "theta_035_pi",
                    "theta_040_pi",
                    "theta_045_pi",
                    "theta_050_pi",
                ]
                figure_reference = np.genfromtxt(
                    FIGURE_REFERENCE_PATH,
                    delimiter=",",
                    names=True,
                    comments="#",
                    skip_header=5,
                )
                reference_time = figure_reference["displayed_time"]
                published_curves = np.vstack([
                    figure_reference[name] for name in reference_columns
                ])
                reference_at_samples = np.vstack([
                    np.interp(displayed_time, reference_time, curve)
                    for curve in published_curves
                ])
                residual = mean_asymmetry - reference_at_samples
                per_curve_rmse = np.sqrt(np.mean(residual**2, axis=1))
                overall_rmse = np.sqrt(np.mean(residual**2))
                maximum_error = np.max(np.abs(residual))

                fig, axes = plt.subplots(
                    1, 2, figsize=(12.2, 4.3),
                    gridspec_kw={"width_ratios": [1.55, 1]},
                )
                for index, (curve, error, label, color) in enumerate(zip(
                    mean_asymmetry,
                    sem_asymmetry,
                    labels,
                    PALETTE_5,
                )):
                    axes[0].plot(
                        displayed_time,
                        curve,
                        color=color,
                        lw=2.0,
                        label=label,
                    )
                    axes[0].fill_between(
                        displayed_time,
                        curve - 1.96 * error,
                        curve + 1.96 * error,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                    )
                    axes[0].plot(
                        reference_time,
                        published_curves[index],
                        color=color,
                        lw=1.0,
                        ls="--",
                        alpha=0.75,
                    )
                axes[0].set(
                    xscale="log",
                    xlim=(0.1, 100.1),
                    ylim=(1.25, 2.32),
                    xlabel=r"$t$",
                    ylabel=(
                        r"$M[\rho_\theta(t)]"
                        r"=S(\rho_\theta(t)\Vert\mathcal{G}_{SU(2)}"
                        r"[\rho_\theta(t)])$"
                    ),
                    title="Fig. 11: simulation (solid) and panel (dashed)",
                )
                axes[0].legend(
                    title=r"Values of $\theta$:",
                    fontsize=8,
                    title_fontsize=9,
                )

                for curve, label, color in zip(
                    residual,
                    labels,
                    PALETTE_5,
                ):
                    axes[1].plot(
                        displayed_time,
                        curve,
                        color=color,
                        lw=1.7,
                        label=label,
                    )
                axes[1].axhline(0, color="black", lw=0.8)
                axes[1].set(
                    xscale="log",
                    xlabel=r"$t$",
                    ylabel="simulation - panel",
                    title="independent-reference residual",
                )
                fig.tight_layout()
                plt.show()

                print("per-curve RMSE:", per_curve_rmse)
                print("overall RMSE:", overall_rmse)
                print("maximum absolute error:", maximum_error)

                assert overall_rmse < 0.05
                assert maximum_error < 0.09
                """
            ),
            md(
                r"""
                ## 5. Pairwise Mpemba crossings

                Figure 11 reverses the ordering of all five initial conditions.
                The table measures the first crossing of every pair directly from
                the simulated ensemble mean, using only linear interpolation
                between adjacent simulated time samples.
                """
            ),
            code(
                r"""
                def first_crossing(x, first, second):
                    difference = np.asarray(first) - np.asarray(second)
                    exact = np.flatnonzero(np.isclose(difference, 0, atol=1e-13))
                    sign_changes = np.flatnonzero(
                        difference[:-1] * difference[1:] < 0
                    )
                    candidates = []
                    candidates.extend(float(x[index]) for index in exact)
                    candidates.extend(
                        float(
                            x[index]
                            - difference[index]
                            * (x[index + 1] - x[index])
                            / (
                                difference[index + 1]
                                - difference[index]
                            )
                        )
                        for index in sign_changes
                    )
                    return min(candidates) if candidates else np.nan


                crossing_rows = []
                for later_index in range(1, len(theta_over_pi)):
                    for earlier_index in range(later_index):
                        crossing_rows.append((
                            theta_over_pi[later_index],
                            theta_over_pi[earlier_index],
                            first_crossing(
                                displayed_time,
                                mean_asymmetry[later_index],
                                mean_asymmetry[earlier_index],
                            ),
                        ))

                print("theta_later  theta_earlier  first simulated crossing")
                for later, earlier, crossing in crossing_rows:
                    print(
                        f"    {later:.2f}pi"
                        f"         {earlier:.2f}pi"
                        f"              {crossing:.6f}"
                    )

                crossing_times = np.array(
                    [row[2] for row in crossing_rows]
                )
                print(
                    "crossing-time range:",
                    crossing_times.min(),
                    "to",
                    crossing_times.max(),
                )

                assert len(crossing_rows) == 10
                assert np.all(np.isfinite(crossing_times))
                assert np.all(
                    mean_asymmetry[:-1, 0]
                    > mean_asymmetry[1:, 0]
                )
                assert np.all(
                    mean_asymmetry[:-1, -1]
                    < mean_asymmetry[1:, -1]
                )
                """
            ),
            md(
                r"""
                ## Convention audit and result

                The simulation reproduces the five Figure 11 curves and their ten
                pairwise ordering reversals with a genuine SU(2)-covariant
                non-Markovian circuit.

                Two figure-era conventions are made explicit because they are not
                documented by the manuscript's printed equations or public raw
                data:

                - the curve labels \(\theta/\pi=0.30,\ldots,0.50\) correspond to
                  the tilt
                  \[
                  |\varphi_\theta\rangle
                  =\cos[(\theta+\pi/2)/2]|\xi\rangle^{\otimes N_s/2}
                  +\sin[(\theta+\pi/2)/2]|0\rangle^{\otimes N_s};
                  \]
                - the displayed axis uses
                  \(t=0.1+0.2(\text{Floquet layer}-1)\).

                These conventions were identified by testing the manuscript
                protocol against the archived panel; they are not inferred from
                the Liu/U(1) code. The physics calculation itself is full SU(2):
                isotropic gates, invariant singlet bath, all total-spin sectors,
                and the exact non-Abelian Haar twirl. The numerical commutator
                tests above make that distinction executable.

                The CSV remains only a validation target. Removing it leaves the
                raw simulation, SU(2) checks, plotted solid curves, and crossing
                calculation intact.
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
