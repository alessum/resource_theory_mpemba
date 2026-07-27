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
                # Random circuits II: non-Markovian U(1)

                **Manuscript map.** Sec. III G 1, Eq. (78).
                The channel-mode audit below uses Eq. (3.10) of
                [Marvian and Spekkens](https://arxiv.org/pdf/1312.0680).

                Here the environment is a finite quantum memory. It starts in
                $|0\rangle^{\otimes N_e}$ and is never reset, so information and
                asymmetry can return to the system.

                **Computation cost.** Producing the stored data required
                3022.5 seconds (50.4 minutes) on a 10-core Apple M4 MacBook Air
                with 16 GB of memory, using 10 concurrent circuit workers. The
                calculation contains 100 circuits, 1000 Floquet layers, five
                tilts, and a joint Hilbert-space dimension of
                $2^{20}=1{,}048{,}576$, for 500,500 exact reduced-state samples.
                The occupied-U(1)-sector engine performs no interpolation,
                curve fitting, or precision reduction.

                The main dataset is the complete production ensemble:
                100 archived circuits, 1000 Floquet layers, and the manuscript
                Hilbert-space size $N_s=8,N_e=12$. Every gate row comes from
                [`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits)
                at the pinned commit recorded below.

                A smaller $N_s=4,N_e=8$ file is retained only for the inexpensive
                explicit-channel audit of Marvian--Spekkens Eq. (3.10). It is not
                used for the ensemble averages or crossing times.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Load the raw arrays and audit their provenance

                Both files store every monotone value and every gate parameter;
                the notebook performs the reductions itself. The exact-size file
                contains all archived circuit rows $0,\ldots,99$ and every time
                sample $t=0,\ldots,1000$.

                Set `REGENERATE_SCALED=True` to rerun the local 24-circuit
                ensemble used by the channel audit. Recreating the paper-scale
                file from its upstream parameter table is an explicit command
                shown below the audit.
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
                # All ensemble statistics below use the full exact-size run.
                asymmetry = reference_asymmetry
                times = reference_times
                theta_over_pi = reference["theta_over_pi"]
                labels = [fr"$\theta={theta:.2f}\pi$" for theta in theta_over_pi]
                # Dark orange denotes the smallest tilt and light orange
                # denotes the largest one.
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
                    f"{reference['source_indices'][0]}..."
                    f"{reference['source_indices'][-1]}",
                )
                print(
                    "exact-size raw array:", reference_asymmetry.shape,
                    f"(Ns={reference['n_system'].item()}, "
                    f"Ne={reference['n_environment'].item()})",
                )
                print(
                    "scaled audit array:", scaled["asymmetry"].shape,
                    f"(Ns={scaled['n_system'].item()}, "
                    f"Ne={scaled['n_environment'].item()})",
                )
                print(
                    "simulation engine:",
                    reference["simulation_engine"].item(),
                )
                print(
                    "simulation workers:",
                    reference["simulation_workers"].item(),
                )

                assert asymmetry.shape == (
                    100, 5, 1001
                )
                assert scaled["asymmetry"].shape == (
                    int(scaled["n_realizations"]),
                    5,
                    len(scaled["times"]),
                )
                assert np.isfinite(asymmetry).all()
                assert np.isfinite(scaled["asymmetry"]).all()
                assert np.array_equal(times, np.arange(1001))
                assert np.array_equal(
                    reference["source_indices"], np.arange(100)
                )
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
                #   --paper-scale \
                #   --reference-checkout /path/to/mpemba_circuits
                #
                # The generator verifies the upstream parameter file's SHA-256
                # and uses the exact occupied-sector engine.
                """
            ),
            md(
                r"""
                ## 2. Marvian--Spekkens mode transfer: Eq. (3.10)

                Let $\{T_m^{(\mu,\alpha)}\}$ and
                $\{S_m^{(\mu,\beta)}\}$ be Hilbert--Schmidt-orthonormal
                irreducible tensor-operator bases at the input and output.
                Marvian and Spekkens' Eq. (3.10) is

                $$
                \mathcal E(X)
                =
                \sum_{\mu,m,\alpha}
                \operatorname{Tr}\!\left[
                    T_m^{(\mu,\alpha)\dagger}X
                \right]
                \sum_\beta
                c_{\beta\alpha}^{(\mu)}
                S_m^{(\mu,\beta)},
                \qquad
                c_{\beta\alpha}^{(\mu)}
                =
                \operatorname{Tr}\!\left[
                    S_m^{(\mu,\beta)\dagger}
                    \mathcal E(T_m^{(\mu,\alpha)})
                \right].
                $$

                For U(1), every irrep is one-dimensional, so $m$ is
                trivial and $\mu=\omega$ is the charge difference. We use the
                normalized matrix units
                $T^{(\omega,\alpha)}=S^{(\omega,\alpha)}
                =|r\rangle\langle s|$, with
                $\omega=q_r-q_s$. The multiplicity label $\alpha=(r,s)$
                distinguishes all matrix units with the same $\omega$.
                Consequently, each matrix $c^{(\omega)}$ is exactly one
                charge-difference block of the channel superoperator.

                The calculation below constructs the one-layer reduced channel
                for the first stored $N_s=4,N_e=8$ circuit. Its environment is
                initialized in the U(1)-invariant density operator
                $|0\cdots0\rangle\langle0\cdots0|$. It then verifies Eq. (3.10)
                in two independent ways: cross-mode entries vanish, and the
                sum over the $c^{(\omega)}$ blocks reconstructs
                $\mathcal E(X)$ for a generic operator.

                Eq. (3.10) is a transfer formula, not a contraction bound. The
                trace-norm bound on a multiplicity-free coefficient appears
                later as Eqs. (3.12)--(3.13) of the same reference.
                """
            ),
            code(
                r"""
                eq310_n_system = int(scaled["n_system"])
                eq310_n_environment = int(scaled["n_environment"])
                eq310_n_total = (
                    eq310_n_system + eq310_n_environment
                )
                eq310_d_system = 2**eq310_n_system
                eq310_d_environment = 2**eq310_n_environment

                eq310_environment = np.zeros(
                    eq310_d_environment, dtype=complex
                )
                eq310_environment[0] = 1
                # Column j is |0...0>_e tensor |j>_s.
                eq310_joint_basis = np.kron(
                    eq310_environment[:, None],
                    np.eye(eq310_d_system, dtype=complex),
                )
                eq310_parameters = scaled["gate_parameters"][0]
                eq310_gates = [
                    cd.fn.gen_u1(row.tolist())
                    for row in eq310_parameters
                ]
                eq310_ordering = cd.fn.gen_gates_order(
                    eq310_n_total, geometry="brickwork"
                )
                eq310_masks = cd.fn.load_mask_memory(eq310_n_total)
                eq310_evolved_basis = cd._apply_layer_batch(
                    eq310_joint_basis,
                    eq310_gates,
                    eq310_ordering,
                    eq310_masks,
                )

                # A[i, e, j] is the amplitude for system output i and
                # environment output e given system input j.
                eq310_amplitudes = eq310_evolved_basis.reshape(
                    eq310_d_environment,
                    eq310_d_system,
                    eq310_d_system,
                ).transpose(1, 0, 2)
                eq310_channel_tensor = np.einsum(
                    "iej,kel->ikjl",
                    eq310_amplitudes,
                    eq310_amplitudes.conj(),
                    optimize=True,
                )
                eq310_channel = eq310_channel_tensor.reshape(
                    eq310_d_system**2,
                    eq310_d_system**2,
                    order="F",
                )

                eq310_blocks = nu.channel_charge_blocks(
                    eq310_channel, eq310_n_system
                )
                eq310_same_mode_channel = np.zeros_like(
                    eq310_channel
                )
                print("omega   dim c^(omega)")
                for omega, (transfer, indices) in eq310_blocks.items():
                    eq310_same_mode_channel[np.ix_(indices, indices)] = (
                        transfer
                    )
                    print(f"{omega:>3d}     {len(indices):>3d}")

                eq310_cross_mode_error = la.norm(
                    eq310_channel - eq310_same_mode_channel
                )

                eq310_rng = np.random.default_rng(310)
                eq310_test_operator = (
                    eq310_rng.normal(
                        size=(eq310_d_system, eq310_d_system)
                    )
                    + 1j
                    * eq310_rng.normal(
                        size=(eq310_d_system, eq310_d_system)
                    )
                )
                eq310_test_operator /= la.norm(eq310_test_operator)
                eq310_input_coefficients = nu.vec(
                    eq310_test_operator
                )
                eq310_output_coefficients = np.zeros_like(
                    eq310_input_coefficients
                )
                for transfer, indices in eq310_blocks.values():
                    eq310_output_coefficients[indices] = (
                        transfer
                        @ eq310_input_coefficients[indices]
                    )
                eq310_direct_output = nu.unvec(
                    eq310_channel @ eq310_input_coefficients,
                    eq310_d_system,
                )
                eq310_reconstructed_output = nu.unvec(
                    eq310_output_coefficients,
                    eq310_d_system,
                )
                eq310_reconstruction_error = la.norm(
                    eq310_direct_output
                    - eq310_reconstructed_output
                )

                eq310_weights = nu.hamming_weights(
                    eq310_n_system
                )
                eq310_phase = 0.731
                eq310_rotation = np.diag(
                    np.exp(-1j * eq310_phase * eq310_weights)
                )

                def eq310_action(operator):
                    return nu.unvec(
                        eq310_channel @ nu.vec(operator),
                        eq310_d_system,
                    )


                eq310_covariance_error = la.norm(
                    eq310_action(
                        eq310_rotation
                        @ eq310_test_operator
                        @ eq310_rotation.conj().T
                    )
                    - eq310_rotation
                    @ eq310_action(eq310_test_operator)
                    @ eq310_rotation.conj().T
                )
                eq310_identity_vector = nu.vec(
                    np.eye(eq310_d_system)
                )
                eq310_trace_preservation_error = la.norm(
                    eq310_identity_vector.conj() @ eq310_channel
                    - eq310_identity_vector.conj()
                )

                print(
                    "cross-mode Frobenius error:",
                    eq310_cross_mode_error,
                )
                print(
                    "Eq. (3.10) reconstruction error:",
                    eq310_reconstruction_error,
                )
                print(
                    "direct U(1)-covariance error:",
                    eq310_covariance_error,
                )
                print(
                    "trace-preservation error:",
                    eq310_trace_preservation_error,
                )

                assert eq310_cross_mode_error < 1e-12
                assert eq310_reconstruction_error < 1e-12
                assert eq310_covariance_error < 1e-12
                assert eq310_trace_preservation_error < 1e-12
                """
            ),
            md(
                r"""
                ## 3. What is evolved?

                The system state is the homogeneous product tilt

                $$
                |\varphi(\theta)\rangle =
                \bigotimes_{n=1}^{N_s} e^{-i\theta s_y^{(n)}}|0\rangle_n.
                $$

                Each realization samples one U(1)-symmetric brickwork layer from
                Eqs. (71)-(72) and repeatedly applies it to the *joint* pure state.
                The exact engine evolves the nine occupied total-charge sectors
                once and reuses them for all five tilts. At every time it
                reconstructs the complete reduced density matrices from shared
                environment-weight Gram products. Thus the plotted values are
                $S(\mathcal G_{\rm U(1)}[\rho_s])-S(\rho_s)$ evaluated on actual
                reduced states, without interpolation or fitted curves.

                For the exact-size file, the gate parameters are not resampled:
                they are rows of `data/U1_rnd_parameters.npy` in the linked
                reference repository. The sector implementation agrees with the
                dense `functions.apply_U` convention and with the former
                exact-size reference trajectories to floating-point precision.
                """
            ),
            md(
                r"""
                ## 4. Raw manuscript-size trajectories

                The complete file contains 100 independent $2^{20}$-dimensional
                circuit evolutions. Six evenly spaced circuit rows are shown
                below so the raw variability remains visible without creating
                an unreadable 100-panel figure. No fitted or illustrative curve
                enters these panels.
                """
            ),
            code(
                r"""
                shown_reference_rows = np.linspace(
                    0,
                    reference_asymmetry.shape[0] - 1,
                    6,
                    dtype=int,
                )
                fig, axes = plt.subplots(
                    2,
                    3,
                    figsize=(11, 6.5),
                    sharex=True,
                    sharey=True,
                )
                for realization, ax in zip(
                    shown_reference_rows, axes.ravel()
                ):
                    for state_index, (label, color) in enumerate(
                        zip(labels, theta_colors)
                    ):
                        ax.plot(
                            reference_times[1:],
                            reference_asymmetry[
                                realization, state_index, 1:
                            ],
                            color=color,
                            lw=1.2,
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
                axes[0, 0].set_ylabel(r"$M_{\rm U(1)}[\rho_s(t)]$")
                axes[1, 0].set_ylabel(r"$M_{\rm U(1)}[\rho_s(t)]$")
                axes[0, -1].legend(fontsize=7)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 5. Ensemble-averaged symmetry restoration

                The panel below averages each tilt over all 100 manuscript-size
                circuits. Logarithmic axes expose the complete time range and
                the nonexponential, nonmonotonic relaxation caused by the finite
                unreinitialized environment. The crossing times are calculated
                from the same mean curves and printed numerically for use in the
                uncertainty analysis that follows.
                """
            ),
            code(
                r"""
                mean_curves = asymmetry.mean(axis=0)
                paired_differences = (
                    asymmetry[:, 1:, :]
                    - asymmetry[:, :1, :]
                )
                mean_differences = paired_differences.mean(axis=0)
                crossing_times = np.array([
                    nu.crossing_time(
                        times,
                        mean_curves[state_index],
                        mean_curves[0],
                    )
                    for state_index in range(1, len(theta_over_pi))
                ])

                print("tilt        D_theta(0)    mean crossing")
                for state_index, tau in enumerate(
                    crossing_times, start=1
                ):
                    print(
                        f"{theta_over_pi[state_index]:.2f}pi"
                        f"       {mean_differences[state_index - 1, 0]: .6f}"
                        f"       t={tau:.2f}"
                    )

                selected_crossing = crossing_times[-1]

                fig, ax = plt.subplots(figsize=(7.6, 4.5))
                for curve, theta, color in zip(
                    mean_curves,
                    theta_over_pi,
                    theta_colors,
                ):
                    ax.plot(
                        times[1:],
                        curve[1:],
                        color=color,
                        lw=2,
                        label=fr"${theta:.2f}\,\pi$",
                    )
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlim=(1, 1000),
                    xlabel=r"$t$",
                    ylabel=(
                        r"$M[\rho_\theta(t)]"
                        r"=S(\rho_\theta(t)\Vert"
                        r"\mathcal{G}[\rho_\theta(t)])$"
                    ),
                )
                ax.text(
                    0.52,
                    0.96,
                    r"$\mathrm{U(1)}$" + "\nnon-Markovian",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )
                ax.legend(
                    title=r"Values of $\theta$:",
                    fontsize=8,
                    title_fontsize=9,
                    loc="upper right",
                )
                fig.tight_layout()
                plt.show()

                assert np.isfinite(crossing_times).all()
                assert mean_differences[-1, 0] > 0
                assert mean_differences[-1, -1] < 0
                """
            ),
            md(
                r"""
                ## 6. Crossing uncertainty and information backflow

                The two diagnostics below answer different questions.

                Define the paired mean difference

                $$
                D_\theta(t)
                =
                \left\langle
                    M_\theta(t)-M_{0.30\pi}(t)
                \right\rangle_{\rm circuits}.
                $$

                Its positive-to-negative zero crossing is the ordering reversal.
                Pairing the two tilts circuit by circuit removes much of the
                between-circuit variation.

                1. The left panel attaches a paired 95% standard-error band to
                   $D_{0.50\pi}(t)$. Its zero crossing is the ordering reversal
                   of the two ensemble means. The band quantifies the
                   realization uncertainty of the 100-circuit paper-scale
                   average.
                2. The right panel tests non-Markovian backflow. For every raw
                   circuit and tilt we form
                   $\Delta M(t)=M(t+1)-M(t)$. It reports the fraction of those
                   500 raw trajectories for which $\Delta M(t)>0$.

                A crossing does not by itself imply backflow: it compares two
                initial states at the same time. Conversely, a positive
                increment is a revival along one trajectory. Such a revival is
                forbidden under repeated application of a fixed covariant
                reset channel, but is allowed here because the unreinitialized
                environment retains memory.
                """
            ),
            code(
                r"""
                increments = np.diff(asymmetry, axis=-1)
                revival_fraction = (increments > 1e-10).mean(axis=(0, 1))
                revival_window = 11
                revival_padding = revival_window // 2
                revival_fraction_smoothed = np.convolve(
                    np.pad(
                        revival_fraction,
                        revival_padding,
                        mode="edge",
                    ),
                    np.ones(revival_window) / revival_window,
                    mode="valid",
                )
                selected_raw_difference = paired_differences[:, -1, :]
                selected_mean_difference = (
                    selected_raw_difference.mean(axis=0)
                )
                selected_sem_difference = (
                    selected_raw_difference.std(axis=0, ddof=1)
                    / np.sqrt(selected_raw_difference.shape[0])
                )

                print(
                    "largest positive one-step change in the raw data:",
                    increments.max(),
                )
                print(
                    "fraction of all raw steps with a revival:",
                    (increments > 1e-10).mean(),
                )
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

                axes[0].plot(
                    times[1:],
                    selected_mean_difference[1:],
                    color=theta_colors[-1],
                    lw=2,
                )
                axes[0].fill_between(
                    times[1:],
                    (
                        selected_mean_difference
                        - 1.96 * selected_sem_difference
                    )[1:],
                    (
                        selected_mean_difference
                        + 1.96 * selected_sem_difference
                    )[1:],
                    color=theta_colors[-1],
                    alpha=0.16,
                    linewidth=0,
                    label="paired 95% standard-error band",
                )
                axes[0].axhline(0, color="0.6", lw=1)
                axes[0].axvline(
                    selected_crossing,
                    color="0.4",
                    ls=":",
                    lw=1,
                )
                axes[0].set(
                    xscale="log",
                    xlabel="Floquet layer",
                    ylabel=r"$D_{0.50\pi}(t)$",
                    title="uncertainty of the selected mean crossing",
                )
                axes[0].legend(fontsize=8)

                axes[1].plot(
                    times[1:],
                    revival_fraction,
                    color=ORANGE,
                    alpha=0.20,
                    lw=0.8,
                    label="raw fraction at each layer",
                )
                axes[1].plot(
                    times[1:],
                    revival_fraction_smoothed,
                    color=ORANGE,
                    lw=2,
                    label=f"{revival_window}-layer moving average",
                )
                axes[1].set(
                    xscale="log",
                    ylim=(-0.02, 1.02),
                    xlabel="Floquet layer",
                    ylabel=r"fraction with $\Delta M(t)>0$",
                    title=(
                        f"backflow among {asymmetry.shape[0]} "
                        "circuits x 5 tilts"
                    ),
                )
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                plt.show()

                assert increments.max() > 0
                assert np.any(revival_fraction > 0)
                """
            ),
            md(
                r"""
                ## Takeaway

                The complete 100-circuit manuscript-size run exposes the ordering
                reversal directly through the paired differences $D_\theta(t)$.
                The highlighted mean crossing is visible, while its
                standard-error band records the realization uncertainty of the
                full archived ensemble. The separate
                positive-increment statistic demonstrates non-Markovian
                revivals: unlike a reset channel, a finite environment can return
                asymmetry to the system.

                Marvian--Spekkens Eq. (3.10) shows simultaneously that this
                return cannot come from mixing different charge-difference
                modes: each reduced channel from the initial time remains
                block diagonal in $\omega$. Memory instead makes the transfer
                matrices $c_t^{(\omega)}$ depend on the elapsed joint
                evolution; there need not be a state-independent CP map from
                one reduced-time snapshot to the next.

                The full dataset can be regenerated with:
                `tools/generate_circuit_data.py --only u1-reference
                --paper-scale --reference-checkout ...`. The exact occupied-sector
                engine makes this a reproducible local calculation; the stored
                result contains every raw trajectory and time sample.
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
                realizations at the paper size $N_s=8,N_e=12$. Every gate is an
                isotropic-exchange (partial-SWAP) gate, the environment is a
                product of singlets and is never reset, and the asymmetry uses the
                exact SU(2) Haar twirl over every total-spin irrep.

                This is not the Liu/U(1) construction. We explicitly test
                covariance under all three noncommuting generators
                $S_x,S_y,S_z$ before plotting any result.
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
                exact block decomposition of the same $2^{20}$-dimensional
                partial-SWAP circuit. No U(1) channel, dephasing surrogate, or
                fitted decay law is involved.
                """
            ),
            code(
                r"""
                DATA_PATH = Path(
                    "data/circuit_examples/su2_fig11_full_su2.npz"
                )

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"{DATA_PATH} is missing. Run "
                        "`python tools/generate_circuit_data.py "
                        "--only su2-fig11 --paper-scale`."
                    )

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
                $S_x^{(1)}+S_x^{(2)}$, $S_y^{(1)}+S_y^{(2)}$, and
                $S_z^{(1)}+S_z^{(2)}$. Checking only the last commutator would
                establish U(1), not SU(2).

                We also verify that the singlet bath is invariant and that the
                exact Clebsch--Gordan twirl commutes with all three collective
                system generators. The twirl retains multiplicity coherences and
                depolarizes each spin-$j$ irrep; it is not a magnetization
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
                $|c^{(\mu)}|\leq1$. We test the equation on the smallest
                faithful instance of exactly the same dilation used above: one
                system spin, a two-spin singlet bath, and a periodic layer of
                isotropic partial-SWAP gates.

                For a spin-$1/2$ system, the normalized rank-zero tensor is
                $I/\sqrt2$, while $X/\sqrt2,Y/\sqrt2,Z/\sqrt2$ span the
                rank-one tensor. Full SU(2) covariance requires the rank-one
                transfer matrix to be $c^{(1)}I_3$; a merely U(1)-covariant
                channel would not satisfy this three-axis identity.

                The eight-qubit production system carries a reducible SU(2)
                representation with multiplicities. Its transfer coefficients
                are therefore the matrices $c^{(\mu)}_{\beta\alpha}$ in
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

                Solid lines and 95% standard-error bands below come from the
                raw 100-realization simulation.
                """
            ),
            code(
                r"""
                fig, ax = plt.subplots(figsize=(7.6, 4.5))
                for curve, error, label, color in zip(
                    mean_asymmetry,
                    sem_asymmetry,
                    labels,
                    PALETTE_5,
                ):
                    ax.plot(
                        displayed_time,
                        curve,
                        color=color,
                        lw=2.0,
                        label=label,
                    )
                    ax.fill_between(
                        displayed_time,
                        curve - 1.96 * error,
                        curve + 1.96 * error,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                    )
                ax.set(
                    xscale="log",
                    xlim=(0.1, 100.1),
                    ylim=(1.25, 2.32),
                    xlabel=r"$t$",
                    ylabel=(
                        r"$M[\rho_\theta(t)]"
                        r"=S(\rho_\theta(t)\Vert\mathcal{G}_{SU(2)}"
                        r"[\rho_\theta(t)])$"
                    ),
                    title="Fig. 11 reproduction",
                )
                ax.legend(
                    title=r"Values of $\theta$:",
                    fontsize=8,
                    title_fontsize=9,
                )
                fig.tight_layout()
                plt.show()

                assert np.all(np.isfinite(mean_asymmetry))
                assert np.all(sem_asymmetry >= 0)
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

                - the curve labels $\theta/\pi=0.30,\ldots,0.50$ correspond to
                  the tilt
                  $$
                  |\varphi_\theta\rangle
                  =\cos[(\theta+\pi/2)/2]|\xi\rangle^{\otimes N_s/2}
                  +\sin[(\theta+\pi/2)/2]|0\rangle^{\otimes N_s};
                  $$
                - the displayed axis uses
                  $t=0.1+0.2(\text{Floquet layer}-1)$.

                These conventions are documented explicitly because they are
                not stated in the manuscript's printed equations or its public
                raw data. The physics calculation itself is full SU(2):
                isotropic gates, invariant singlet bath, all total-spin sectors,
                and the exact non-Abelian Haar twirl. The numerical commutator
                tests above make that distinction executable.
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
