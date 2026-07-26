"""Build the data-backed notebooks for manuscript Figs. 9--11."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def _source(text: str) -> str:
    return dedent(text).strip() + "\n"


def md(text: str):
    return nbf.v4.new_markdown_cell(_source(text))


def code(text: str):
    return nbf.v4.new_code_cell(_source(text))


def write_notebook(filename: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, ROOT / filename)


COMMON_IMPORTS = r"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
                # Random circuits III: non-Abelian SU(2) data

                **Manuscript map.** Sec. III G 2, Fig. 11, Eq. (79).

                This notebook analyzes an actual $N_s=8,N_e=12$ simulation. Every
                realization evolves two complex state vectors of length $2^{20}$
                through the original partial-swap brickwork gates, then constructs
                all five Eq. (79) states by linearity. The relative entropy of
                asymmetry is evaluated with the exact SU(2) Schur decomposition, not
                a finite Monte Carlo group average.

                The stored production dataset contains 100 complete realizations,
                matching both the manuscript Hilbert-space size and the Fig. 11
                ensemble size.
                """
            ),
            code(COMMON_IMPORTS),
            md(
                r"""
                ## 1. Load the long state-vector run

                `REGENERATE_DATA=True` reruns all 100 exact-size realizations and
                can take several hours. The equivalent command-line run is
                `python tools/generate_circuit_data.py --only su2 --paper-scale`.
                """
            ),
            code(
                r"""
                DATA_PATH = Path("data/circuit_examples/su2_nonmarkovian.npz")
                REGENERATE_DATA = False

                if REGENERATE_DATA:
                    generated = cd.generate_su2_nonmarkovian(
                        n_realizations=100,
                        coefficient_order="figure",
                    )
                    cd.save_dataset(DATA_PATH, generated)

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        f"{DATA_PATH} is missing. Run "
                        "`python tools/generate_circuit_data.py --only su2`."
                    )
                data = load_npz(DATA_PATH)
                asymmetry = data["asymmetry"]
                times = data["times"]
                theta_over_pi = data["theta_over_pi"]
                labels = [fr"$\theta={theta:.2f}\pi$" for theta in theta_over_pi]

                print(data["protocol"].item())
                print(data["data_level"].item())
                print(
                    f"Ns={data['n_system'].item()}, "
                    f"Ne={data['n_environment'].item()}, "
                    f"R={data['n_realizations'].item()}, "
                    f"T={times[-1]}"
                )
                print("raw asymmetry array:", asymmetry.shape)
                print(
                    "coupling range: [-pi/5, pi/5]; "
                    "environment: product of singlets"
                )
                print(
                    "coefficient convention:",
                    data["coefficient_order"].item(),
                )

                assert asymmetry.shape == (
                    int(data["n_realizations"]), 5, len(times)
                )
                assert np.isfinite(asymmetry).all()
                """
            ),
            md(
                r"""
                ## 2. Exact SU(2) twirl

                The printed Eq. (79) writes the initial state and environment as

                $$
                |\varphi(\theta)\rangle =
                \cos(\theta/2)|\xi\rangle^{\otimes N_s/2}
                +\sin(\theta/2)|0\rangle^{\otimes N_s},\qquad
                |\pi_e\rangle=|\xi\rangle^{\otimes N_e/2},
                $$

                where $|\xi\rangle=(|01\rangle-|10\rangle)/\sqrt2$. The Hilbert
                space decomposes as
                $\mathcal H_s=\bigoplus_j\mathcal M_j\otimes\mathcal V_j$.
                SU(2) twirling preserves the multiplicity state and replaces each
                spin-$j$ irrep by $I_{2j+1}/(2j+1)$. The data generator therefore
                evaluates

                $$
                M_{\rm SU(2)}(\rho_s)
                =S(\mathcal G_{\rm SU(2)}[\rho_s])-S(\rho_s)
                $$

                exactly in the coupled-spin basis.

                The stored production run records `coefficient_order="figure"`:
                it exchanges the two coefficients to reproduce the theta ordering
                visible in Fig. 11. Setting the metadata value to `"equation"`
                instead follows Eq. (79) literally. Keeping this convention
                explicit prevents a silent relabeling of the initial states.
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

                # At t=0 the system is pure and occupies j=0 plus
                # |j=Ns/2,m=j>. This gives an analytic entropy check.
                if data["coefficient_order"].item() == "equation":
                    p_polarized = (
                        np.sin(theta_over_pi * np.pi / 2) ** 2
                    )
                else:
                    p_polarized = (
                        np.cos(theta_over_pi * np.pi / 2) ** 2
                    )
                binary_entropy = -(
                    p_polarized * np.log(p_polarized)
                    + (1 - p_polarized) * np.log(1 - p_polarized)
                )
                analytic_initial = (
                    binary_entropy
                    + p_polarized * np.log(int(data["n_system"]) + 1)
                )
                measured_initial = asymmetry[:, :, 0].mean(axis=0)

                print("analytic M(0):", analytic_initial)
                print("stored   M(0):", measured_initial)
                assert np.allclose(
                    measured_initial, analytic_initial, atol=1e-11
                )
                """
            ),
            md(
                r"""
                ## 3. Raw trajectories and the full ensemble

                The left panel exposes ten individual realizations per angle. The
                right panel computes the mean and 16th-84th percentile interval
                from all 100 circuits. No pre-averaged or illustrative curve is
                loaded.
                """
            ),
            code(
                r"""
                mean_curves = asymmetry.mean(axis=0)
                lower_curves, upper_curves = np.quantile(
                    asymmetry, [0.16, 0.84], axis=0
                )

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
                for state_index, (label, color) in enumerate(
                    zip(labels, PALETTE_5)
                ):
                    for realization in range(10):
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
                    title="ten raw exact-size realizations",
                )
                axes[1].set(title="100-circuit mean and 16-84% interval")
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                plt.show()
                """
            ),
            md(
                r"""
                ## 4. Crossings and non-Markovian backflow

                We test all pairwise mean-curve crossings and quantify
                non-Markovian revivals. A missing crossing is reported as such; it is
                not replaced by an illustrative curve. Both quantities below are
                evaluated from the complete 100-realization dataset.
                """
            ),
            code(
                r"""
                finite_crossings = []
                for high in range(1, len(theta_over_pi)):
                    for low in range(high):
                        tau = nu.crossing_time(
                            times, mean_curves[high], mean_curves[low]
                        )
                        if np.isfinite(tau):
                            finite_crossings.append(
                                (theta_over_pi[high], theta_over_pi[low], tau)
                            )

                if finite_crossings:
                    for high, low, tau in finite_crossings:
                        print(
                            f"theta={high:.2f}pi crosses "
                            f"theta={low:.2f}pi at t={tau:.2f}"
                        )
                else:
                    print(
                        "No pairwise crossing in the 100-realization mean."
                    )

                increments = np.diff(asymmetry, axis=-1)
                print(
                    "largest raw one-step revival:", increments.max()
                )
                print(
                    "fraction of raw steps with positive increment:",
                    (increments > 1e-10).mean(),
                )
                assert increments.max() > 0
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
                    title="mean increments: positive values are backflow",
                )
                ax.legend(ncol=3, fontsize=8)
                plt.show()
                """
            ),
            md(
                r"""
                ## Takeaway

                The notebook executes the exact symmetry construction and
                visualizes the complete 100-circuit, $2^{20}$-dimensional
                production ensemble. The mean crossings, uncertainty band, and
                finite-environment revivals are all calculated from those raw
                trajectories.
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
