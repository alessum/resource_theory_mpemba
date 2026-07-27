"""Build the clean, pedagogical notebooks in the repository root.

The notebooks are generated from plain Python strings so their structure,
metadata, and presentation stay consistent.  Execute this script from the
repository root after editing notebook prose or code below.
"""

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
import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

_TOOLS_DIR = (Path.cwd() / "tools").resolve()
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import notebook_utils as nu

warnings.filterwarnings("ignore", category=RuntimeWarning)
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
GREEN = "#3A7D44"
PURPLE = "#7C4D9E"
"""


write_notebook(
    "atherm_ex1.ipynb",
    [
        md(
            r"""
            # Athermality, Example 1: thermalization of a classical spin chain

            **Manuscript map.** Sec. II B, Fig. 2, Eqs. (11)-(18) of
            *Resource-Theoretical Unification of Mpemba Effects: Classical and Quantum*
            (Summer *et al.*, PRX **16**, 011065 (2026)).

            This notebook builds the seven-spin Markov generator, identifies its
            slow modes, constructs a spectrum-preserving optimized state, and
            reproduces the thermal Mpemba crossing with the KL and Renyi
            divergences. The full calculation is deterministic (fixed random seed)
            and takes a few seconds.

            **Learning goals**

            1. Interpret athermality as the distance from the Gibbs distribution.
            2. See why overlap with the slowest Liouvillian modes controls late-time decay.
            3. Verify that the crossing time depends on the chosen Renyi monotone.

            The longer production script and stored arrays used for the paper live in
            `asymmetry_and_mpemba/thermal_mpemba/`.
            """
        ),
        md(
            r"""
            ## 1. Resource-theory statement

            The unique free state is the Gibbs distribution

            $$
            \pi_{\beta,i}=\frac{e^{-\beta E_i}}{Z_\beta},
            $$

            and the free dynamics is the detailed-balance Markov semigroup
            $\mathbf p(t)=e^{tL_{\rm cl}}\mathbf p(0)$. We quantify athermality by

            $$
            M[\mathbf p(t)] = D(\mathbf p(t)\Vert\boldsymbol\pi_\beta)
            =\sum_i p_i(t)\log\frac{p_i(t)}{\pi_{\beta,i}}.
            $$

            A Mpemba effect occurs when the initially more athermal distribution
            later has the smaller value of this monotone.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 2. Fixed-boundary Ising model and Markov generator

            A microstate is a string $s_n\in\{-1,+1\}$. Following Eq. (12), its energy is

            $$
            E_i=-J\left(\sum_{n=1}^{N_s-1}s_ns_{n+1}+s_1+s_{N_s}\right)
                -h\sum_{n=1}^{N_s}s_n .
            $$

            Only single-spin flips are allowed. The off-diagonal rate
            $L_{ii'}=\exp[-\beta(E_i-E_{i'})/2]$ obeys detailed balance; each
            diagonal entry is then fixed by probability conservation.
            """
        ),
        code(
            r"""
            N_SITES = 7
            J = -0.4
            H = 0.2
            BETA_INITIAL = 0.5
            BETA_BATH = 1.0

            L, energies, configurations, pi = nu.classical_ising_generator(
                N_SITES, J, H, BETA_BATH
            )
            eigenvalues, right_modes = nu.weighted_eigendecomposition(L, pi)

            assert np.allclose(L.sum(axis=0), 0, atol=1e-12)
            assert np.allclose(L @ pi, 0, atol=1e-12)
            assert np.allclose(
                L * pi[None, :], L.T * pi[:, None], atol=1e-12
            )

            print(f"Number of microstates: {len(pi)}")
            print("Ten eigenvalues closest to zero:")
            print(eigenvalues[:10])
            """
        ),
        md(
            r"""
            ## 3. Initial states and slow-mode optimization

            The reference state is a Gibbs distribution at $\beta'=0.5$. The
            optimized state is a permutation of exactly the same probabilities, so
            the two initial states have the same Shannon entropy. A simulated
            annealing search minimizes Eq. (13),

            $$
            C=\sum_{k=2}^{3}\left|\langle\mathbf p,\mathbf r_k\rangle_{\pi_\beta}\right|,
            \qquad
            \langle\mathbf p,\mathbf r_k\rangle_{\pi_\beta}
            =\sum_i\frac{p_i r_{k,i}}{\pi_{\beta,i}} .
            $$

            Keeping this optimization visible is important: the Mpemba state is not
            another thermal state, but a deliberately rearranged nonequilibrium state.
            """
        ),
        code(
            r"""
            p_thermal = np.exp(-BETA_INITIAL * (energies - energies.min()))
            p_thermal /= p_thermal.sum()

            p_optimized, cost_history = nu.optimize_probability_permutation(
                p_thermal,
                pi,
                right_modes[:, 1:3],
                iterations=50_000,
                seed=12,
            )

            def overlaps(probability):
                return (right_modes.T / pi) @ probability

            overlap_thermal = overlaps(p_thermal)
            overlap_optimized = overlaps(p_optimized)

            print(f"Cost before optimization: {cost_history[0]:.3e}")
            print(f"Cost after optimization:  {cost_history[-1]:.3e}")
            print("First five nonstationary overlaps:")
            print("thermal  :", np.abs(overlap_thermal[1:6]))
            print("optimized:", np.abs(overlap_optimized[1:6]))
            assert np.allclose(np.sort(p_thermal), np.sort(p_optimized))
            """
        ),
        md(
            r"""
            ## 4. Time evolution and the KL crossing

            The detailed-balance eigenmodes are orthonormal in the
            $\pi_\beta$-weighted inner product, so the complete time evolution follows
            directly from their spectral expansion. At late times,

            $$
            D(\mathbf p(t)\Vert\boldsymbol\pi_\beta)
            \sim \tfrac12 |a_j|^2 e^{2\lambda_j t},
            $$

            where $j$ is the slowest mode with nonzero initial overlap.
            """
        ),
        code(
            r"""
            times = np.linspace(0, 6, 2_001)
            inverse_modes = la.inv(right_modes)

            def evolve(probability):
                coefficients = inverse_modes @ probability
                return np.real(
                    right_modes
                    @ (
                        coefficients[:, None]
                        * np.exp(eigenvalues[:, None] * times[None, :])
                    )
                )

            evolution_thermal = evolve(p_thermal)
            evolution_optimized = evolve(p_optimized)
            kl_thermal = np.array([
                nu.classical_relative_entropy(evolution_thermal[:, i], pi)
                for i in range(len(times))
            ])
            kl_optimized = np.array([
                nu.classical_relative_entropy(evolution_optimized[:, i], pi)
                for i in range(len(times))
            ])
            tau_kl = nu.crossing_time(times, kl_optimized, kl_thermal)

            print(f"Initial KL: thermal={kl_thermal[0]:.4f}, "
                  f"optimized={kl_optimized[0]:.4f}")
            print(f"KL crossing time: tau = {tau_kl:.4f}")
            assert kl_optimized[0] > kl_thermal[0]
            assert np.isfinite(tau_kl)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

            axes[0].semilogy(times, kl_thermal, color=PURPLE, lw=2,
                            label="thermal initial state")
            axes[0].semilogy(times, kl_optimized, color=PURPLE, lw=2, ls="--",
                            label="optimized permutation")
            axes[0].axvline(tau_kl, color="0.25", lw=1, ls=":",
                            label=fr"$\tau={tau_kl:.3f}$")
            axes[0].set(xlabel=r"$t$", ylabel=r"$D(\mathbf{p}(t)\Vert\pi_\beta)$",
                        title="Thermal Mpemba crossing")
            axes[0].legend()

            axes[1].plot(cost_history, color=ORANGE, lw=1)
            axes[1].set_yscale("log")
            axes[1].set(xlabel="accepted/proposed update", ylabel=r"$C_{\rm best}$",
                        title="Slow-mode optimization")

            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 5. Renyi-monotone dependence

            Replacing KL divergence by

            $$
            D_\alpha(\mathbf p\Vert\boldsymbol\pi_\beta)
            =\frac{1}{\alpha-1}\log\sum_i p_i^\alpha\pi_{\beta,i}^{1-\alpha}
            $$

            changes the numerical crossing time even though the states and dynamics
            are unchanged. This is why the paper treats the crossing time as a
            monotone-dependent diagnostic, not an intrinsic relaxation timescale.
            """
        ),
        code(
            r"""
            alphas = np.linspace(0.1, 1.9, 25)
            crossing_times = []
            for alpha in alphas:
                d_thermal = np.array([
                    nu.classical_renyi(evolution_thermal[:, i], pi, alpha)
                    for i in range(len(times))
                ])
                d_optimized = np.array([
                    nu.classical_renyi(evolution_optimized[:, i], pi, alpha)
                    for i in range(len(times))
                ])
                crossing_times.append(
                    nu.crossing_time(times, d_optimized, d_thermal)
                )

            fig, ax = plt.subplots()
            ax.plot(alphas, crossing_times, marker="o", ms=3, color=ORANGE)
            ax.axvline(1, color="0.4", ls=":", lw=1)
            ax.set(xlabel=r"Renyi parameter $\alpha$",
                   ylabel=r"crossing time $\tau_\alpha$",
                   title="The crossing depends on the monotone")
            plt.show()
            """
        ),
        md(
            r"""
            ## 6. Reproduction check against the archived paper arrays

            The repository includes the high-budget production output. The following
            check does not replace the calculation above; it confirms that the
            notebook has reproduced the same ordering, crossing, and late-time
            speedup.
            """
        ),
        code(
            r"""
            data_dir = Path(
                "asymmetry_and_mpemba/thermal_mpemba/data_for_figures/"
                "thermal_mpemba_classical_figure"
            )
            paper_t = np.load(data_dir / "time_list.npy")
            paper_thermal = np.load(data_dir / "KL_distance.npy")
            paper_optimized = np.load(data_dir / "KL_distance_2.npy")
            paper_tau = nu.crossing_time(
                paper_t, paper_optimized, paper_thermal
            )

            print(f"Notebook crossing: {tau_kl:.4f}")
            print(f"Archived production crossing: {paper_tau:.4f}")
            print("Both runs show the same strong Mpemba ordering:",
                  paper_optimized[0] > paper_thermal[0]
                  and paper_optimized[-1] < paper_thermal[-1])
            """
        ),
        md(
            r"""
            ## Takeaway

            The optimized distribution begins farther from equilibrium, but its
            overlap with the two slowest modes is suppressed by roughly three orders
            of magnitude. Its late-time decay is therefore governed by a faster
            eigenvalue, forcing a Mpemba crossing. The mechanism is spectral; the
            reported crossing time is specific to the chosen resource monotone.
            """
        ),
    ],
)


write_notebook(
    "atherm_ex2.ipynb",
    [
        md(
            r"""
            # Athermality, Example 2: a single-qubit Davies map

            **Manuscript map.** Sec. II C, Fig. 3, Eqs. (19)-(24).

            This notebook gives a self-contained reproduction of the quantum thermal
            Mpemba example. It uses the two Bloch vectors and parameters quoted in
            the Fig. 3 caption, evolves them analytically under a phase-covariant
            Davies generator, and evaluates both quantum relative entropy and the
            Petz-Renyi family.

            **Learning goals**

            1. Separate population relaxation from coherence decay.
            2. Understand why a diagonal, highly excited state can thermalize faster
               than a less athermal coherent state.
            3. Relate the late-time slopes to the complex Liouvillian spectrum.
            """
        ),
        md(
            r"""
            ## 1. Model and free-energy monotone

            For $\hat H=(h/2)\sigma_z$, the fixed point is

            $$
            \hat\pi_\beta=\frac{e^{-\beta\hat H}}{\operatorname{Tr}e^{-\beta\hat H}}.
            $$

            The dimensionless excess free energy is exactly the relative entropy of
            athermality,

            $$
            M[\hat\rho(t)] = S(\hat\rho(t)\Vert\hat\pi_\beta)
            =\beta\,[F(\hat\rho(t))-F(\hat\pi_\beta)].
            $$

            The Davies map has one population rate $\Gamma_1$ and a conjugate pair of
            coherence eigenvalues $-\Gamma_2\pm ih$. We choose
            $\Gamma_1=2\Gamma_2$, the standard no-extra-dephasing relation.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            r"""
            H_FREQUENCY = 10.0
            BETA = 0.1
            GAMMA_2 = 1.0
            GAMMA_1 = 2.0 * GAMMA_2

            r_random = np.array([0.221, 0.867, 0.206])
            r_optimized = np.array([0.0, 0.0, 0.919])

            def rho_from_bloch(vector):
                return (
                    nu.I2
                    + vector[0] * nu.X
                    + vector[1] * nu.Y
                    + vector[2] * nu.Z
                ) / 2

            z_equilibrium = -np.tanh(BETA * H_FREQUENCY / 2)
            pi = rho_from_bloch([0, 0, z_equilibrium])
            rho_1 = rho_from_bloch(r_random)
            rho_2 = rho_from_bloch(r_optimized)

            for state in (pi, rho_1, rho_2):
                nu.validate_density(state)

            print("Equilibrium Bloch z:", z_equilibrium)
            print("Liouvillian spectrum:",
                  [0, -GAMMA_1,
                   -GAMMA_2 + 1j * H_FREQUENCY,
                   -GAMMA_2 - 1j * H_FREQUENCY])
            """
        ),
        md(
            r"""
            ## 2. Analytic Davies evolution

            In Bloch coordinates,

            $$
            r_z(t)=r_z^{\rm eq}+[r_z(0)-r_z^{\rm eq}]e^{-\Gamma_1 t},
            \qquad
            r_x(t)+ir_y(t)=[r_x(0)+ir_y(0)]e^{(-\Gamma_2+ih)t}.
            $$

            The optimized state has no overlap with the slow coherent pair. Its
            athermality is initially larger because it is strongly population
            inverted, but it decays with the faster population rate.
            """
        ),
        code(
            r"""
            times = np.linspace(0, 5, 2_001)

            def evolve_bloch(vector, time):
                rotation = H_FREQUENCY * time
                transverse = np.exp(-GAMMA_2 * time)
                x = transverse * (
                    vector[0] * np.cos(rotation)
                    - vector[1] * np.sin(rotation)
                )
                y = transverse * (
                    vector[0] * np.sin(rotation)
                    + vector[1] * np.cos(rotation)
                )
                z = z_equilibrium + (
                    vector[2] - z_equilibrium
                ) * np.exp(-GAMMA_1 * time)
                return rho_from_bloch([x, y, z])

            states_1 = np.array([evolve_bloch(r_random, t) for t in times])
            states_2 = np.array([evolve_bloch(r_optimized, t) for t in times])
            monotone_1 = np.array([
                nu.quantum_relative_entropy(state, pi) for state in states_1
            ])
            monotone_2 = np.array([
                nu.quantum_relative_entropy(state, pi) for state in states_2
            ])
            tau = nu.crossing_time(times, monotone_2, monotone_1)

            print(f"M(rho_1, 0) = {monotone_1[0]:.4f}")
            print(f"M(rho_2, 0) = {monotone_2[0]:.4f}")
            print(f"Crossing time = {tau:.4f}")
            assert monotone_2[0] > monotone_1[0]
            assert np.isfinite(tau)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

            axes[0].semilogy(times, monotone_1, color=PURPLE, lw=2,
                            label=r"$\rho_1$: coherent")
            axes[0].semilogy(times, monotone_2, color=PURPLE, lw=2, ls="--",
                            label=r"$\rho_2$: optimized")
            axes[0].axvline(tau, color="0.25", lw=1, ls=":",
                            label=fr"$\tau={tau:.3f}$")
            axes[0].set(xlabel=r"$t$", ylabel=r"$S(\rho(t)\Vert\pi_\beta)$",
                        xlim=(0, 1.5), title="Quantum thermal Mpemba effect")
            axes[0].legend()

            spectrum = np.array([
                0,
                -GAMMA_1,
                -GAMMA_2 + 1j * H_FREQUENCY,
                -GAMMA_2 - 1j * H_FREQUENCY,
            ])
            axes[1].scatter(spectrum.real[:2], spectrum.imag[:2],
                            color=GREEN, label="populations")
            axes[1].scatter(spectrum.real[2:], spectrum.imag[2:],
                            color=ORANGE, label="coherences")
            axes[1].axhline(0, color="0.6", lw=1)
            axes[1].set(xlabel=r"$\operatorname{Re}\lambda$",
                        ylabel=r"$\operatorname{Im}\lambda$",
                        title="Davies spectrum")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. Petz-Renyi crossing times

            The same pair of trajectories can be quantified with
            $D_\alpha(\rho\Vert\pi_\beta)$. Nothing in the physical evolution
            changes, but the inferred crossing time does.
            """
        ),
        code(
            r"""
            alphas = np.linspace(0.1, 2.0, 28)
            sample = slice(None, None, 5)
            sampled_times = times[sample]
            crossing_times = []
            for alpha in alphas:
                d1 = np.array([
                    nu.petz_renyi(state, pi, alpha) for state in states_1[sample]
                ])
                d2 = np.array([
                    nu.petz_renyi(state, pi, alpha) for state in states_2[sample]
                ])
                crossing_times.append(nu.crossing_time(sampled_times, d2, d1))

            fig, ax = plt.subplots()
            ax.plot(alphas, crossing_times, color=ORANGE, marker="o", ms=3)
            ax.axvline(1, color="0.4", ls=":", lw=1)
            ax.set(xlabel=r"Petz-Renyi parameter $\alpha$",
                   ylabel=r"crossing time $\tau_\alpha$",
                   title="Monotone-dependent crossing time")
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            The diagonal state $\rho_2$ starts farther from the Gibbs state but has
            exactly zero overlap with the slow coherence modes. It therefore relaxes
            with $\Gamma_1=2\Gamma_2$ and crosses the coherent trajectory. The
            complex eigenvalues are not decorative: their real part is the slow
            envelope that the optimized state avoids.
            """
        ),
    ],
)


write_notebook(
    "asymmetry_modes_example.ipynb",
    [
        md(
            r"""
            # Modes of asymmetry: tilted three-qubit states

            **Manuscript map.** Sec. III C, Fig. 4, Eqs. (48)-(53).

            Before studying symmetry Mpemba effects, this notebook visualizes the
            mode decomposition itself. For U(1) rotations generated by total
            magnetization $S_z$, operator space splits into charges
            $\mu\in\{0,\pm1,\pm2,\pm3\}$. A matrix element
            $|m\rangle\langle m'|$ belongs to mode $\mu=m-m'$.

            We prepare

            $$
            |\varphi(\theta)\rangle
            =\bigotimes_{n=1}^3 e^{-i\theta\sigma_y/2}|0\rangle_n
            $$

            for $\theta=0,\pi/4,\pi/2$ and inspect which modes are populated.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. State preparation and mode projectors

            The U(1) twirl deletes matrix elements connecting different
            magnetization sectors. The full state can be recovered from its modes:

            $$
            \rho=\sum_{\mu=-N_s}^{N_s}\rho^{(\mu)},\qquad
            \mathcal G[\rho]=\rho^{(0)}.
            $$
            """
        ),
        code(
            r"""
            N_SITES = 3
            THETAS = [0, np.pi / 4, np.pi / 2]

            def tilted_product_state(theta):
                ket = np.array([1.0 + 0j, 0j])
                single_qubit = la.expm(-0.5j * theta * nu.Y) @ ket
                state = nu.kron_all([single_qubit] * N_SITES)
                return np.outer(state, state.conj())

            states = [tilted_product_state(theta) for theta in THETAS]
            modes = [nu.asymmetry_modes_u1(state, N_SITES) for state in states]

            for state, mode_dict in zip(states, modes):
                reconstructed = sum(mode_dict.values())
                assert np.allclose(reconstructed, state)
                assert np.allclose(mode_dict[0], nu.u1_twirl(state, N_SITES))
            """
        ),
        md(
            r"""
            ## 2. Rescaled mode weights

            As in Eq. (53), we use the trace norm of each mode and normalize it by
            the corresponding weight of the maximally tilted state:

            $$
            p_\theta^{(\mu)}
            =\frac{\|\rho_\theta^{(\mu)}\|_1}
                   {\|\rho_{\pi/2}^{(\mu)}\|_1}.
            $$
            """
        ),
        code(
            r"""
            charges = np.arange(-N_SITES, N_SITES + 1)
            raw_weights = np.array([
                [np.sum(la.svdvals(mode_dict[charge])) for charge in charges]
                for mode_dict in modes
            ])
            rescaled_weights = raw_weights / raw_weights[-1]

            print("Rows: theta = 0, pi/4, pi/2; columns: mu = -3,...,3")
            print(rescaled_weights)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
            colors = [BLUE, ORANGE, "#F4C95D"]

            for column, (theta, state, weights, color) in enumerate(
                zip(THETAS, states, rescaled_weights, colors)
            ):
                axes[0, column].bar(charges, weights, color=color)
                axes[0, column].set(
                    title=fr"$\theta={theta / np.pi:.2g}\pi$",
                    xlabel=r"mode $\mu$",
                    ylabel=r"$p_\theta^{(\mu)}$" if column == 0 else None,
                    ylim=(0, 1.08),
                    xticks=charges,
                )

                image = axes[1, column].imshow(
                    np.abs(state), cmap="magma", vmin=0, vmax=1
                )
                axes[1, column].set(
                    xlabel="basis column",
                    ylabel="basis row" if column == 0 else None,
                )

            fig.suptitle("U(1) mode occupancy and density-matrix support", y=1.02)
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            At $\theta=0$, only $\mu=0$ is present and the state is symmetric.
            Tilting creates coherences between magnetization sectors, populating
            nonzero modes. At $\theta=\pi/2$, every allowed charge is occupied. The
            symmetry Mpemba mechanism will exploit the fact that different
            nonzero-$\mu$ sectors can decay at different rates.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex1.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 1: a classical Z4 symmetry

            **Manuscript map.** Sec. III D, Fig. 5, Eqs. (54)-(58).

            A four-state continuous-time Markov chain lives on a ring. Its generator
            is covariant under the cyclic shift $n\mapsto n+1\pmod 4$, so each
            Fourier mode belongs to a single irreducible representation of
            $\mathbb Z_4$.

            We compare a weakly asymmetric distribution supported in the slow
            $\mu=1,3$ sectors with a more asymmetric distribution supported only in
            the fast $\mu=2$ sector. This is the smallest fully classical symmetry
            Mpemba example in the paper.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. Generator and Fourier modes

            With chirality $\varepsilon$, Eq. (54) has eigenvalues

            $$
            \lambda_0=0,\qquad
            \lambda_{1,3}=-2\pm 2i\varepsilon,\qquad
            \lambda_2=-4.
            $$

            The $\mu=2$ component therefore decays twice as fast in amplitude as the
            $\mu=1,3$ pair.
            """
        ),
        code(
            r"""
            EPSILON = 0.25
            L = np.array([
                [-2, 1 + EPSILON, 0, 1 - EPSILON],
                [1 - EPSILON, -2, 1 + EPSILON, 0],
                [0, 1 - EPSILON, -2, 1 + EPSILON],
                [1 + EPSILON, 0, 1 - EPSILON, -2],
            ], dtype=float)

            phase = np.exp(2j * np.pi / 4)
            fourier_modes = {
                mu: np.array([phase ** (mu * n) for n in range(4)]) / 2
                for mu in range(4)
            }
            eigenvalues = {
                0: 0,
                1: -2 + 2j * EPSILON,
                2: -4,
                3: -2 - 2j * EPSILON,
            }

            for mu in range(4):
                assert np.allclose(
                    L @ fourier_modes[mu],
                    eigenvalues[mu] * fourier_modes[mu],
                )
            assert np.allclose(L.sum(axis=0), 0)
            """
        ),
        md(
            r"""
            ## 2. Initial distributions and asymmetry monotone

            Eq. (57) chooses

            $$
            \mathbf p_1=\tfrac12\mathbf r_0
            +\tfrac14\mathbf r_1+\tfrac14\mathbf r_3,
            \qquad
            \mathbf p_2=\tfrac12\mathbf r_0+\tfrac12\mathbf r_2.
            $$

            Here $\mathbf p_1$ is the slow-sector state and $\mathbf p_2$ is the
            initially more asymmetric fast-sector state. The group twirl sends every
            distribution to the uniform vector, so

            $$
            M[\mathbf p]=D(\mathbf p\Vert\mathcal G\mathbf p)
            =\sum_n p_n\log(4p_n).
            $$
            """
        ),
        code(
            r"""
            p_1 = (
                fourier_modes[0] / 2
                + fourier_modes[1] / 4
                + fourier_modes[3] / 4
            ).real
            p_2 = (
                fourier_modes[0] / 2
                + fourier_modes[2] / 2
            ).real
            uniform = np.full(4, 0.25)

            def asymmetry(probability):
                return nu.classical_relative_entropy(probability, uniform)

            for probability in (p_1, p_2):
                assert np.isclose(probability.sum(), 1)
                assert np.all(probability >= 0)

            print("p_1 (slow sectors):", p_1)
            print("p_2 (fast sector): ", p_2)
            print("Initial asymmetries:", asymmetry(p_1), asymmetry(p_2))
            """
        ),
        md(
            r"""
            ## 3. Dynamics and Mpemba crossing

            The relative entropy is quadratic close to the symmetric state. Hence
            Eq. (58) predicts

            $$
            M[\mathbf p_1(t)]\sim\tfrac14e^{-4t},
            \qquad
            M[\mathbf p_2(t)]\sim\tfrac12e^{-8t}.
            $$
            """
        ),
        code(
            r"""
            times = np.linspace(0, 2, 600)
            evolution_1 = np.array([la.expm(L * t) @ p_1 for t in times])
            evolution_2 = np.array([la.expm(L * t) @ p_2 for t in times])
            M_1 = np.array([asymmetry(p) for p in evolution_1])
            M_2 = np.array([asymmetry(p) for p in evolution_2])
            tau = nu.crossing_time(times, M_2, M_1)

            print(f"Crossing time: {tau:.4f}")
            assert M_2[0] > M_1[0]
            assert np.isfinite(tau)
            """
        ),
        code(
            r"""
            positions = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
            fig = plt.figure(figsize=(11, 4))
            grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.2])

            for index, (probability, title) in enumerate(
                [(p_1, r"$\mathbf{p}_1$: slow"), (p_2, r"$\mathbf{p}_2$: fast")]
            ):
                ax = fig.add_subplot(grid[0, index])
                square = np.vstack([positions, positions[0]])
                ax.plot(square[:, 0], square[:, 1], color="0.4")
                ax.scatter(
                    positions[:, 0], positions[:, 1],
                    s=1_100 * probability + 30,
                    c=probability, cmap="Oranges", vmin=0, vmax=0.5,
                    edgecolor="0.2",
                )
                for site, (x, y) in enumerate(positions):
                    ax.text(x, y, str(site), ha="center", va="center")
                ax.set(aspect="equal", title=title)
                ax.axis("off")

            ax = fig.add_subplot(grid[0, 2])
            ax.semilogy(times, M_1, color=BLUE, lw=2,
                        label=r"$M[\mathbf{p}_1(t)]$")
            ax.semilogy(times, M_2, color=ORANGE, lw=2,
                        label=r"$M[\mathbf{p}_2(t)]$")
            ax.semilogy(times, 0.25 * np.exp(-4 * times), color=BLUE,
                        ls="--", alpha=0.55, label=r"$e^{-4t}/4$")
            ax.semilogy(times, 0.5 * np.exp(-8 * times), color=ORANGE,
                        ls="--", alpha=0.55, label=r"$e^{-8t}/2$")
            ax.axvline(tau, color="0.25", ls=":", lw=1)
            ax.set(xlabel=r"$t$", ylabel=r"$M[\mathbf{p}(t)]$",
                   title="Classical symmetry Mpemba crossing")
            ax.legend(ncol=2, fontsize=9)
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            Chirality produces oscillatory phases in the $\mu=1,3$ pair, but their
            real decay rate remains $-2$. The more asymmetric $\mu=2$ state avoids
            those slow symmetry-restoring sectors and loses its resource at twice the
            amplitude rate, creating the crossing.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex2.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 2: a quantum Z4 symmetry

            **Manuscript map.** Sec. III E, Fig. 6, Eqs. (59)-(64).

            A single particle hops coherently and incoherently on a four-site ring.
            The Lindblad generator commutes with cyclic translations and therefore
            splits into four $\mathbb Z_4$ charge blocks. We construct two physical
            density matrices with controlled block support and show that the state
            concentrated in a faster sector restores symmetry first.

            This implementation deliberately fixes eigenoperator phases by
            hermitizing the selected modes. That avoids the phase-dependent,
            non-Hermitian states produced by the original exploratory notebook on
            some SciPy versions.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. Covariant Lindbladian and irreducible tensor basis

            The Hamiltonian is nearest-neighbor hopping with $J=1$. Biased left/right
            jumps have rates $1\pm\varepsilon$, with $\varepsilon=1/4$.
            In the tensor basis $\{T_\alpha^{(\mu)}\}$ of Eq. (59), covariance requires
            the superoperator to be block diagonal in $\mu$.
            """
        ),
        code(
            r"""
            N = 4
            EPSILON = 0.25
            HOPPING = 1.0

            L = nu.quantum_ring_liouvillian(N, EPSILON, HOPPING)
            tensor_basis = nu.z4_fourier_tensor_basis(N)
            B = np.column_stack([nu.vec(operator) for operator in tensor_basis])
            L_blocks_basis = la.inv(B) @ L @ B

            gram = B.conj().T @ B
            block_mask = np.zeros_like(L_blocks_basis, dtype=bool)
            for mu in range(N):
                block_mask[mu * N : (mu + 1) * N,
                           mu * N : (mu + 1) * N] = True
            leakage = la.norm(np.where(block_mask, 0, L_blocks_basis))

            print(f"Basis orthonormality error: {la.norm(gram - np.eye(N**2)):.2e}")
            print(f"Off-block Liouvillian norm: {leakage:.2e}")
            assert leakage < 1e-10
            """
        ),
        code(
            r"""
            block_eigenvalues = {}
            block_eigenvectors = {}
            for mu in range(N):
                block = L_blocks_basis[
                    mu * N : (mu + 1) * N,
                    mu * N : (mu + 1) * N,
                ]
                values, vectors = la.eig(block)
                block_eigenvalues[mu] = values
                block_eigenvectors[mu] = vectors
                print(
                    f"mu={mu}:",
                    np.array([
                        complex(round(value.real, 4), round(value.imag, 4))
                        for value in values
                    ]),
                )
            """
        ),
        md(
            r"""
            ## 2. Phase-stable slow- and fast-sector states

            We mirror the design of Eq. (63):

            - $\rho_1$ has modest support in the slow conjugate sectors
              $\mu=1,3$, whose eigenvalues have real part $-2$.
            - $\rho_2$ has larger initial asymmetry but support in a fast,
              real-eigenvalue mode of the self-conjugate $\mu=2$ sector.

            Each traceless Hermitian mode is added to the maximally mixed steady
            state. Its amplitude is chosen as a fixed fraction of the positivity
            boundary, so physicality is guaranteed independently of eigenvector phase.
            """
        ),
        code(
            r"""
            def embedded_eigenoperator(mu, column):
                coefficients = np.zeros(N**2, dtype=complex)
                coefficients[mu * N : (mu + 1) * N] = (
                    block_eigenvectors[mu][:, column]
                )
                return (B @ coefficients).reshape(N, N, order="F")

            # A slow mu=1 mode and its adjoint (which lies in mu=3).
            slow_column = int(np.argmax(block_eigenvalues[1].imag))
            slow_raw = embedded_eigenoperator(1, slow_column)
            slow_mode = nu.hermitize(slow_raw + slow_raw.conj().T)
            slow_mode -= np.trace(slow_mode) * np.eye(N) / N
            slow_mode /= la.norm(slow_mode)

            # The fastest real eigenmode in the self-conjugate mu=2 block.
            real_columns = np.flatnonzero(
                np.abs(block_eigenvalues[2].imag) < 1e-8
            )
            fast_column = real_columns[
                np.argmin(block_eigenvalues[2][real_columns].real)
            ]
            fast_mode = nu.hermitize(
                embedded_eigenoperator(2, int(fast_column))
            )
            fast_mode -= np.trace(fast_mode) * np.eye(N) / N
            fast_mode /= la.norm(fast_mode)

            def state_from_mode(mode, positivity_fraction):
                lower_bound = (1 / N) / (-la.eigvalsh(mode).min())
                return np.eye(N) / N + positivity_fraction * lower_bound * mode

            rho_1 = state_from_mode(slow_mode, positivity_fraction=0.30)
            rho_2 = state_from_mode(fast_mode, positivity_fraction=0.70)
            for state in (rho_1, rho_2):
                nu.validate_density(state)

            M0 = [
                nu.asymmetry_relative_entropy(state, nu.z4_twirl(state))
                for state in (rho_1, rho_2)
            ]
            print("Initial asymmetry [rho_1, rho_2]:", M0)
            print("Selected fast eigenvalue:",
                  block_eigenvalues[2][fast_column])
            assert M0[1] > M0[0]
            """
        ),
        md(
            r"""
            ## 3. Dynamics, sector norms, and crossing

            For every snapshot we compute the relative entropy of asymmetry

            $$
            M[\rho(t)]=S(\rho(t)\Vert\mathcal G[\rho(t)])
            =S(\mathcal G[\rho(t)])-S(\rho(t)),
            $$

            together with the trace norm $\|\rho^{(\mu)}(t)\|_1$ of each nonzero
            mode.
            """
        ),
        code(
            r"""
            times = np.linspace(0, 2, 500)

            def evolve_and_measure(rho0):
                asymmetry = []
                mode_norms = np.zeros((len(times), N))
                for index, time in enumerate(times):
                    rho_t = (la.expm(L * time) @ nu.vec(rho0)).reshape(
                        N, N, order="F"
                    )
                    rho_t = nu.normalize_density(rho_t)
                    asymmetry.append(
                        nu.asymmetry_relative_entropy(rho_t, nu.z4_twirl(rho_t))
                    )
                    for mu, mode in nu.z4_modes(rho_t).items():
                        mode_norms[index, mu] = np.sum(la.svdvals(mode))
                return np.asarray(asymmetry), mode_norms

            M_1, norms_1 = evolve_and_measure(rho_1)
            M_2, norms_2 = evolve_and_measure(rho_2)
            tau = nu.crossing_time(times, M_2, M_1)

            print(f"Crossing time: {tau:.4f}")
            assert np.isfinite(tau)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

            axes[0].semilogy(times, M_1, color=BLUE, lw=2,
                            label=r"$M[\rho_1(t)]$: slow sectors")
            axes[0].semilogy(times, M_2, color=ORANGE, lw=2,
                            label=r"$M[\rho_2(t)]$: fast sector")
            axes[0].axvline(tau, color="0.25", ls=":", lw=1)
            axes[0].set(xlabel=r"$t$", ylabel=r"$M[\rho(t)]$",
                        title=r"Quantum $Z_4$ Mpemba crossing")
            axes[0].legend(fontsize=9)

            for mu in range(1, N):
                if norms_1[:, mu].max() > 1e-10:
                    axes[1].semilogy(
                        times, norms_1[:, mu], color=BLUE,
                        ls=["-", "--", ":"][mu - 1],
                        label=fr"$\rho_1^{{({mu})}}$",
                    )
                if norms_2[:, mu].max() > 1e-10:
                    axes[1].semilogy(
                        times, norms_2[:, mu], color=ORANGE,
                        ls=["-", "--", ":"][mu - 1],
                        label=fr"$\rho_2^{{({mu})}}$",
                    )
            axes[1].set(xlabel=r"$t$", ylabel=r"$\|\rho^{(\mu)}(t)\|_1$",
                        title="Occupied symmetry sectors")
            axes[1].legend(ncol=2, fontsize=9)
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            Covariance makes the Liouvillian block diagonal in symmetry charge. The
            initial amount of asymmetry does not determine the restoration speed by
            itself: the relevant information is *where* that asymmetry sits. The
            more asymmetric $\rho_2$ is concentrated in a block with a more negative
            decay exponent and therefore crosses below $\rho_1$.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex3.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 3: time-translation symmetry in a Davies map

            **Manuscript map.** Sec. III F, Fig. 7, Eqs. (65)-(67).

            A four-spin transverse-field Ising model thermalizes under a Davies
            generator. Because the generator commutes with Hamiltonian evolution,
            operator space splits into Bohr-frequency modes. Populations form the
            $\mu=0$ block; energy-basis coherences form the $\mu\ne0$ blocks.

            We construct two isospectral initial states. A local-unitary search gives
            the second state very small overlap with the slowest coherent mode while
            retaining larger initial asymmetry. The resulting crossing is the
            time-translation-symmetry Mpemba effect.

            **Convention note.** The released simulation script sets a bath
            temperature $T=0.1$ and passes $\beta=1/T=10$ to the Davies generator.
            This notebook follows that executable convention. Set `BETA_BATH=0.1`
            below to study the literal inverse-temperature value printed in the
            figure caption.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. TFIM and Davies spectrum

            The released code represents spin operators as $s^\alpha=\sigma^\alpha/2$:

            $$
            H_s=-J\sum_{n=1}^{N_s-1}s_n^z s_{n+1}^z
                +h\sum_{n=1}^{N_s}s_n^x .
            $$

            The compact generator below is algebraically equivalent to summing the
            rank-one Davies jump operators in the energy basis. Population rates obey
            detailed balance, while each coherence is an independent complex
            eigenmode.
            """
        ),
        code(
            r"""
            N_SITES = 4
            J = 1.0
            H_FIELD = 1.0
            BETA_BATH = 10.0
            BETA_INITIAL = 1.0
            NOISE_STRENGTH = 0.05

            hamiltonian = nu.tfim_hamiltonian(
                N_SITES, J, H_FIELD, spin_half_operators=True
            )
            L, energies, energy_basis, pi = nu.davies_liouvillian(
                hamiltonian, BETA_BATH
            )

            eigenvalues, left_vectors = la.eig(L, left=True, right=False)
            order = np.argsort(np.abs(eigenvalues.real))
            eigenvalues = eigenvalues[order]
            left_vectors = left_vectors[:, order]

            print("Eight eigenvalues with real part closest to zero:")
            print(eigenvalues[:8])
            assert np.allclose(L @ nu.vec(pi), 0, atol=1e-10)
            """
        ),
        md(
            r"""
            ## 2. Initial state and fast local-unitary rotation

            Following Eq. (66),

            $$
            \rho_1=\frac{\rho_{\rm Gibbs}(\beta_i)+\gamma X^\dagger X}
                         {\mathcal N}.
            $$

            The noise matrix and seed match the released script. We identify the
            first nonstationary complex mode and search over products of single-qubit
            Haar unitaries. Among candidates with at most 5% of the original overlap,
            we retain the most asymmetric state. This is a quiet, deterministic
            version of the paper's Metropolis rotation.
            """
        ),
        code(
            r"""
            rho_1 = nu.manuscript_noisy_gibbs_state(
                energies, BETA_INITIAL, NOISE_STRENGTH, seed=0
            )

            coherent_indices = []
            for index, value in enumerate(eigenvalues):
                matrix = nu.unvec(left_vectors[:, index], 2**N_SITES)
                off_diagonal = matrix - np.diag(np.diag(matrix))
                if value.real < -1e-10 and la.norm(off_diagonal) > 1e-8:
                    coherent_indices.append(index)

            slow_index = coherent_indices[0]
            slow_eigenvalue = eigenvalues[slow_index]
            slow_left_mode = nu.unvec(
                left_vectors[:, slow_index], 2**N_SITES
            )
            slow_left_mode /= la.norm(slow_left_mode)

            rho_2, rotation, diagnostics = nu.find_fast_asymmetric_state(
                rho_1,
                slow_left_mode,
                N_SITES,
                samples=12_000,
                cost_fraction=0.05,
                seed=15,
            )

            nu.validate_density(rho_1)
            nu.validate_density(rho_2)
            assert np.allclose(la.eigvalsh(rho_1), la.eigvalsh(rho_2))
            print("Slow coherent eigenvalue:", slow_eigenvalue)
            for key, value in diagnostics.items():
                print(f"{key:>20s}: {value:.6g}")
            """
        ),
        md(
            r"""
            ## 3. Symmetry monotone and crossing

            Time-translation twirling is dephasing in the energy eigenbasis:

            $$
            \mathcal G[\rho]=\operatorname{diag}(\rho).
            $$

            Therefore

            $$
            M[\rho(t)]
            =S(\rho(t)\Vert\mathcal G[\rho(t)])
            =S(\mathcal G[\rho(t)])-S(\rho(t)).
            $$
            """
        ),
        code(
            r"""
            times = np.linspace(0, 10, 1_001)
            states_1 = nu.evolve_density(L, rho_1, times)
            states_2 = nu.evolve_density(L, rho_2, times)

            M_1 = np.array([
                nu.asymmetry_relative_entropy(state, nu.energy_dephase(state))
                for state in states_1
            ])
            M_2 = np.array([
                nu.asymmetry_relative_entropy(state, nu.energy_dephase(state))
                for state in states_2
            ])
            tau = nu.crossing_time(times, M_2, M_1)

            print(f"Initial M: rho_1={M_1[0]:.4f}, rho_2={M_2[0]:.4f}")
            print(f"Crossing time: {tau:.4f}")
            assert M_2[0] > M_1[0]
            assert np.isfinite(tau)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
            axes[0].semilogy(times, M_1, color=ORANGE, lw=2,
                            label=r"$\rho_1$: Gibbs + random")
            axes[0].semilogy(times, M_2, color=ORANGE, lw=2, ls="--",
                            label=r"$\rho_2$: rotated")
            axes[0].axvline(tau, color="0.25", ls=":", lw=1)
            axes[0].set(xlabel=r"$t$", ylabel=r"$S(\rho(t)\Vert\mathcal{G}[\rho(t)])$",
                        title="Time-translation symmetry Mpemba effect")
            axes[0].legend(fontsize=9)

            shown = eigenvalues[:12]
            is_coherent = []
            for index in range(min(12, len(eigenvalues))):
                matrix = nu.unvec(left_vectors[:, index], 2**N_SITES)
                is_coherent.append(
                    la.norm(matrix - np.diag(np.diag(matrix))) > 1e-8
                )
            is_coherent = np.asarray(is_coherent)
            axes[1].scatter(shown.real[~is_coherent], shown.imag[~is_coherent],
                            color=GREEN, marker="^", label=r"$\mu=0$")
            axes[1].scatter(shown.real[is_coherent], shown.imag[is_coherent],
                            color=ORANGE, label=r"$\mu\ne0$")
            axes[1].axhline(0, color="0.6", lw=1)
            axes[1].set(xlabel=r"$\operatorname{Re}\lambda$",
                        ylabel=r"$\operatorname{Im}\lambda$",
                        title="Davies modes")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## 4. Petz-Renyi dependence

            We repeat the crossing analysis for
            $D_\alpha(\rho\Vert\mathcal G[\rho])$. A coarser time grid keeps this
            diagnostic fast while preserving the trend.
            """
        ),
        code(
            r"""
            alphas = np.linspace(0.2, 1.8, 13)
            sample = slice(None, None, 5)
            sampled_times = times[sample]
            crossing_times = []
            for alpha in alphas:
                d1 = np.array([
                    nu.petz_renyi(state, nu.energy_dephase(state), alpha)
                    for state in states_1[sample]
                ])
                d2 = np.array([
                    nu.petz_renyi(state, nu.energy_dephase(state), alpha)
                    for state in states_2[sample]
                ])
                crossing_times.append(
                    nu.crossing_time(sampled_times, d2, d1)
                )

            fig, ax = plt.subplots()
            ax.plot(alphas, crossing_times, marker="o", ms=3, color=ORANGE)
            ax.axvline(1, color="0.4", ls=":", lw=1)
            ax.set(xlabel=r"Petz-Renyi parameter $\alpha$",
                   ylabel=r"crossing time $\tau_\alpha$",
                   title="Monotone-dependent symmetry crossing")
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            Only the nonzero Bohr-frequency blocks contribute to the asymmetry
            monotone. The rotated state is isospectral with the original one and
            initially more asymmetric, but its overlap with the slowest coherent
            eigenmode is strongly suppressed. Its late-time decay is therefore set by
            a faster coherent mode.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex4.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 4A: one Markovian U(1) circuit

            **Manuscript map.** Sec. III G 1, Figs. 8-9, Eqs. (68)-(77).

            This notebook is the conceptual, single-realization companion to
            `asymm_ex4.1.a.ipynb`. It constructs one U(1)-symmetric brickwork
            circuit, resets a two-qubit environment after every Floquet layer, and
            inspects the resulting Markovian channel charge by charge.

            The goal is to make the mechanism visible before ensemble averaging:
            block-size-$b$ initial states occupy only charges that are multiples of
            $b$, while high-$|\mu|$ channel blocks typically have smaller
            eigenvalues and decay faster.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. Initial states with selected mode support

            Eq. (75) applies disjoint $b$-body rotations,

            $$
            |\varphi(\theta,b)\rangle
            =\bigotimes_n
            \exp[-i(\theta/2)Y^{\otimes b}]\,|0\rangle^{\otimes N_s}.
            $$

            We use the three parameter pairs from Fig. 9:
            $(\theta/\pi,b)=(0.1,1),(0.2,2),(0.5,3)$.
            """
        ),
        code(
            r"""
            N_SYSTEM = 4
            N_ENVIRONMENT = 2
            CONFIGS = [(0.1 * np.pi, 1), (0.2 * np.pi, 2), (0.5 * np.pi, 3)]

            initial_states = []
            support = []
            for theta, block_size in CONFIGS:
                ket = nu.multi_spin_tilted_state(
                    N_SYSTEM, theta, block_size
                )
                rho = np.outer(ket, ket.conj())
                initial_states.append(rho)
                norms = nu.mode_trace_norms_u1(rho, N_SYSTEM)
                support.append([
                    charge for charge, value in norms.items() if value > 1e-10
                ])

            for (theta, block_size), charges in zip(CONFIGS, support):
                print(
                    f"b={block_size}, theta/pi={theta/np.pi:.1f}: "
                    f"occupied charges {charges}"
                )
            """
        ),
        md(
            r"""
            ## 2. Random U(1)-symmetric Floquet channel

            Each two-qubit gate is an XXZ interaction with local $z$ rotations, so it
            commutes with total $S_z$. A full even/odd brickwork layer acts on
            system plus environment. Tracing out and resetting the maximally mixed
            environment gives the same CPTP map $\mathcal E$ at every step:

            $$
            \rho_s(t+1)=\mathcal E[\rho_s(t)].
            $$
            """
        ),
        code(
            r"""
            rng = np.random.default_rng(100)
            unitary = nu.brickwork_unitary_u1(
                N_SYSTEM + N_ENVIRONMENT, rng
            )
            channel = nu.reduced_channel(
                unitary, N_SYSTEM, N_ENVIRONMENT
            )

            identity_vector = nu.vec(np.eye(2**N_SYSTEM))
            trace_error = la.norm(identity_vector.conj() @ channel
                                  - identity_vector.conj())

            test_state = initial_states[0]
            covariance_error = la.norm(
                nu.apply_channel(channel, nu.u1_twirl(test_state, N_SYSTEM))
                - nu.u1_twirl(
                    nu.apply_channel(channel, test_state), N_SYSTEM
                )
            )
            print(f"Trace-preservation error: {trace_error:.2e}")
            print(f"U(1)-covariance error:    {covariance_error:.2e}")
            assert trace_error < 1e-10
            assert covariance_error < 1e-10
            """
        ),
        md(
            r"""
            ## 3. Charge-resolved spectrum

            Covariance makes the channel block diagonal:

            $$
            \mathcal E=\bigoplus_{\mu=-N_s}^{N_s}\mathcal E^{(\mu)}.
            $$

            For each block we record its largest nonstationary eigenvalue in
            magnitude. The decay exponent per circuit step is
            $\lambda_\mu=\log|\eta_\mu|$.
            """
        ),
        code(
            r"""
            slowest_values, left_modes = nu.slowest_charge_data(
                channel, N_SYSTEM
            )
            charges = np.arange(-N_SYSTEM, N_SYSTEM + 1)
            rates = np.array([
                np.log(abs(slowest_values[int(charge)]))
                for charge in charges
            ])

            for charge, rate in zip(charges, rates):
                print(f"mu={charge:+d}: log|eta_mu|={rate:.4f}")
            """
        ),
        md(
            r"""
            ## 4. Symmetry restoration

            The relative entropy of U(1) asymmetry is
            $M[\rho]=S(\mathcal G[\rho])-S(\rho)$. The state with $b=3$ begins with
            the most asymmetry but avoids the slow low-charge sectors.
            """
        ),
        code(
            r"""
            STEPS = 50
            times = np.arange(STEPS + 1)
            curves = np.zeros((len(CONFIGS), STEPS + 1))

            for state_index, rho0 in enumerate(initial_states):
                rho = rho0.copy()
                for time in times:
                    curves[state_index, time] = nu.asymmetry_relative_entropy(
                        rho, nu.u1_twirl(rho, N_SYSTEM)
                    )
                    if time < STEPS:
                        rho = nu.apply_channel(channel, rho)

            tau_31 = nu.crossing_time(times, curves[2], curves[0])
            print(f"b=3 versus b=1 crossing: {tau_31:.3f} layers")
            assert np.isfinite(tau_31)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
            palette = [BLUE, ORANGE, "#F4C95D"]

            for curve, (theta, block_size), color in zip(
                curves, CONFIGS, palette
            ):
                axes[0].semilogy(
                    times, np.maximum(curve, 1e-15), color=color, lw=2,
                    label=fr"$\theta={theta/np.pi:.1f}\pi,\ b={block_size}$",
                )
            axes[0].set(xlabel="Floquet layer", ylabel=r"$M[\rho_s(t)]$",
                        title="One U(1)-symmetric channel")
            axes[0].legend(fontsize=9)

            positive = charges >= 0
            axes[1].plot(
                charges[positive], rates[positive],
                marker="o", color="0.35"
            )
            axes[1].set(xlabel=r"$|\mu|$",
                        ylabel=r"$\log|\eta_\mu|$",
                        title="Slowest decay in each charge block")
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            Block size is a state-engineering knob: it removes overlap with entire
            symmetry sectors before the dynamics begins. In this realization the
            high-charge blocks decay fastest, so the $b=3$ state sheds a larger
            initial resource more quickly. The ensemble notebook checks that this
            ordering is typical rather than a lucky draw.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex4.1.a.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 4B: ensemble reproduction of the Markovian U(1) circuit

            **Manuscript map.** Sec. III G 1 and Fig. 9.

            This is the production companion to `asymm_ex4.ipynb`. It averages the
            Markovian U(1) channel over random brickwork realizations and reconstructs
            the four diagnostics in Fig. 9:

            1. mean asymmetry trajectories;
            2. charge-resolved channel spectrum;
            3. slowest decay exponent versus $|\mu|$;
            4. initial overlap with the slowest left mode of each sector.

            The default verification run uses 40 realizations and 60 layers. Set
            `PAPER_SCALE=True` to use the manuscript values (100 realizations and
            180 layers). No unpublished parameter files are required.
            """
        ),
        code(COMMON_IMPORTS),
        code(
            r"""
            PAPER_SCALE = False
            N_REALIZATIONS = 100 if PAPER_SCALE else 40
            STEPS = 180 if PAPER_SCALE else 60
            N_SYSTEM = 4
            N_ENVIRONMENT = 2
            CONFIGS = [(0.1 * np.pi, 1), (0.2 * np.pi, 2), (0.5 * np.pi, 3)]

            initial_states = []
            for theta, block_size in CONFIGS:
                ket = nu.multi_spin_tilted_state(
                    N_SYSTEM, theta, block_size
                )
                initial_states.append(np.outer(ket, ket.conj()))

            curves = np.zeros(
                (N_REALIZATIONS, len(CONFIGS), STEPS + 1)
            )
            rates = np.zeros((N_REALIZATIONS, N_SYSTEM + 1))
            overlaps = np.zeros(
                (N_REALIZATIONS, len(CONFIGS), N_SYSTEM + 1)
            )
            """
        ),
        md(
            r"""
            ## 1. Ensemble calculation

            Every realization uses a new set of random U(1)-symmetric gates but a
            fixed map in time. The environment is maximally mixed and reset after
            every layer, exactly implementing Eq. (73).
            """
        ),
        code(
            r"""
            for realization in range(N_REALIZATIONS):
                rng = np.random.default_rng(1_000 + realization)
                unitary = nu.brickwork_unitary_u1(
                    N_SYSTEM + N_ENVIRONMENT, rng
                )
                channel = nu.reduced_channel(
                    unitary, N_SYSTEM, N_ENVIRONMENT
                )
                slow_values, slow_left_modes = nu.slowest_charge_data(
                    channel, N_SYSTEM
                )

                for charge in range(N_SYSTEM + 1):
                    rates[realization, charge] = np.log(
                        abs(slow_values[charge])
                    )

                for state_index, rho0 in enumerate(initial_states):
                    rho = rho0.copy()
                    for time in range(STEPS + 1):
                        curves[realization, state_index, time] = (
                            nu.asymmetry_relative_entropy(
                                rho, nu.u1_twirl(rho, N_SYSTEM)
                            )
                        )
                        if time < STEPS:
                            rho = nu.apply_channel(channel, rho)

                    state_vector = nu.vec(rho0)
                    for charge in range(N_SYSTEM + 1):
                        overlaps[
                            realization, state_index, charge
                        ] = abs(
                            np.vdot(
                                slow_left_modes[charge], state_vector
                            )
                        )

            mean_curves = curves.mean(axis=0)
            std_curves = curves.std(axis=0)
            mean_rates = rates.mean(axis=0)
            std_rates = rates.std(axis=0)
            mean_overlaps = overlaps.mean(axis=0)
            std_overlaps = overlaps.std(axis=0)
            print(f"Completed {N_REALIZATIONS} circuit realizations.")
            """
        ),
        md(
            r"""
            ## 2. Mpemba crossings and sector hierarchy

            The average state with $b=3$ starts most asymmetric and crosses both
            lower-block-size curves. The spectrum panel tests the statistical
            mechanism: $\operatorname{Re}\lambda_\mu=\log|\eta_\mu|$ becomes more
            negative as $|\mu|$ grows.
            """
        ),
        code(
            r"""
            times = np.arange(STEPS + 1)
            labels = [
                fr"$\theta={theta/np.pi:.1f}\pi,\ b={block_size}$"
                for theta, block_size in CONFIGS
            ]
            for high in (1, 2):
                for low in range(high):
                    tau = nu.crossing_time(
                        times, mean_curves[high], mean_curves[low]
                    )
                    print(f"{labels[high]} vs {labels[low]}: tau={tau:.3f}")

            assert np.all(np.diff(mean_rates) < 0)
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
            palette = [BLUE, ORANGE, "#F4C95D"]

            for index, (mean, spread, label, color) in enumerate(
                zip(mean_curves, std_curves, labels, palette)
            ):
                axes[0, 0].semilogy(
                    times, np.maximum(mean, 1e-15),
                    color=color, lw=2, label=label
                )
                lower = np.maximum(mean - spread, 1e-15)
                upper = np.maximum(mean + spread, 1e-15)
                axes[0, 0].fill_between(
                    times, lower, upper, color=color, alpha=0.16
                )
            axes[0, 0].set(
                xlabel="Floquet layer", ylabel=r"$M[\rho_s(t)]$",
                title="(a) Ensemble-averaged asymmetry"
            )
            axes[0, 0].legend(fontsize=9)

            # Representative spectrum, colored by |mu|.
            blocks = nu.channel_charge_blocks(channel, N_SYSTEM)
            for charge, (block, _) in blocks.items():
                values = la.eigvals(block)
                axes[0, 1].scatter(
                    values.real, values.imag, s=9,
                    color=plt.cm.Oranges(abs(charge) / N_SYSTEM),
                    alpha=0.7,
                )
            axes[0, 1].axhline(0, color="0.6", lw=1)
            axes[0, 1].set(
                xlabel=r"$\operatorname{Re}\eta$",
                ylabel=r"$\operatorname{Im}\eta$",
                title="(b) Spectrum of one channel"
            )

            charges = np.arange(N_SYSTEM + 1)
            axes[1, 0].errorbar(
                charges, mean_rates, yerr=std_rates,
                color="0.35", marker="o", capsize=3
            )
            axes[1, 0].set(
                xlabel=r"$|\mu|$", ylabel=r"$\langle\log|\eta_\mu|\rangle$",
                title="(c) High charges decay faster"
            )

            for state_index, (label, color) in enumerate(zip(labels, palette)):
                axes[1, 1].errorbar(
                    charges, mean_overlaps[state_index],
                    yerr=std_overlaps[state_index],
                    color=color, marker="o", capsize=2, label=label,
                )
            axes[1, 1].set(
                xlabel=r"$|\mu|$",
                ylabel=r"$|\operatorname{Tr}(\ell_\mu^\dagger\rho_0)|$",
                title="(d) Initial slow-mode overlap"
            )
            axes[1, 1].legend(fontsize=8)

            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Reproduction note

            Random-circuit averages fluctuate with the seed and with the gate
            parameter distribution. The robust statements to reproduce are the
            sector selection, the monotonic hierarchy of mean decay rates with
            $|\mu|$, and the ordering/crossing of the three averaged trajectories.
            The `PAPER_SCALE` switch changes only statistical resolution and time
            horizon, not the algorithm.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex5.ipynb",
    [
        md(
            r"""
            # Asymmetry, Example 4C: non-Abelian SU(2) circuits

            **Manuscript map.** Sec. III G 2, Fig. 11, Eq. (79).

            The conserved quantity is now total angular momentum, not one commuting
            charge. Two-qubit gates are partial swaps and commute with collective
            SU(2) rotations. The system starts in a superposition of a singlet
            product and the polarized state:

            $$
            |\varphi(\theta)\rangle
            =\cos(\theta/2)|\xi\rangle^{\otimes N_s/2}
             +\sin(\theta/2)|0\rangle^{\otimes N_s},
            \quad
            |\xi\rangle=(|01\rangle-|10\rangle)/\sqrt2 .
            $$

            The paper averages 100 circuits with $N_s=8,N_e=12$, a Hilbert space of
            dimension $2^{20}$. The default notebook uses $N_s=N_e=6$ and eight
            realizations so the complete derivation runs on a laptop. It verifies
            the exact SU(2) twirl, covariance, information backflow, and the
            finite-size trend. Set the constants in the configuration cell for a
            production/HPC run.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. Exact non-Abelian twirl

            The Hilbert space decomposes as

            $$
            \mathcal H=\bigoplus_j \mathcal V_j\otimes\mathcal M_j,
            $$

            where $\mathcal V_j$ carries spin $j$ and $\mathcal M_j$ is its
            multiplicity space. Haar twirling replaces each representation factor by
            the maximally mixed state while preserving the multiplicity state:

            $$
            \mathcal G_{\rm SU(2)}[\rho]
            =\bigoplus_j \frac{I_{\mathcal V_j}}{2j+1}
             \otimes\operatorname{Tr}_{\mathcal V_j}(\Pi_j\rho\Pi_j).
            $$

            `notebook_utils.su2_schur_basis` uses the manual block recursion
            and column ordering of the original notebooks via
            `tools/build_cg_basis.py`, so no Monte Carlo group average is needed.
            """
        ),
        code(
            r"""
            N_SYSTEM = 6
            N_ENVIRONMENT = 6
            N_TOTAL = N_SYSTEM + N_ENVIRONMENT
            N_REALIZATIONS = 8
            STEPS = 30
            THETAS = np.array([0.30, 0.40, 0.45, 0.50, 0.55]) * np.pi

            schur_basis, paths_by_spin = nu.su2_schur_basis(
                N_SYSTEM, convention="manuscript"
            )
            print("Multiplicity by doubled spin 2j:")
            print({spin: len(paths) for spin, paths in paths_by_spin.items()})
            assert np.allclose(
                schur_basis.conj().T @ schur_basis,
                np.eye(2**N_SYSTEM),
                atol=1e-10,
            )
            """
        ),
        md(
            r"""
            ## 2. SU(2)-symmetric non-Markovian circuit

            Every gate is

            $$
            u(J)=\cos(J/2)I-i\sin(J/2)\,\mathrm{SWAP},
            \qquad J\sim\mathcal U[-\pi/5,\pi/5],
            $$

            and therefore commutes with simultaneous rotations of its two qubits.
            The environment begins in a singlet product and is **not** reset. The
            reduced dynamics can consequently be non-Markovian, and the asymmetry
            need not decrease at every intermediate step.
            """
        ),
        code(
            r"""
            environment = nu.singlet_product(N_ENVIRONMENT)
            keep_system = list(range(N_ENVIRONMENT, N_TOTAL))
            curves = np.zeros(
                (N_REALIZATIONS, len(THETAS), STEPS + 1)
            )

            for realization in range(N_REALIZATIONS):
                rng = np.random.default_rng(2_000 + realization)
                couplings = rng.uniform(
                    -np.pi / 5,
                    np.pi / 5,
                    len(nu.brickwork_pairs(N_TOTAL)),
                )

                for theta_index, theta in enumerate(THETAS):
                    system = nu.su2_tilted_state(N_SYSTEM, theta)
                    state = np.kron(environment, system)

                    for time in range(STEPS + 1):
                        rho_s = nu.partial_trace_pure_state(
                            state, keep_system, N_TOTAL
                        )
                        twirled = nu.su2_twirl_exact(
                            rho_s, schur_basis, paths_by_spin
                        )
                        curves[realization, theta_index, time] = (
                            nu.asymmetry_relative_entropy(rho_s, twirled)
                        )
                        if time < STEPS:
                            state = nu.run_su2_brickwork_layer(
                                state, couplings, N_TOTAL
                            )

            mean_curves = curves.mean(axis=0)
            std_curves = curves.std(axis=0)
            print(f"Completed {N_REALIZATIONS} scaled circuit realizations.")
            """
        ),
        md(
            r"""
            ## 3. Scaled result and finite-size interpretation

            The laptop-scale curves display decay plus revivals from information
            backflow. For $N_s=N_e=6$, their ordering remains stable over this time
            window; the pronounced crossings in Fig. 11 emerge at the paper's
            $N_s=8,N_e=12$ scale. This absence is a useful finite-size result, not a
            failed assertion hidden by hand-picked data.
            """
        ),
        code(
            r"""
            times = np.arange(STEPS + 1)
            palette = plt.cm.Oranges(np.linspace(0.35, 0.95, len(THETAS)))

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for mean, spread, theta, color in zip(
                mean_curves, std_curves, THETAS, palette
            ):
                ax.plot(
                    times, mean, color=color, lw=2,
                    label=fr"$\theta={theta/np.pi:.2f}\pi$"
                )
                ax.fill_between(
                    times,
                    np.maximum(mean - spread, 0),
                    mean + spread,
                    color=color,
                    alpha=0.15,
                )
            ax.set(xlabel="Floquet layer",
                   ylabel=r"$M_{\rm SU(2)}[\rho_s(t)]$",
                   title="Scaled non-Markovian SU(2) circuit")
            ax.legend(ncol=2, fontsize=9)
            plt.show()

            increments = np.diff(mean_curves, axis=1)
            print("Largest mean one-step revival:", increments.max())
            """
        ),
        md(
            r"""
            ## 4. Paper-scale audit checklist

            These settings reproduce the printed Eq. (79) model, but they must
            not be advertised as a reproduction of Fig. 11 without the missing
            raw-data provenance:

            - set `N_SYSTEM=8`, `N_ENVIRONMENT=12`;
            - use 100 realizations;
            - keep the five $\theta$ values above;
            - evolve a $2^{20}$ pure state (never a dense $2^{20}\times2^{20}$
              density matrix);
            - compute the exact SU(2) twirl on the $2^8$ reduced system;
            - average the relative entropy of asymmetry only after computing it for
              each realization.

            The published vector curves fail the covariance bound implied by
            literal Eq. (79), while the archived helper code also contains mixed
            entropy-log conventions. The maintained `asymm_ex5.ipynb` performs
            that consistency audit and keeps vectorized figure data separate from
            raw simulation trajectories.
            """
        ),
        md(
            r"""
            ## Takeaway

            The SU(2) example needs non-Abelian representation theory, but the
            resource-theory logic is unchanged: symmetric gates define a covariant
            reduced map, the singlet environment is free, and different initial
            symmetry content relaxes at different rates. Non-Markovian backflow
            explains the revivals and removes any expectation of step-by-step
            monotonic decay.
            """
        ),
    ],
)


write_notebook(
    "asymm_ex6.ipynb",
    [
        md(
            r"""
            # Unified thermal and symmetry Mpemba effects in one Davies process

            **Manuscript map.** Sec. IV, Figs. 12-13, Eqs. (80)-(83).

            For a symmetry-invariant steady state $\pi$, relative entropy splits as

            $$
            S(\rho\Vert\pi)
            =
            \underbrace{S(\rho\Vert\mathcal G[\rho])}_{\text{symmetry breaking}}
            +
            \underbrace{S(\mathcal G[\rho]\Vert\pi)}_
                       {\text{symmetry-respecting athermality}}.
            $$

            This notebook verifies the identity along a Davies trajectory and
            reproduces the two qualitative panels of Fig. 13:

            - weak noise $\gamma=0.05$: classical and total thermal crossings;
            - strong noise $\gamma=0.25$: only a symmetry crossing.

            The same Hamiltonian, bath, optimization target, and free operation are
            used in both panels. Only the initial noise strength changes.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            r"""
            ## 1. Shared TFIM Davies generator

            We follow the normalization of the released code,
            $s^\alpha=\sigma^\alpha/2$, with $N_s=4$, $J=h=1$, bath inverse
            temperature $\beta=2$, and initial inverse temperature $\beta_i=1$.
            """
        ),
        code(
            r"""
            N_SITES = 4
            BETA_BATH = 2.0
            BETA_INITIAL = 1.0

            hamiltonian = nu.tfim_hamiltonian(
                N_SITES, 1.0, 1.0, spin_half_operators=True
            )
            L, energies, energy_basis, pi = nu.davies_liouvillian(
                hamiltonian, BETA_BATH
            )
            eigenvalues, left_vectors = la.eig(L, left=True, right=False)
            order = np.argsort(np.abs(eigenvalues.real))
            eigenvalues = eigenvalues[order]
            left_vectors = left_vectors[:, order]

            slow_eigenvalue = eigenvalues[1]
            slow_left_mode = nu.unvec(
                left_vectors[:, 1], 2**N_SITES
            )
            slow_left_mode /= la.norm(slow_left_mode)

            print("Slowest nonstationary eigenvalue:", slow_eigenvalue)
            assert np.allclose(L @ nu.vec(pi), 0, atol=1e-10)
            """
        ),
        md(
            r"""
            ## 2. Two noise strengths, one optimization rule

            The noisy Gibbs state exactly follows the released script: the random
            matrix is multiplied by $\gamma$ before forming $X^\dagger X$, so the
            positive perturbation scales as $\gamma^2$. For each $\gamma$, a
            fixed-seed local-unitary search minimizes overlap with the same slowest
            left mode.
            """
        ),
        code(
            r"""
            GAMMAS = [0.05, 0.25]
            initial_pairs = {}
            optimization_histories = {}

            for gamma in GAMMAS:
                rho_1 = nu.manuscript_noisy_gibbs_state(
                    energies, BETA_INITIAL, gamma, seed=0
                )
                rho_2, rotation, history = nu.minimize_mode_overlap(
                    rho_1,
                    slow_left_mode,
                    N_SITES,
                    samples=5_000,
                    seed=7 + int(100 * gamma),
                )
                initial_pairs[gamma] = (rho_1, rho_2)
                optimization_histories[gamma] = history
                assert np.allclose(la.eigvalsh(rho_1), la.eigvalsh(rho_2))
                print(
                    f"gamma={gamma:.2f}: overlap "
                    f"{history[0]:.3e} -> {history[-1]:.3e}"
                )
            """
        ),
        md(
            r"""
            ## 3. Decomposition along the trajectories

            Here $\mathcal G$ is energy dephasing. We evaluate all three terms
            independently and assert Eq. (82) at every sampled time, rather than
            defining the total as the sum by construction.
            """
        ),
        code(
            r"""
            times = np.linspace(0, 1, 501)
            results = {}

            def three_monotones(states):
                symmetry = []
                respecting = []
                total = []
                for state in states:
                    twirled = nu.energy_dephase(state)
                    symmetry.append(
                        nu.asymmetry_relative_entropy(state, twirled)
                    )
                    respecting.append(
                        nu.quantum_relative_entropy(twirled, pi)
                    )
                    total.append(
                        nu.quantum_relative_entropy(state, pi)
                    )
                return {
                    "symmetry": np.asarray(symmetry),
                    "respecting": np.asarray(respecting),
                    "total": np.asarray(total),
                }

            for gamma, (rho_1, rho_2) in initial_pairs.items():
                states_1 = nu.evolve_density(L, rho_1, times)
                states_2 = nu.evolve_density(L, rho_2, times)
                result_1 = three_monotones(states_1)
                result_2 = three_monotones(states_2)
                results[gamma] = (result_1, result_2)

                for result in (result_1, result_2):
                    error = np.max(np.abs(
                        result["total"]
                        - result["symmetry"]
                        - result["respecting"]
                    ))
                    assert error < 1e-9
                print(f"gamma={gamma:.2f}: decomposition verified.")
            """
        ),
        md(
            r"""
            ## 4. Which resource exhibits a Mpemba effect?

            Line style labels the two isospectral initial states; color labels the
            resource. A crossing is reported only in the direction where the
            initially larger curve later becomes smaller.
            """
        ),
        code(
            r"""
            for gamma, (first, second) in results.items():
                print(f"\ngamma={gamma:.2f}")
                for key in ("symmetry", "respecting", "total"):
                    if second[key][0] > first[key][0]:
                        tau = nu.crossing_time(times, second[key], first[key])
                        direction = "rho_2 starts larger"
                    else:
                        tau = nu.crossing_time(times, first[key], second[key])
                        direction = "rho_1 starts larger"
                    print(f"  {key:>10s}: {direction}, tau={tau}")
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
            colors = {
                "symmetry": ORANGE,
                "respecting": GREEN,
                "total": PURPLE,
            }
            labels = {
                "symmetry": r"$S(\rho\Vert\mathcal{G}[\rho])$",
                "respecting": r"$S(\mathcal{G}[\rho]\Vert\pi)$",
                "total": r"$S(\rho\Vert\pi)$",
            }

            for ax, gamma in zip(axes, GAMMAS):
                first, second = results[gamma]
                for key in ("symmetry", "respecting", "total"):
                    ax.semilogy(
                        times, np.maximum(first[key], 1e-15),
                        color=colors[key], lw=2,
                        label=labels[key] + r", $\rho_1$",
                    )
                    ax.semilogy(
                        times, np.maximum(second[key], 1e-15),
                        color=colors[key], lw=2, ls="--",
                        label=labels[key] + r", $\rho_2$",
                    )
                ax.set(xlabel=r"$t$", title=fr"$\gamma={gamma}$")

            axes[0].set_ylabel("relative-entropy monotone")
            axes[0].legend(fontsize=7.6, ncol=2)
            fig.suptitle("Different Mpemba effects in the same Davies process")
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Takeaway

            “The Mpemba effect” is not a property of a trajectory without a chosen
            resource. The same state pair and physical dynamics can show a crossing
            in one component of Eq. (82) and not another. Weak noise produces
            crossings in symmetry-respecting and total athermality; strong noise
            instead makes the original state the more asymmetric one and yields only
            a symmetry-restoration crossing.
            """
        ),
    ],
)


# The circuit walkthroughs have their own data-backed builder.  Run it last so
# the executed NPZ analyses supersede the early exploratory versions above.
from build_circuit_notebooks import build_all as build_circuit_notebooks

build_circuit_notebooks()
print("Wrote 10 publishable notebooks.")
