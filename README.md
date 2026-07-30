# 🧊 Resource-Theoretical Unification of Mpemba Effects: Classical and Quantum

[![Paper](https://img.shields.io/badge/paper-Physical%20Review%20X-6f42c1)](https://doi.org/10.1103/rbt4-psfd)
[![arXiv](https://img.shields.io/badge/arXiv-2507.16976-b31b1b)](https://arxiv.org/abs/2507.16976)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-2f6f9f.svg)](LICENSE)

Notebooks accompanying:
> Alessandro Summer, Mattia Moroder, Laetitia P. Bettmann, Xhek Turkeshi,
> Iman Marvian, and John Goold, **“Resource-Theoretical Unification of
> Mpemba Effects: Classical and Quantum,”** *Physical Review X* **16**,
> 011065 (2026).

**TL;DR.** We recast anomalous relaxation — the *Mpemba effect* — as resource dissipation in a quantum resource theory. Thermal and symmetry Mpemba effects become parallel instances of the same phenomenon: a state that is *more resourceful* initially can lose its resource faster than a less resourceful one, driven by its overlap with the slowest eigenmodes of the dissipative dynamics. This repository provides an executable companion to every worked example in the paper, from classical Markov chains to non-Abelian SU(2) random circuits. Each walkthrough identifies whether it is an exact reproduction, a raw-data-backed calculation, or an independent reconstruction necessitated by unavailable manuscript-era inputs.

<!-- Regenerate with: python tools/generate_readme_gif.py -->
![Animated schematic of the resource-theoretic Mpemba mechanism: two states evolve under the same free dynamics; state rho_1 starts with more of the chosen resource but has less overlap with its slowest relevant decay mode, so its resource monotone falls below that of state rho_2 at the Mpemba crossing](figures/figure1.gif)

*Under the same free dynamics, an initially more resourceful state can lose the chosen resource faster when it has less overlap with the slowest decay mode relevant to that resource, producing a crossing of resource monotones. This mode-overlap mechanism provides a common description of thermal, symmetry, coherence, and nonstationarity Mpemba effects in classical and quantum settings.*

## 📰 Media coverage and background

- **Science:** [“Hot things can freeze faster than cool ones. Now, this paradox has gone quantum”](https://www.science.org/content/article/hot-things-can-freeze-faster-cool-ones-now-paradox-has-gone-quantum) — a popular-science feature connecting the classical freezing paradox to recent quantum Mpemba research and this resource-theoretic framework.
- **Phys.org:** [“Framework unifies the classical and quantum Mpemba effects”](https://phys.org/news/2026-03-framework-classical-quantum-mpemba-effects.html) — an accessible overview of this work and its common slow-mode mechanism.
- **Wikipedia:** This work can also be found among the references on [John Goold (physicist)](https://en.wikipedia.org/wiki/John_Goold_(physicist)). For a general introduction to the effect, see [Mpemba effect](https://en.wikipedia.org/wiki/Mpemba_effect), and for historical and biographical context, see [Erasto B. Mpemba](https://en.wikipedia.org/wiki/Erasto_B._Mpemba).

## 🎯 Main contributions

- **🧩 Resource theory supplies a common definition of different Mpemba effects.**
  Thermal and symmetry Mpemba effects are both reversals of resource ordering under free dynamics, specialized to athermality and asymmetry. Crucially, the two canonical diagnostics used here share the [relative entropy of resource](https://doi.org/10.1103/RevModPhys.91.025001) construction, $R_{\mathrm{rel}}^{\mathcal F}(\rho)=\min_{\sigma\in\mathcal F}D(\rho\Vert\sigma)$, with only the free-state set changing. For athermality it gives $D(\rho\Vert\pi_\beta)=\beta[F(\rho)-F(\pi_\beta)]$, the [excess nonequilibrium free energy](https://doi.org/10.1103/PhysRevLett.133.140404) in units of $k_{\mathrm B}T$. For asymmetry it gives $D(\rho\Vert\mathcal G[\rho])=S(\mathcal G[\rho])-S(\rho)$, the relative entropy of asymmetry (called [entanglement asymmetry](https://doi.org/10.1038/s41467-023-37747-8) in the reduced-subsystem setting). These are distinct monotones, but instances of the same construction: a Mpemba crossing is precisely the initially more resourceful state becoming the less resourceful one.
  [Manuscript: Sec. I.1](https://arxiv.org/html/2507.16976#S1.SS1) ·
  [Sec. IV](https://arxiv.org/html/2507.16976#S4) ·
  [Walkthrough](notebooks/asymm_ex6.ipynb)

- **📉 A common spectral mechanism links thermal and symmetry Mpemba effects.**
  We show that anomalous relaxation is governed by the initial state’s overlap with the resourceful eigenmodes of the dissipative map — a mechanism previously recognized in thermal and nonstationary Mpemba effects by [Nava and Fabrizio](https://doi.org/10.1103/PhysRevB.100.125102) and [Carollo, Lasanta, and Lesanovsky](https://doi.org/10.1103/PhysRevLett.127.060401). In the symmetry setting, a strongly symmetry-broken state can relax faster when its overlap with the slowest symmetry-restoring mode is small or vanishes.
  [Manuscript: Sec. II](https://arxiv.org/html/2507.16976#S2) ·
  [Sec. III.3](https://arxiv.org/html/2507.16976#S3.SS3) ·
  [Thermal walkthrough](notebooks/atherm_ex1.ipynb) ·
  [Symmetry-mode walkthrough](notebooks/asymmetry_modes_example.ipynb)

- **⚖️ The symmetry Mpemba effect is not inherently quantum.**
  It already occurs in classical Markovian systems. Entanglement can accompany particular realizations, but it is not required for the effect.
  [Manuscript: Sec. III.4](https://arxiv.org/html/2507.16976#S3.SS4) ·
  [Walkthrough](notebooks/asymm_ex1.ipynb)

- **🌐 Open vs. closed is not the fundamental distinction; Markovian vs. non-Markovian is.**
  Dissipating a resource requires an environment. In "closed" Mpemba settings — entanglement-asymmetry circuits, ETH baths, any subsystem of a globally unitary composite — the joint universe conserves total asymmetry (or athermality) exactly; what decreases *locally* is the reduced-state monotone, because the resource flows into non-local system-environment correlations the partial trace discards. Whether the environment is an external bath or the complementary subsystem, the reduced dynamics is the same G-covariant CPTP map: its memory — Markovian vs. non-Markovian — sets the phenomenology, not the openness of the universe. Appendix F makes this precise: whenever the *universe* is *strongly* symmetric — a G-invariant joint unitary acting on a G-invariant environment — the reduced system dynamics is *weakly* symmetric (a G-covariant CPTP map).
  [Manuscript: Sec. III.1](https://arxiv.org/html/2507.16976#S3.SS1) ·
  [Sec. III.7](https://arxiv.org/html/2507.16976#S3.SS7) ·
  [Sec. V](https://arxiv.org/html/2507.16976#S5) ·
  [Appendix (G-invariance → G-covariance)](https://arxiv.org/html/2507.16976#A6) ·
  [Non-Markovian U(1) circuits](notebooks/asymm_ex4.1.a.ipynb) ·
  [ETH isolated bath](notebooks/Mpemba_ETH.ipynb)

- **🎲 Random circuits reveal the role of modes of asymmetry.**
  In random symmetry-preserving circuits — both Abelian U(1) (magnetization-conserving) and non-Abelian SU(2) — sectors carrying higher _modes of asymmetry_ decay faster on average. Restricting the initial state to those sectors suppresses its overlap with the slowest symmetry-restoring mode and produces a _spectrum-agnostic_ strong Mpemba effect: the mechanism operates statistically across random realizations without diagonalizing any specific channel.
  [Manuscript: Sec. III.7](https://arxiv.org/html/2507.16976#S3.SS7) ·
  [Markovian U(1)](notebooks/asymm_ex4.ipynb) ·
  [Non-Markovian U(1)](notebooks/asymm_ex4.1.a.ipynb) ·
  [Non-Abelian SU(2)](notebooks/asymm_ex5.ipynb)

## 📁 Repository contents

The thirteen notebooks in [`notebooks/`](notebooks/) are publication-oriented walkthroughs of the main-text and appendix examples. Each identifies its manuscript location and reproduction status, constructs the model, verifies the relevant structural properties, and only then tests the Mpemba crossing. The SU(2) notebook recomputes the five Fig. 11 curves from 100 raw full-SU(2) circuit realizations and reproduces all ten pairwise crossings. It verifies covariance under $S_x,S_y,S_z$ and uses the exact non-Abelian Haar twirl.


## 🚀 Quick start

Create a virtual environment, install the dependencies, and execute the
lightweight single-qubit introduction:

```bash
git clone https://github.com/alessum/resource_theory_mpemba.git
cd resource_theory_mpemba
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/execute_notebooks.py atherm_ex2.ipynb
```

The final command checks the notebook entirely in memory and does not modify
the tracked file. See the [notebook guide](NOTEBOOKS.md#running-the-notebooks)
for the full notebook map and computational scope.


### Appendix companions

- 🌀 [`Mpemba_nonstatioinarity.ipynb`](notebooks/Mpemba_nonstatioinarity.ipynb) —    independent reconstruction of the Appendix A nonstationarity example. Both initial states are generated transparently; no pickle or stored trajectory is used.
- ⛓️ [`Mpemba_ETH.ipynb`](notebooks/Mpemba_ETH.ipynb) — manuscript-scale Appendix B calculation for an $N=15$ ETH bath, including the full sparse $2^{16}$-dimensional unitary evolution and Rényi crossing-time inset.
- 📐 [`Mpemba_QFI_monotone.ipynb`](notebooks/Mpemba_QFI_monotone.ipynb) — exact Appendix QFI reconstruction showing crossings for SLD and Wigner-Yanase metrics but no crossing for the harmonic-mean metric.
- [`NOTEBOOKS.md`](NOTEBOOKS.md) gives the recommended reading order and maps every notebook to the relevant manuscript section and figure.
- [`tools/notebook_utils.py`](tools/notebook_utils.py) contains shared numerical routines.
- [`tools/circuit_data.py`](tools/circuit_data.py) implements the reproducible random-circuit simulations.
- [`data/circuit_examples/`](data/circuit_examples/) contains raw, per-realization circuit data rather than fitted or illustrative curves.
- [`tools/`](tools/) contains notebook builders, data generators, and the top-to-bottom execution check.

## ▶️ Running the notebooks

Run commands from the repository root. The notebook checker executes in memory
by default, so it does not alter tracked outputs:

```bash
python -m pip install -r requirements.txt
python tools/execute_notebooks.py
```

Pass notebook filenames to check only a subset. Add `--write` only when you intentionally want to refresh stored outputs:

```bash
python tools/execute_notebooks.py --write atherm_ex2.ipynb
```

To rebuild and refresh only the appendix companions:

```bash
python tools/build_publishable_notebooks.py \
  Mpemba_nonstatioinarity.ipynb Mpemba_ETH.ipynb Mpemba_QFI_monotone.ipynb
python tools/execute_notebooks.py --write \
  Mpemba_nonstatioinarity.ipynb Mpemba_ETH.ipynb Mpemba_QFI_monotone.ipynb
```

To rebuild the publishable notebooks, regenerate the standard local circuit datasets, and refresh every stored output:

```bash
python tools/build_publishable_notebooks.py
python tools/generate_circuit_data.py
python tools/execute_notebooks.py --write
```

Paper-scale circuit datasets have explicit commands, runtimes, provenance, and scope in [`data/circuit_examples/README.md`](data/circuit_examples/README.md). The exact-size non-Markovian U(1) reference dataset uses archived gate parameters from the companion repository:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --paper-scale \
  --reference-checkout /path/to/mpemba_circuits
```

The generator verifies the upstream parameter file by SHA-256 before running. The exact occupied-U(1)-sector engine evolves each charge sector once, reuses it for all five tilts, and parallelizes independent circuits. The stored file contains the complete 100-circuit, 1000-layer Fig. 10 calculation. 

## 📊 Circuit-data scope

The stored data are deliberately explicit about their statistical scope:

- The Markovian U(1) evolution (Fig. 9) is a paper-scale independent reproduction using the manuscript system ($N_s{=}4$), environment ($N_e{=}2$), and ensemble size ($100$ deterministic circuit realizations); it is not a replay of an unpublished original random ensemble.
- The non-Markovian U(1) walkthrough is reference-backed: it uses every archived parameter row from the pinned companion repository for the complete 100-circuit $N_s=8,N_e=12$, 1000-layer ensemble. The smaller 24-circuit $N_s=4,N_e=8$ file is retained only for the explicit channel audit of [Marvian--Spekkens](https://arxiv.org/abs/1312.0680) Eq. (3.10) as a mode-transfer reconstruction identity.
- The SU(2) walkthrough is a paper-scale, figure-matching independent reconstruction at $N_s=8,N_e=12$ with 100 realizations. It evolves isotropic partial-SWAP gates with an invariant singlet bath and computes asymmetry with the exact SU(2) Schur-basis Haar twirl. Its inferred initial-angle and displayed-time conventions are recorded explicitly because they are absent from the printed equations and public original data.
- The U(1) and SU(2) non-Markovian checks audit the same Marvian--Spekkens Eq. (3.10): as an explicit mode-transfer reconstruction identity for U(1), and as a contraction bound on the rank-1 irreducible tensor for SU(2).

## 📚 Citation

Use GitHub’s **Cite this repository** control, backed by [`CITATION.cff`](CITATION.cff), for machine-readable citation metadata. The published article remains the preferred scholarly citation:

```bibtex
@article{Summer2026Mpemba,
  title   = {Resource-Theoretical Unification of Mpemba Effects: Classical and Quantum},
  author  = {Summer, Alessandro and Moroder, Mattia and Bettmann, Laetitia P. and Turkeshi, Xhek and Marvian, Iman and Goold, John},
  journal = {Physical Review X},
  volume  = {16},
  pages   = {011065},
  year    = {2026},
  doi     = {10.1103/rbt4-psfd}
}
```

## ⚖️ License

The repository code is available under the [MIT License](LICENSE).
Third-party materials retain their original terms.
