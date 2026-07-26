# Resource-Theoretical Unification of Mpemba Effects

Notebooks accompanying:

> Alessandro Summer, Mattia Moroder, Laetitia P. Bettmann, Xhek Turkeshi,
> Iman Marvian, and John Goold, **“Resource-Theoretical Unification of
> Mpemba Effects: Classical and Quantum,”** *Physical Review X* **16**,
> 011065 (2026).

[Published paper](https://journals.aps.org/prx/abstract/10.1103/rbt4-psfd)
· [DOI](https://doi.org/10.1103/rbt4-psfd)
· [arXiv:2507.16976](https://arxiv.org/abs/2507.16976)

![Resource-theoretic picture of the Mpemba effect](figures/figure_1_resource_theory_mpemba.png)

*Figure 1. A Mpemba effect occurs when the initially more resourceful state
loses its resource faster, causing the resource monotones to cross.*


## Main contributions

- **One resource-theoretic framework for different Mpemba effects.**
  Free states and free operations determine which resource is dissipated during
  relaxation. Nonequilibrium free energy is expressed through relative entropy
  to the thermal state, while the relative entropy of asymmetry measures
  distance from the symmetry-invariant states. The thermal and symmetry Mpemba
  effects are therefore parallel instances of anomalously fast resource
  depletion: symmetry restoration is the dissipation of asymmetry.
  [Manuscript: Sec. I.1](https://arxiv.org/html/2507.16976#S1.SS1) ·
  [Sec. IV](https://arxiv.org/html/2507.16976#S4) ·
  [Walkthrough](asymm_ex6.ipynb)

- **A common spectral mechanism to the thermal Mpemba effect.**
  We show that anomalous relaxation is governed by the initial state’s overlap with the eigenmodes of the dissipative map-a mechanism that has been recognised only in the thermal and nonstationary Mpemba effect by [Nava and Fabrizio](https://doi.org/10.1103/PhysRevB.100.125102) and [Carollo, Lasanta, and Lesanovsky](https://doi.org/10.1103/PhysRevLett.127.060401), a strongly symmetry-broken state can relax faster when its overlap with the   slowest symmetry-restoring mode is small or vanishes.
  [Manuscript: Sec. II](https://arxiv.org/html/2507.16976#S2) ·
  [Sec. III.3](https://arxiv.org/html/2507.16976#S3.SS3) ·
  [Thermal walkthrough](atherm_ex1.ipynb) ·
  [Symmetry-mode walkthrough](asymmetry_modes_example.ipynb)

- **The symmetry Mpemba effect is not inherently quantum.**
  It already occurs in classical Markovian systems. Entanglement can accompany particular realizations, but it is not required for the effect.
  [Manuscript: Sec. III.4](https://arxiv.org/html/2507.16976#S3.SS4) ·
  [Walkthrough](asymm_ex1.ipynb)

- **Random circuits reveal the role of modes of asymmetry.**
  In symmetry-preserving circuits, different charge modes decay at different rates. More strongly tilted initial states tend to populate Hilbert-space sectors statistically associated with faster-decaying eigenmodes, providing a mode-resolved explanation for their faster symmetry restoration.
  [Manuscript: Sec. III.7](https://arxiv.org/html/2507.16976#S3.SS7) ·
  [Markovian U(1)](asymm_ex4.ipynb) ·
  [Non-Markovian U(1)](asymm_ex4.1.a.ipynb) ·
  [Non-Abelian SU(2)](asymm_ex5.ipynb)

## Repository contents

The top-level notebooks are designed as readable walkthroughs of the examples in
the paper. Each notebook introduces the resource and dynamics, verifies the
relevant structural properties, and then tests the corresponding Mpemba
crossing. The SU(2) notebook additionally checks the published vector curves
against the covariance bound implied by printed Eq. (79), without treating
digitized figure coordinates as raw simulation data.

- [`NOTEBOOKS.md`](NOTEBOOKS.md) gives the recommended reading order and maps
  every notebook to the relevant manuscript section and figure.
- [`asymmetry_and_mpemba/`](asymmetry_and_mpemba/) contains the original manuscript-analysis material and figure-generation code.
- [`notebook_utils.py`](notebook_utils.py) contains shared numerical routines.
- [`circuit_data.py`](circuit_data.py) implements the reproducible random-circuit simulations.
- [`data/circuit_examples/`](data/circuit_examples/) contains raw, per-realization circuit data rather than fitted or illustrative curves.
- [`tools/`](tools/) contains notebook builders, data generators, and the top-to-bottom execution check.

## Running the notebooks

The notebooks require Python 3 with NumPy, SciPy, Matplotlib, `nbformat`, and `nbclient`. Run commands from the repository root.

To execute every notebook and refresh its stored outputs:

```bash
python tools/execute_notebooks.py
```

To rebuild the publishable notebooks, regenerate the standard circuit datasets, and execute everything:

```bash
python tools/build_publishable_notebooks.py
python tools/generate_circuit_data.py
python tools/execute_notebooks.py
```

The exact-size non-Markovian U(1) reference dataset uses archived gate parameters from the companion repository. Rebuild it with:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --reference-checkout /path/to/mpemba_circuits
```

The generator verifies the upstream parameter file by SHA-256 before running.
Adding `--paper-scale` requests the full 100-circuit, 1000-layer Fig. 10 calculation and should be used only on suitable HPC resources.

## Circuit-data scope

The stored data are deliberately explicit about their statistical scope:

- Fig. 9 uses the manuscript system and ensemble size.
- The non-Markovian U(1) walkthrough combines three   \(N_s=8,N_e=12\) reference-parameter runs with a longer 24-circuit \(N_s=4,N_e=8\) ensemble for local crossing and backflow statistics.
- The SU(2) walkthrough uses the manuscript Hilbert-space size \(N_s=8,N_e=12\)
  with three complete Eq. (79) validation runs and an exact SU(2) twirl. A
  separately labelled vector-figure reference exposes an unresolved
  inconsistency between Eq. (79) and Fig. 11; no coefficient swap is presented
  as a recovered raw-data convention. The notebook also distinguishes the
  genuinely SU(2)-invariant partial-SWAP gates from the mismatched archived
  execution path and tests, but rejects, a basis-dependent \(J^2\)-dephasing
  explanation for the curves.

Reduced local ensembles are never presented as replacements for the 100-realization manuscript averages.

## Citation

```bibtex
@article{Summer2026Mpemba,
  title   = {Resource-Theoretical Unification of Mpemba Effects:
             Classical and Quantum},
  author  = {Summer, Alessandro and Moroder, Mattia and Bettmann,
             Laetitia P. and Turkeshi, Xhek and Marvian, Iman and
             Goold, John},
  journal = {Physical Review X},
  volume  = {16},
  pages   = {011065},
  year    = {2026},
  doi     = {10.1103/rbt4-psfd}
}
```
