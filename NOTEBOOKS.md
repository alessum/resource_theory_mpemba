# Reproducible notebook walkthroughs

These notebooks are the companion walkthroughs to *Resource-Theoretical
Unification of Mpemba Effects: Classical and Quantum*. They are designed to be read
in order: each one introduces the resource, constructs the dynamics, verifies the
relevant structural properties, and only then tests the Mpemba crossing. A
missing crossing or a convention mismatch is reported rather than replaced by
an illustrative curve.

| Order | Notebook | Manuscript location | Main result |
|---:|---|---|---|
| 1 | [`atherm_ex1.ipynb`](atherm_ex1.ipynb) | Sec. II B, Fig. 2 | Classical thermal Mpemba effect in an Ising chain |
| 2 | [`atherm_ex2.ipynb`](atherm_ex2.ipynb) | Sec. II C, Fig. 3 | Analytic single-qubit thermal example |
| 3 | [`asymmetry_modes_example.ipynb`](asymmetry_modes_example.ipynb) | Sec. III C, Fig. 4 | U(1) mode decomposition of clustered product states |
| 4 | [`asymm_ex1.ipynb`](asymm_ex1.ipynb) | Sec. III D, Fig. 5 | Exact classical \(Z_4\) symmetry example |
| 5 | [`asymm_ex2.ipynb`](asymm_ex2.ipynb) | Sec. III E, Fig. 6 | Quantum \(Z_4\) example and sector-resolved decay |
| 6 | [`asymm_ex3.ipynb`](asymm_ex3.ipynb) | Sec. III F, Fig. 7 | Time-translation asymmetry under Davies dynamics |
| 7 | [`asymm_ex4.ipynb`](asymm_ex4.ipynb) | Sec. III G 1, Fig. 9 | Data-backed Markovian U(1) ensemble and charge-resolved spectrum |
| 8 | [`asymm_ex4.1.a.ipynb`](asymm_ex4.1.a.ipynb) | Sec. III G 1, Fig. 10 | Exact-size reference-repository trajectories, local ensemble crossings, and information backflow |
| 9 | [`asymm_ex5.ipynb`](asymm_ex5.ipynb) | Sec. III G 2, Fig. 11 | Full-SU(2), 100-realization recomputation of the five curves and all ten crossings |
| 10 | [`asymm_ex6.ipynb`](asymm_ex6.ipynb) | Sec. IV, Figs. 12–13 | Decomposition into symmetry-breaking and symmetry-respecting resources |

## Running the notebooks

The notebooks use Python 3 with NumPy, SciPy, and Matplotlib. Run them from the
repository root so that they can import [`notebook_utils.py`](notebook_utils.py).
Every notebook has stored outputs from a clean top-to-bottom execution.

The circuit notebooks consume raw per-realization arrays in
`data/circuit_examples/`. They show individual trajectories before any ensemble
reduction and expose regeneration instructions. The Markovian U(1) file uses the
full Fig. 9 system and ensemble size. The non-Markovian U(1) walkthrough combines
three \(N_s=8,N_e=12\) runs using pinned gate parameters from
[`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits) with an
explicitly labelled \(N_s=4,N_e=8\) local ensemble. The SU(2) file uses the
manuscript Hilbert-space size \(N_s=8,N_e=12\) and all 100 realizations. Its
isotropic partial-SWAP gates commute with all three collective-spin
generators, its singlet bath is SU(2)-invariant, and its asymmetry is evaluated
with the exact Clebsch--Gordan Haar twirl. The notebook documents the
figure-matching angle and time-axis conventions and compares the resulting raw
ensemble mean with a separately labelled vector-figure reference. The
reference is never used as simulation input.

## Maintenance

Shared numerical routines live in [`notebook_utils.py`](notebook_utils.py), keeping
the notebooks focused on the physics. To regenerate and execute all notebooks:

```bash
python tools/build_publishable_notebooks.py
python tools/generate_circuit_data.py
python tools/execute_notebooks.py
```

The exact-size U(1) reference file has its own pinned reproduction command:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --reference-checkout /path/to/mpemba_circuits
```

The execution script uses `nbformat` and `nbclient` in addition to the runtime
dependencies above. Add `--paper-scale` to the U(1) reference command only on an
HPC system; it requests all 100 archived circuits for the 1000-layer Fig. 10
time range.
