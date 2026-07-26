# Circuit datasets

These files are the raw inputs of `asymm_ex4.ipynb`,
`asymm_ex4.1.a.ipynb`, and `asymm_ex5.ipynb`. Regenerate them from the
repository root with:

```bash
python tools/generate_circuit_data.py
```

Every NPZ is pickle-free and contains the monotone value for every circuit
realization, initial state, and sampled time. Ensemble means and uncertainty
bands are computed in the notebooks, not stored as preprocessed curves.

## Files

- `u1_markovian.npz`: Fig. 9 protocol at the manuscript size
  (`Ns=4`, `Ne=2`, 100 circuits, 200 layers). It also stores all
  charge-resolved channel eigenvalues, slow rates, overlaps, and sampled gate
  parameters.
- `u1_nonmarkovian.npz`: locally reproducible Fig. 10 protocol
  (`Ns=4`, `Ne=8`, 24 circuits, 300 layers). This is explicitly a scaled
  verification run; the NPZ records the manuscript production target.
- `u1_nonmarkovian_reference.npz`: three circuits at the Fig. 10
  Hilbert-space size (`Ns=8`, `Ne=12`, 100 layers), using archived parameter
  rows 40, 43, and 46 from
  [`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits).
  The file embeds those rows and records upstream commit
  `5042f3600a5b14c93b515e0bd0dab0e8fa4d5509` plus the SHA-256 of
  `data/U1_rnd_parameters.npy`. It is a fidelity check at the exact system
  size, not a substitute for the 100-circuit production average.
- `su2_nonmarkovian.npz`: Fig. 11 protocol at the manuscript Hilbert-space
  size (`Ns=8`, `Ne=12`, 100 layers) for three circuits. Each value uses the
  exact SU(2) Schur-basis twirl. The manuscript ensemble contains 100 circuits.

To rebuild the exact-size U(1) reference data from the upstream table:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --reference-checkout /path/to/mpemba_circuits
```

The generator refuses a parameter file whose SHA-256 differs from the pinned
reference. Add `--paper-scale` only in an HPC environment; it requests all 100
archived circuits for the 1000-layer Fig. 10 time range.
