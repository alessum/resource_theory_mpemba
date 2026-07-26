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
- `su2_nonmarkovian.npz`: Eq. (79)-consistent protocol at the manuscript
  Hilbert-space size (`Ns=8`, `Ne=12`, 100 layers) for three circuits. Each value uses the
  exact SU(2) Schur-basis twirl. The manuscript ensemble contains 100 circuits.
  This legacy file follows Eq. (79) in natural-log units and is now described
  as a reduced validation run, not as a reproduction of the published curve.
- `su2_fig11_vector_reference.csv`: 1,001 coordinates per curve extracted from
  the archived Illustrator vector figure. It is a figure-level consistency
  reference, not raw circuit output. Its metadata record the source hash and
  the displayed time grid; the conversion from that grid to Floquet layers is
  not documented in the manuscript.

The public [`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits)
history confirms that `gen_su2` constructs SU(2)-invariant partial-SWAP gates.
It does not contain a reproducible Fig. 11 run: the committed runner selects
U(1), its dormant SU(2) path and coupling law do not match the caption, its
initial state and environment do not match Eq. (79), and no SU(2) parameter
table or raw SU(2) result is stored there.

To rebuild the Fig. 11 vector reference from the archived single-panel PDF:

```bash
python tools/extract_su2_figure_reference.py \
  /path/to/circuit_su2.pdf \
  data/circuit_examples/su2_fig11_vector_reference.csv
```

To test explicitly labelled alternative explanations for the vector curves:

```bash
python tools/analyze_su2_curve_hypotheses.py \
  --candidate equation79 \
  --measure j2_eigenbasis_dephasing
```

Complete dephasing in a numerical \(J^2\) eigenbasis partly explains the
published left-edge scale, but it is basis dependent within degenerate spin
sectors and is not the SU(2) Haar twirl. Its simulated time dependence also
does not reproduce the five published curves.

To rebuild the exact-size U(1) reference data from the upstream table:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --reference-checkout /path/to/mpemba_circuits
```

The generator refuses a parameter file whose SHA-256 differs from the pinned
reference. Add `--paper-scale` only in an HPC environment; it requests all 100
archived circuits for the 1000-layer Fig. 10 time range.
