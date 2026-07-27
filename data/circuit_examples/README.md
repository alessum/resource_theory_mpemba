# Circuit datasets

These files are the raw inputs of `asymm_ex4.ipynb`,
`asymm_ex4.1.a.ipynb`, and `asymm_ex5.ipynb`, which reproduce Figs. 9, 10,
and 11 of *Resource-Theoretical Unification of Mpemba Effects: Classical and
Quantum* respectively. Regenerate them from the repository root with:

```bash
python tools/generate_circuit_data.py
```

Every NPZ is pickle-free and contains the monotone value for every circuit
realization, initial state, and sampled time. Ensemble means and uncertainty
bands are computed in the notebooks, not stored as preprocessed curves.

File ↔ figure map:

| Figure | File | Notebook |
|---|---|---|
| Fig. 9 | `u1_markovian.npz` | `asymm_ex4.ipynb` |
| Fig. 10 | `u1_nonmarkovian.npz`, `u1_nonmarkovian_reference.npz` | `asymm_ex4.1.a.ipynb` |
| Fig. 11 | `su2_fig11_full_su2.npz` | `asymm_ex5.ipynb` |
| Fig. 11 (methodology check) | `su2_nonmarkovian.npz` | `asymm_ex5.ipynb` |

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
- `su2_nonmarkovian.npz`: three-circuit, Eq. (79)-consistent methodology check
  at the manuscript Hilbert-space size (`Ns=8`, `Ne=12`, 100 layers). Each
  value uses the exact SU(2) Schur-basis twirl built with
  `tools/build_cg_basis.py`, following the manual block recursion and column
  ordering of the original notebooks. It is a low-cost sanity check for the
  Fig. 11 pipeline in natural-log units; the manuscript reproduction of the
  five Fig. 11 curves and all ten crossings is the 100-realization run stored
  in `su2_fig11_full_su2.npz` below.
- `su2_fig11_full_su2.npz`: the Fig. 11 reproduction at the manuscript
  Hilbert-space and ensemble size (`Ns=8`, `Ne=12`, 100 circuits, 501
  Floquet layers). It stores raw per-realization trajectories and all
  isotropic couplings. Evolution is an exact fixed-magnetization-sector
  decomposition of the full partial-SWAP circuit, while every reported
  asymmetry uses the complete SU(2) Schur-basis Haar twirl. Regenerate it with:

  ```bash
  python tools/generate_circuit_data.py --only su2-fig11 --paper-scale
  ```

  The file records the figure-matching shifted-angle convention and the
  displayed-time calibration explicitly. Neither convention is taken from
  the companion repository's U(1) runner.
The public [`alessum/mpemba_circuits`](https://github.com/alessum/mpemba_circuits)
history confirms that `gen_su2` constructs SU(2)-invariant partial-SWAP gates,
but it does not contain a reproducible Fig. 11 run: the committed runner
selects U(1), its dormant SU(2) path and coupling law do not match the
caption, its initial state and environment do not match Eq. (79), and no
SU(2) parameter table or raw SU(2) result is stored there. This repository
closes that gap: `su2_fig11_full_su2.npz` together with `asymm_ex5.ipynb`
regenerate the five Fig. 11 curves and all ten crossings at the manuscript
Hilbert-space and ensemble size, entirely from the isotropic-coupling
protocol and the exact SU(2) Schur-basis twirl.

Verify the manuscript-era CG basis without writing the basis matrix:

```bash
python tools/build_cg_basis.py --sizes 8 --no-save
```

The builder requires NumPy and SymPy. Only the eight-system-qubit basis is
needed for Fig. 11; the twelve environment qubits remain in a known singlet
product and are not twirled numerically.

To rebuild the exact-size U(1) reference data from the upstream table:

```bash
python tools/generate_circuit_data.py \
  --only u1-reference \
  --reference-checkout /path/to/mpemba_circuits
```

The generator refuses a parameter file whose SHA-256 differs from the pinned
reference. Add `--paper-scale` only in an HPC environment; it requests all 100
archived circuits for the 1000-layer Fig. 10 time range.
