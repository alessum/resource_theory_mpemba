#!/usr/bin/env python3
"""
Build the SU(2) Clebsch-Gordan basis for N spin-1/2 particles.

For each N in {2, 4, 6, 8, 12} this script constructs the unitary matrix
``U_CG_N`` of shape (2**N, 2**N) whose columns are simultaneous eigenvectors
of J² and J_z, grouped by total spin j (smallest j first) and by the
multiplicity of j-copies:

    U_CG_N.conj().T @ J²  @ U_CG_N   is block-diagonal in j
    U_CG_N.conj().T @ J_z @ U_CG_N   is diagonal within each j-block

Within a j-block, the layout is (multiplicity, 2j+1) with the fastest
running index being M in ascending order. This is the ordering used by
``manual_NX_SU2_tw`` in the original notebooks.

Recursion structure (matching the original notebooks):

    N=2  : hard-coded (triplet + singlet)
    N=4  : N=2 ⊗ N=2
    N=6  : N=4 ⊗ N=2
    N=8  : N=4 ⊗ N=4
    N=12 : N=6 ⊗ N=6

This module was supplied with the manuscript implementation and is kept as
the provenance-preserving CG constructor for the SU(2) circuit audit.
"""

from __future__ import annotations

import argparse
from functools import reduce
from typing import Dict, Tuple

import numpy as np
from sympy import S, sympify
from sympy.physics.quantum.cg import CG


_ID = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _op_on_site(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    operators = [_ID] * n_qubits
    operators[site] = op
    return reduce(np.kron, operators)


def build_Jx(n_qubits: int) -> np.ndarray:
    return 0.5 * sum(
        _op_on_site(_SX, site, n_qubits) for site in range(n_qubits)
    )


def build_Jy(n_qubits: int) -> np.ndarray:
    return 0.5 * sum(
        _op_on_site(_SY, site, n_qubits) for site in range(n_qubits)
    )


def build_Jz(n_qubits: int) -> np.ndarray:
    return 0.5 * sum(
        _op_on_site(_SZ, site, n_qubits) for site in range(n_qubits)
    )


def build_J2(n_qubits: int) -> np.ndarray:
    jx, jy, jz = (
        build_Jx(n_qubits),
        build_Jy(n_qubits),
        build_Jz(n_qubits),
    )
    return jx @ jx + jy @ jy + jz @ jz


def _ket(label: str, n_qubits: int) -> np.ndarray:
    """Computational ket with the leftmost bit denoting site zero."""
    vector = np.zeros(2**n_qubits, dtype=np.complex128)
    vector[int(label, 2)] = 1.0
    return vector


def _lc(coefficients: Dict[str, complex], n_qubits: int) -> np.ndarray:
    """Create a computational-basis linear combination."""
    vector = np.zeros(2**n_qubits, dtype=np.complex128)
    for label, coefficient in coefficients.items():
        vector[int(label, 2)] += coefficient
    return vector


_CG_CACHE: Dict[Tuple[int, int], Dict[tuple, float]] = {}


def _cg_table(j_left: int, j_right: int) -> Dict[tuple, float]:
    """Return all nonzero CG coefficients for two integer-spin irreps."""
    if (j_left, j_right) in _CG_CACHE:
        return _CG_CACHE[(j_left, j_right)]

    table: Dict[tuple, float] = {}
    for total in range(
        abs(j_left - j_right), j_left + j_right + 1
    ):
        for magnetic in range(-total, total + 1):
            for magnetic_left in range(-j_left, j_left + 1):
                magnetic_right = magnetic - magnetic_left
                if -j_right <= magnetic_right <= j_right:
                    value = CG(
                        S(j_left),
                        S(magnetic_left),
                        S(j_right),
                        S(magnetic_right),
                        S(total),
                        S(magnetic),
                    ).doit()
                    if value != 0:
                        table[
                            (
                                j_left,
                                magnetic_left,
                                j_right,
                                magnetic_right,
                                total,
                                magnetic,
                            )
                        ] = float(sympify(value))
    _CG_CACHE[(j_left, j_right)] = table
    return table


def _enum_couplings(total: int, jmax_left: int, jmax_right: int):
    """Yield factor-spin pairs in the original notebook order."""
    for shell in range(jmax_left + jmax_right, -1, -1):
        j_left_high = min(shell, jmax_left)
        j_left_low = max(
            (shell + 1) // 2, shell - jmax_right
        )
        for j_left in range(j_left_high, j_left_low - 1, -1):
            j_right = shell - j_left
            if j_right < 0 or j_right > jmax_right:
                continue
            if (
                total < abs(j_left - j_right)
                or total > j_left + j_right
            ):
                continue
            yield (j_left, j_right)
            if (
                j_left != j_right
                and j_right <= jmax_left
                and j_left <= jmax_right
            ):
                yield (j_right, j_left)


def couple_bases(
    states_left: Dict[str, np.ndarray],
    multiplicities_left: Dict[int, int],
    n_left: int,
    states_right: Dict[str, np.ndarray],
    multiplicities_right: Dict[int, int],
    n_right: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[int, int]]:
    """Couple two CG bases using the original notebook enumeration."""
    n_qubits = n_left + n_right
    dimension = 2**n_qubits
    jmax_left = max(multiplicities_left)
    jmax_right = max(multiplicities_right)
    jmax_total = jmax_left + jmax_right

    per_total: Dict[int, list[np.ndarray]] = {
        total: [] for total in range(jmax_total + 1)
    }
    total_states: Dict[str, np.ndarray] = {}
    multiplicities_total: Dict[int, int] = {}

    for total in range(jmax_total, -1, -1):
        copy_total = 0
        for j_left, j_right in _enum_couplings(
            total, jmax_left, jmax_right
        ):
            coefficients = _cg_table(j_left, j_right)
            magnetic_left_values = list(
                range(-j_left, j_left + 1)
            )
            for copy_left in range(multiplicities_left[j_left]):
                for copy_right in range(
                    multiplicities_right[j_right]
                ):
                    for magnetic in range(
                        total, -total - 1, -1
                    ):
                        state = np.zeros(
                            dimension, dtype=np.complex128
                        )
                        for magnetic_left in magnetic_left_values:
                            magnetic_right = magnetic - magnetic_left
                            if abs(magnetic_right) > j_right:
                                continue
                            coefficient = coefficients.get(
                                (
                                    j_left,
                                    magnetic_left,
                                    j_right,
                                    magnetic_right,
                                    total,
                                    magnetic,
                                ),
                                0.0,
                            )
                            if coefficient == 0.0:
                                continue
                            state += coefficient * np.kron(
                                states_left[
                                    f"{j_left},{magnetic_left},{copy_left}"
                                ],
                                states_right[
                                    f"{j_right},{magnetic_right},{copy_right}"
                                ],
                            )
                        norm = np.linalg.norm(state)
                        if norm < 1e-12:
                            raise RuntimeError(
                                "Zero-norm CG state at "
                                f"J={total}, M={magnetic}, "
                                f"j1={j_left}, j2={j_right}, "
                                f"r1={copy_left}, r2={copy_right}."
                            )
                        state /= norm
                        total_states[
                            f"{total},{magnetic},{copy_total}"
                        ] = state
                        per_total[total].append(state)
                    copy_total += 1
        multiplicities_total[total] = copy_total

    columns: list[np.ndarray] = []
    for total in range(jmax_total + 1):
        columns.extend(per_total[total][::-1])

    basis = (
        np.column_stack(columns)
        if columns
        else np.eye(dimension, dtype=np.complex128)
    )
    return basis, total_states, multiplicities_total


def build_N2() -> Tuple[
    np.ndarray, Dict[str, np.ndarray], Dict[int, int]
]:
    """Return the hard-coded two-spin basis of the original notebooks."""
    inverse_sqrt_two = 1.0 / np.sqrt(2)
    states: Dict[str, np.ndarray] = {
        "1,1,0": _ket("11", 2),
        "1,0,0": _lc(
            {"01": inverse_sqrt_two, "10": inverse_sqrt_two}, 2
        ),
        "1,-1,0": _ket("00", 2),
        "0,0,0": _lc(
            {"01": inverse_sqrt_two, "10": -inverse_sqrt_two}, 2
        ),
    }
    columns = [
        states["0,0,0"],
        states["1,-1,0"],
        states["1,0,0"],
        states["1,1,0"],
    ]
    basis = np.column_stack(columns)
    return basis, states, {0: 1, 1: 1}


def verify_basis(
    basis: np.ndarray,
    multiplicities: Dict[int, int],
    n_qubits: int,
    tolerance: float = 1e-8,
) -> None:
    """Verify unitarity and the expected J²/Jz block structure."""
    dimension = 2**n_qubits
    if basis.shape != (dimension, dimension):
        raise ValueError(
            f"CG basis has shape {basis.shape}; "
            f"expected ({dimension}, {dimension})."
        )

    gram_error = np.linalg.norm(
        basis.conj().T @ basis - np.eye(dimension)
    )
    if gram_error > tolerance:
        raise AssertionError(
            f"U_CG_{n_qubits} is not unitary "
            f"(error={gram_error:.2e})."
        )

    j_squared = build_J2(n_qubits)
    j_z = build_Jz(n_qubits)
    transformed_j_squared = basis.conj().T @ j_squared @ basis
    transformed_j_z = basis.conj().T @ j_z @ basis

    off_diagonal = np.linalg.norm(
        transformed_j_squared
        - np.diag(np.diag(transformed_j_squared))
    )
    if off_diagonal > tolerance:
        raise AssertionError(
            f"J² is not diagonal in U_CG_{n_qubits} "
            f"(off-diagonal error={off_diagonal:.2e})."
        )

    offset = 0
    for total in range(max(multiplicities) + 1):
        multiplicity = multiplicities.get(total, 0)
        block_size = multiplicity * (2 * total + 1)
        if block_size == 0:
            continue
        eigenvalues = np.real(
            np.diag(transformed_j_squared)[
                offset : offset + block_size
            ]
        )
        if (
            np.max(
                np.abs(eigenvalues - total * (total + 1))
            )
            > tolerance
        ):
            raise AssertionError(
                f"Incorrect J² eigenvalues in J={total} block."
            )

        j_z_block = transformed_j_z[
            offset : offset + block_size,
            offset : offset + block_size,
        ]
        j_z_off_diagonal = np.linalg.norm(
            j_z_block - np.diag(np.diag(j_z_block))
        )
        if j_z_off_diagonal > tolerance:
            raise AssertionError(
                f"Jz is not diagonal in J={total} block "
                f"(error={j_z_off_diagonal:.2e})."
            )
        reshaped = np.real(np.diag(j_z_block)).reshape(
            multiplicity, 2 * total + 1
        )
        expected = np.arange(total, -total - 1, -1)
        if (
            np.max(np.abs(reshaped - expected[None, :]))
            > tolerance
        ):
            raise AssertionError(
                f"Incorrect Jz pattern in J={total} block."
            )
        offset += block_size

    if offset != dimension:
        raise AssertionError(
            f"CG block sizes sum to {offset}, expected {dimension}."
        )


_RECIPE: Dict[int, Tuple[int, int]] = {
    2: (0, 0),
    4: (2, 2),
    6: (4, 2),
    8: (4, 4),
    12: (6, 6),
}


def build_all(
    sizes=(2, 4, 6, 8, 12),
    verify: bool = True,
    verbose: bool = True,
):
    """Build requested CG bases while respecting their dependencies."""
    needed = set(sizes)
    stack = list(needed)
    while stack:
        n_qubits = stack.pop()
        if n_qubits not in _RECIPE:
            raise ValueError(
                f"No manuscript CG recipe is defined for N={n_qubits}."
            )
        left, right = _RECIPE[n_qubits]
        for factor in (left, right):
            if factor > 0 and factor not in needed:
                needed.add(factor)
                stack.append(factor)

    results: Dict[
        int, Tuple[np.ndarray, Dict[str, np.ndarray], Dict[int, int]]
    ] = {}
    for n_qubits in sorted(needed):
        if verbose:
            print(f"[build] N = {n_qubits:>2} ...", end=" ", flush=True)
        if n_qubits == 2:
            basis, states, multiplicities = build_N2()
        else:
            left, right = _RECIPE[n_qubits]
            _, states_left, multiplicities_left = results[left]
            _, states_right, multiplicities_right = results[right]
            basis, states, multiplicities = couple_bases(
                states_left,
                multiplicities_left,
                left,
                states_right,
                multiplicities_right,
                right,
            )
        results[n_qubits] = (basis, states, multiplicities)
        if verbose:
            summary = ", ".join(
                f"j={total}:{multiplicities[total]}"
                for total in sorted(multiplicities)
                if multiplicities[total] > 0
            )
            print(f"dim={basis.shape[0]:<5} ({summary})")

        if verify:
            if verbose:
                print("        verify J², Jz ...", end=" ", flush=True)
            verify_basis(basis, multiplicities, n_qubits)
            if verbose:
                print("ok")

    return {n_qubits: results[n_qubits] for n_qubits in sizes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 12],
        choices=sorted(_RECIPE),
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--outdir", default=".")
    arguments = parser.parse_args()

    results = build_all(
        sizes=tuple(arguments.sizes),
        verify=not arguments.no_verify,
        verbose=True,
    )
    if not arguments.no_save:
        from pathlib import Path

        output_directory = Path(arguments.outdir)
        output_directory.mkdir(parents=True, exist_ok=True)
        for n_qubits, (basis, _, _) in results.items():
            path = output_directory / f"U_CG_{n_qubits}.npy"
            np.save(path, basis)
            print(f"[save] {path} ({basis.shape[0]}x{basis.shape[1]})")


if __name__ == "__main__":
    main()
