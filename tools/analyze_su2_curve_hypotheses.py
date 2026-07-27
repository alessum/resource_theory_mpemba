"""Test historically plausible protocols against the vectorized Fig. 11 curves.

This is an analysis tool, not a claim that the manuscript text defines any
candidate implemented here.  The ``historical_product`` candidate is recovered
from the public ``alessum/mpemba_circuits`` history:

* 20 spins on a periodic brickwork ring;
* the last 8 tensor factors are the observed system;
* the system starts in ``Ry(theta)|0>`` on every spin;
* the other 12 spins start in ``|0>``;
* one fixed set of SU(2)-invariant partial-SWAP gates is repeated.

The polarized environment is not SU(2)-invariant.  Consequently, the reduced
system dynamics is not an SU(2)-covariant channel even though every gate
commutes with the global SU(2) action.

The staggered candidates come from Eq. (S7) of Liu et al., Phys. Rev. Lett.
133, 140405 (2024).  That paper diagnoses block coherence between joint
``(J^2, J_z)`` sectors.  Its diagnostic is distinct from the full SU(2) Haar
twirl, as the authors explicitly note, so both observables are available.

The ``eq79_swapped_singlet_environment`` candidate tests a narrow provenance
hypothesis suggested by the published curves: reverse the two coefficients in
Eq. (79), retain the invariant singlet environment, and optionally report
entropy in bits.  It is deliberately named as a hypothesis rather than as the
manuscript protocol.

The ``equation79`` candidate follows the printed state and singlet
environment. Besides the correct Haar twirl, it can be paired with complete
dephasing in a numerical eigenbasis of ``J^2``. That operation destroys
coherence within degenerate spin sectors, is basis dependent, and is *not* an
SU(2) twirl. It is included only as a falsifiable historical-bug hypothesis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import circuit_data as cd
import functions as fn
import notebook_utils as nu


THETA_OVER_PI = np.array([0.30, 0.35, 0.40, 0.45, 0.50])


def _spin_coherent_product(n_qubits: int, theta: float) -> np.ndarray:
    local = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    return nu.kron_all([local] * n_qubits)


def _staggered_tilt_product(n_qubits: int, theta: float) -> np.ndarray:
    locals_ = [
        np.array(
            [
                np.cos(theta / 2),
                ((-1) ** site) * np.sin(theta / 2),
            ],
            dtype=complex,
        )
        for site in range(n_qubits)
    ]
    return nu.kron_all(locals_)


def _pair_product_state(
    n_qubits: int, theta: float, *, swapped: bool
) -> np.ndarray:
    """Product over pairs of a singlet/polarized superposition."""
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    polarized = np.array([1, 0, 0, 0], dtype=complex)
    if swapped:
        pair = np.sin(theta / 2) * singlet + np.cos(theta / 2) * polarized
    else:
        pair = np.cos(theta / 2) * singlet + np.sin(theta / 2) * polarized
    return nu.kron_all([pair] * (n_qubits // 2))


def _initial_states(candidate: str) -> np.ndarray:
    n_system = 8
    n_environment = 12
    if candidate == "equation79":
        environment = nu.singlet_product(n_environment)
        return np.column_stack(
            [
                np.kron(
                    environment,
                    nu.su2_tilted_state(
                        n_system, theta_fraction * np.pi
                    ),
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "equation79_double_angle":
        environment = nu.singlet_product(n_environment)
        system_singlet = nu.singlet_product(n_system)
        system_polarized = np.zeros(2**n_system, dtype=complex)
        system_polarized[0] = 1
        return np.column_stack(
            [
                np.kron(
                    environment,
                    np.cos(theta_fraction * np.pi) * system_singlet
                    + np.sin(theta_fraction * np.pi) * system_polarized,
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "equation79_shifted_half_angle":
        environment = nu.singlet_product(n_environment)
        system_singlet = nu.singlet_product(n_system)
        system_polarized = np.zeros(2**n_system, dtype=complex)
        system_polarized[0] = 1
        return np.column_stack(
            [
                np.kron(
                    environment,
                    np.cos(np.pi * (0.25 + theta_fraction / 2))
                    * system_singlet
                    + np.sin(np.pi * (0.25 + theta_fraction / 2))
                    * system_polarized,
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate in {
        "equation79_polarized_environment",
        "equation79_swapped_polarized_environment",
    }:
        environment = np.zeros(2**n_environment, dtype=complex)
        environment[0] = 1
        system_singlet = nu.singlet_product(n_system)
        system_polarized = np.zeros(2**n_system, dtype=complex)
        system_polarized[0] = 1
        swapped = candidate == "equation79_swapped_polarized_environment"
        return np.column_stack(
            [
                np.kron(
                    environment,
                    (
                        np.sin(theta_fraction * np.pi / 2)
                        * system_singlet
                        + np.cos(theta_fraction * np.pi / 2)
                        * system_polarized
                    )
                    if swapped
                    else (
                        np.cos(theta_fraction * np.pi / 2)
                        * system_singlet
                        + np.sin(theta_fraction * np.pi / 2)
                        * system_polarized
                    ),
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "historical_product":
        environment = np.zeros(2**n_environment, dtype=complex)
        environment[0] = 1
        return np.column_stack(
            [
                np.kron(
                    environment,
                    _spin_coherent_product(
                        n_system, theta_fraction * np.pi
                    ),
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "eq79_swapped_singlet_environment":
        environment = nu.singlet_product(n_environment)
        system_singlet = nu.singlet_product(n_system)
        system_polarized = np.zeros(2**n_system, dtype=complex)
        system_polarized[0] = 1
        return np.column_stack(
            [
                np.kron(
                    environment,
                    np.sin(theta_fraction * np.pi / 2) * system_singlet
                    + np.cos(theta_fraction * np.pi / 2)
                    * system_polarized,
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate in {
        "pair_product_singlet_environment",
        "pair_product_swapped_singlet_environment",
    }:
        environment = nu.singlet_product(n_environment)
        swapped = candidate == "pair_product_swapped_singlet_environment"
        return np.column_stack(
            [
                np.kron(
                    environment,
                    _pair_product_state(
                        n_system,
                        theta_fraction * np.pi,
                        swapped=swapped,
                    ),
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "staggered_system_singlet_environment":
        environment = nu.singlet_product(n_environment)
        return np.column_stack(
            [
                np.kron(
                    environment,
                    _staggered_tilt_product(
                        n_system, theta_fraction * np.pi
                    ),
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    if candidate == "staggered_global":
        return np.column_stack(
            [
                _staggered_tilt_product(
                    n_system + n_environment,
                    theta_fraction * np.pi,
                )
                for theta_fraction in THETA_OVER_PI
            ]
        )
    raise ValueError(f"unknown candidate: {candidate}")


def _jm_dephasing_asymmetry_from_global_pure(
    state: np.ndarray,
    n_system: int,
    n_environment: int,
    schur_basis: np.ndarray,
    paths_by_spin: dict[int, list[tuple[int, ...]]],
) -> float:
    """Entropy increase under dephasing between joint ``(J^2, J_z)`` sectors."""
    amplitudes = state.reshape(2**n_environment, 2**n_system).T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reduced = amplitudes @ amplitudes.conj().T
    if not np.isfinite(reduced).all():
        raise FloatingPointError("Non-finite reduced state.")
    entropy_reduced = cd._entropy(reduced)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = schur_basis.conj().T @ amplitudes
    if not np.isfinite(transformed).all():
        raise FloatingPointError("Non-finite Schur-basis amplitudes.")

    entropy_dephased = 0.0
    offset = 0
    for spin_twice in sorted(paths_by_spin):
        multiplicity = len(paths_by_spin[spin_twice])
        irrep_dimension = spin_twice + 1
        block_size = multiplicity * irrep_dimension
        block = transformed[offset : offset + block_size].reshape(
            multiplicity, irrep_dimension, 2**n_environment
        )
        for magnetic_index in range(irrep_dimension):
            amplitudes_jm = block[:, magnetic_index, :]
            sector_state = amplitudes_jm @ amplitudes_jm.conj().T
            entropy_dephased += cd._entropy(sector_state)
        offset += block_size
    return max(0.0, entropy_dephased - entropy_reduced)


def _basis_dephasing_asymmetry_from_global_pure(
    state: np.ndarray,
    n_system: int,
    n_environment: int,
    dephasing_basis: np.ndarray,
) -> float:
    """Entropy increase under complete dephasing in ``dephasing_basis``.

    For a degenerate generator such as ``J^2``, this operation depends on the
    arbitrary eigenvectors returned by the diagonalization routine. It is not
    the symmetry twirl and is used only to test a possible implementation bug.
    """
    amplitudes = state.reshape(2**n_environment, 2**n_system).T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reduced = amplitudes @ amplitudes.conj().T
        transformed = dephasing_basis.conj().T @ amplitudes
    if not np.isfinite(reduced).all() or not np.isfinite(transformed).all():
        raise FloatingPointError("Non-finite basis-dephasing calculation.")
    entropy_reduced = cd._entropy(reduced)
    probabilities = np.sum(np.abs(transformed) ** 2, axis=1).real
    probabilities = probabilities[probabilities > 1e-14]
    entropy_dephased = -np.sum(probabilities * np.log(probabilities))
    return max(0.0, float(entropy_dephased - entropy_reduced))


def _load_figure_reference(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(
            csv.DictReader(line for line in handle if not line.startswith("#"))
        )
    times = np.array([float(row["displayed_time"]) for row in rows])
    curves = np.column_stack(
        [
            np.array([float(row[f"theta_{theta:.2f}_pi"]) for row in rows])
            for theta in THETA_OVER_PI
        ]
    )
    return times, curves


def _reference_at(
    displayed_times: np.ndarray,
    reference_times: np.ndarray,
    reference_curves: np.ndarray,
) -> np.ndarray:
    indices = np.abs(
        reference_times[:, None] - displayed_times[None, :]
    ).argmin(axis=0)
    return reference_curves[indices].T


def run_candidate(
    *,
    candidate: str,
    measure: str,
    temporal_disorder: str,
    n_realizations: int,
    steps: int,
    sample_times: np.ndarray,
    coupling_scale: float,
    seed: int,
) -> np.ndarray:
    """Return raw curves with shape ``(realization, theta, sample_time)``."""
    n_system = 8
    n_environment = 12
    n_total = n_system + n_environment
    if sample_times[0] != 0 or sample_times[-1] > steps:
        raise ValueError("sample_times must include 0 and not exceed steps.")

    two_branch_candidate = candidate in {
        "equation79",
        "equation79_double_angle",
        "equation79_shifted_half_angle",
        "eq79_swapped_singlet_environment",
        "equation79_polarized_environment",
        "equation79_swapped_polarized_environment",
    }
    if two_branch_candidate:
        if candidate.endswith("polarized_environment"):
            environment = np.zeros(2**n_environment, dtype=complex)
            environment[0] = 1
        else:
            environment = nu.singlet_product(n_environment)
        system_singlet = nu.singlet_product(n_system)
        system_polarized = np.zeros(2**n_system, dtype=complex)
        system_polarized[0] = 1
        initial_states = np.column_stack(
            [
                np.kron(environment, system_singlet),
                np.kron(environment, system_polarized),
            ]
        )
    else:
        initial_states = _initial_states(candidate)

    schur_basis, paths_by_spin = nu.su2_schur_basis(
        n_system, convention="manuscript"
    )
    j2_eigenbasis = None
    if measure == "j2_eigenbasis_dephasing":
        collective = nu.collective_spin(n_system)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            j_squared = sum(
                component @ component for component in collective
            )
        if not np.isfinite(j_squared).all():
            raise FloatingPointError("Non-finite J^2 operator.")
        _, j2_eigenbasis = np.linalg.eigh(j_squared)
    masks = fn.load_mask_memory(n_total)
    ordering = fn.gen_gates_order(n_total, geometry="brickwork")
    time_to_index = {
        int(time): index for index, time in enumerate(sample_times)
    }
    curves = np.zeros(
        (n_realizations, len(THETA_OVER_PI), len(sample_times))
    )
    started = perf_counter()
    for realization in range(n_realizations):
        rng = np.random.default_rng(seed + realization)
        gates = None
        if temporal_disorder == "fixed":
            couplings = rng.uniform(
                -coupling_scale, coupling_scale, n_total
            )
            gates = [
                fn.gen_su2(float(coupling)) for coupling in couplings
            ]
        states = initial_states.copy()

        for time in range(steps + 1):
            if time in time_to_index:
                sample_index = time_to_index[time]
                for theta_index, theta_fraction in enumerate(THETA_OVER_PI):
                    if two_branch_candidate:
                        theta = theta_fraction * np.pi
                        if candidate == "equation79_double_angle":
                            singlet_amplitude = np.cos(theta)
                            polarized_amplitude = np.sin(theta)
                        elif candidate == "equation79_shifted_half_angle":
                            effective_angle = np.pi * (
                                0.25 + theta_fraction / 2
                            )
                            singlet_amplitude = np.cos(effective_angle)
                            polarized_amplitude = np.sin(effective_angle)
                        elif candidate in {
                            "equation79",
                            "equation79_polarized_environment",
                        }:
                            singlet_amplitude = np.cos(theta / 2)
                            polarized_amplitude = np.sin(theta / 2)
                        else:
                            singlet_amplitude = np.sin(theta / 2)
                            polarized_amplitude = np.cos(theta / 2)
                        state = np.ascontiguousarray(
                            singlet_amplitude * states[:, 0]
                            + polarized_amplitude * states[:, 1]
                        )
                    else:
                        state = np.ascontiguousarray(
                            states[:, theta_index]
                        )
                    if measure == "full_twirl":
                        value = cd._su2_asymmetry_from_global_pure(
                            state,
                            n_system,
                            n_environment,
                            schur_basis,
                            paths_by_spin,
                        )
                    elif measure == "jm_dephasing":
                        value = _jm_dephasing_asymmetry_from_global_pure(
                            state,
                            n_system,
                            n_environment,
                            schur_basis,
                            paths_by_spin,
                        )
                    else:
                        assert j2_eigenbasis is not None
                        value = _basis_dephasing_asymmetry_from_global_pure(
                            state,
                            n_system,
                            n_environment,
                            j2_eigenbasis,
                        )
                    curves[realization, theta_index, sample_index] = (
                        value
                    )
            if time < steps:
                if temporal_disorder == "fresh":
                    couplings = rng.uniform(
                        -coupling_scale, coupling_scale, n_total
                    )
                    gates = [
                        fn.gen_su2(float(coupling))
                        for coupling in couplings
                    ]
                assert gates is not None
                states = cd._apply_layer_batch(
                    states, gates, ordering, masks
                )

        elapsed = perf_counter() - started
        print(
            f"{candidate}, {measure}, {temporal_disorder}, "
            f"scale={coupling_scale:.8g}: "
            f"{realization + 1}/{n_realizations} realizations "
            f"({elapsed:.1f} s)"
        )
    return curves


def _parse_integer_list(value: str) -> np.ndarray:
    result = np.unique(
        np.array([int(item.strip()) for item in value.split(",")], dtype=int)
    )
    if result.size == 0 or result[0] < 0:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of nonnegative integers"
        )
    return result


def _parse_float_list(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of floats"
        ) from error
    if not result or any(not np.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "coupling scales must be finite and nonnegative"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=[
            "equation79",
            "equation79_double_angle",
            "equation79_shifted_half_angle",
            "equation79_polarized_environment",
            "equation79_swapped_polarized_environment",
            "historical_product",
            "eq79_swapped_singlet_environment",
            "pair_product_singlet_environment",
            "pair_product_swapped_singlet_environment",
            "staggered_system_singlet_environment",
            "staggered_global",
        ],
        default="historical_product",
    )
    parser.add_argument(
        "--measure",
        choices=[
            "full_twirl",
            "jm_dephasing",
            "j2_eigenbasis_dephasing",
        ],
        default="full_twirl",
    )
    parser.add_argument(
        "--temporal-disorder",
        choices=["fixed", "fresh"],
        default="fixed",
    )
    parser.add_argument(
        "--log-base",
        choices=["e", "2"],
        default="e",
        help="Entropy unit used for both the candidate and figure comparison.",
    )
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--sample-times",
        type=_parse_integer_list,
        default=_parse_integer_list("0,1,2,5,10,20"),
    )
    parser.add_argument(
        "--coupling-scales",
        type=_parse_float_list,
        default=[np.pi / 5],
        help="Comma-separated half-widths of the uniform coupling law.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--displayed-time-per-layer",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "data/circuit_examples/su2_fig11_vector_reference.csv"
        ),
    )
    args = parser.parse_args()

    if args.realizations < 1:
        parser.error("--realizations must be positive")
    if args.steps < 0 or args.sample_times[-1] > args.steps:
        parser.error("--steps must cover every --sample-times entry")
    if args.sample_times[0] != 0:
        parser.error("--sample-times must contain 0")
    if (
        not np.isfinite(args.displayed_time_per_layer)
        or args.displayed_time_per_layer <= 0
    ):
        parser.error("--displayed-time-per-layer must be positive")

    reference_times, reference_curves = _load_figure_reference(
        args.reference
    )
    displayed_times = (
        args.sample_times.astype(float) * args.displayed_time_per_layer
    )
    positive = args.sample_times > 0
    reference = _reference_at(
        displayed_times[positive],
        reference_times,
        reference_curves,
    )

    for coupling_scale in args.coupling_scales:
        raw = run_candidate(
            candidate=args.candidate,
            measure=args.measure,
            temporal_disorder=args.temporal_disorder,
            n_realizations=args.realizations,
            steps=args.steps,
            sample_times=args.sample_times,
            coupling_scale=coupling_scale,
            seed=args.seed,
        )
        if args.log_base == "2":
            raw = raw / np.log(2)
        mean = raw.mean(axis=0)
        residual = mean[:, positive] - reference
        rmse = float(np.sqrt(np.mean(residual**2)))

        print(
            f"\n{args.candidate}; {args.measure}; "
            f"{args.temporal_disorder}; "
            f"log base {args.log_base}; "
            f"coupling scale {coupling_scale:.12g}; RMSE={rmse:.6g}"
        )
        header = "theta " + " ".join(
            f"t={time:g}"
            for time in displayed_times
        )
        print(header)
        for theta, values in zip(THETA_OVER_PI, mean):
            print(
                f"{theta:.2f}  "
                + " ".join(f"{value:.8f}" for value in values)
            )
        print("Reference at positive displayed times")
        for theta, values in zip(THETA_OVER_PI, reference):
            print(
                f"{theta:.2f}  "
                + " ".join(f"{value:.8f}" for value in values)
            )


if __name__ == "__main__":
    main()
