"""Reproducible data pipelines for the manuscript's random-circuit examples.

The routines in this module use the gate definitions and brickwork convention
from :mod:`functions` and :mod:`circuit_obj`.  They return raw, per-realization
arrays; plotting and ensemble reduction deliberately remain in the notebooks.

The non-Markovian U(1) pipeline can also consume the archived parameter rows
from ``alessum/mpemba_circuits``.  This keeps the reference-repository run
bit-for-bit tied to its published circuit samples while using the faster,
validated batched state-vector implementation below.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.linalg as la

import functions as fn
import notebook_utils as nu
from circuit_obj import Circuit


Progress = Callable[[str], None]


def _entropy_from_eigenvalues(values: np.ndarray, cutoff: float = 1e-13) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    values = values[values > cutoff]
    return float(-np.sum(values * np.log(values)))


def _entropy(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.conj().T) / 2
    return _entropy_from_eigenvalues(la.eigvalsh(matrix).real)


def _u1_asymmetry(state: np.ndarray, n_system: int) -> float:
    """Relative entropy of U(1) asymmetry, S(G[rho]) - S(rho)."""
    weights = nu.hamming_weights(n_system)
    twirled_entropy = 0.0
    for weight in range(n_system + 1):
        indices = np.flatnonzero(weights == weight)
        block = state[np.ix_(indices, indices)]
        twirled_entropy += _entropy(block)
    return max(0.0, twirled_entropy - _entropy(state))


def _reduced_density_from_pure(
    state: np.ndarray, n_system: int, n_environment: int
) -> np.ndarray:
    """Trace the leading environment qubits from ``|state><state|``."""
    amplitudes = state.reshape(2**n_environment, 2**n_system).T
    density = amplitudes @ amplitudes.conj().T
    return (density + density.conj().T) / 2


def _reduced_densities_from_pure_batch(
    states: np.ndarray, n_system: int, n_environment: int
) -> np.ndarray:
    """Trace the environment from several column-stacked pure states."""
    n_states = states.shape[1]
    amplitudes = states.reshape(
        2**n_environment, 2**n_system, n_states
    ).transpose(1, 0, 2)
    densities = np.einsum(
        "sek,tek->stk", amplitudes, amplitudes.conj(), optimize=True
    )
    return (
        densities + densities.conj().transpose(1, 0, 2)
    ) / 2


def _apply_channel(channel: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply a numerically trace-corrected column-vectorized channel."""
    dimension = state.shape[0]
    # Some Accelerate/BLAS builds emit spurious floating-point warnings for
    # finite complex matrix products.  The explicit finiteness check below still
    # catches genuine numerical failures.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output = nu.unvec(channel @ nu.vec(state), dimension)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite value produced by the channel.")
    output = (output + output.conj().T) / 2
    return output / np.trace(output)


def _random_u1_gates(
    rng: np.random.Generator, n_total: int, parameter_divisor: float
) -> tuple[list[np.ndarray], np.ndarray]:
    parameters = rng.uniform(-np.pi, np.pi, (n_total, 5)) / parameter_divisor
    gates = [fn.gen_u1(row.tolist()) for row in parameters]
    return gates, parameters


def _apply_layer_batch(
    states: np.ndarray,
    gates: list[np.ndarray],
    ordering: np.ndarray,
    masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Apply one original brickwork layer to several state vectors at once.

    ``states[:, k]`` is one state vector.  The gather/apply/scatter operation is
    algebraically identical to ``functions.apply_U`` and merely batches the
    shared local gates to reduce the production runtime.
    """
    output = np.asarray(states, dtype=np.complex128)
    for gate, site in zip(gates, ordering):
        local = output[masks[int(site)]]
        transformed = np.einsum(
            "ab,bmk->amk", gate, local, optimize=True
        )
        updated = np.empty_like(output)
        updated[masks[int(site)]] = transformed
        output = updated
    return output


def _validate_batched_layer() -> None:
    """Guard the optimized state-vector path against convention drift."""
    n_total = 4
    rng = np.random.default_rng(91)
    masks = fn.load_mask_memory(n_total)
    ordering = fn.gen_gates_order(n_total, geometry="brickwork")
    gates, _ = _random_u1_gates(rng, n_total, parameter_divisor=3)
    states = rng.normal(size=(2**n_total, 2)) + 1j * rng.normal(
        size=(2**n_total, 2)
    )
    states /= la.norm(states, axis=0)
    batched = _apply_layer_batch(states, gates, ordering, masks)
    reference = np.column_stack(
        [
            fn.apply_U(states[:, index], gates, ordering, masks, 2)
            for index in range(states.shape[1])
        ]
    )
    if not np.allclose(batched, reference, atol=1e-12):
        raise RuntimeError("The batched layer disagrees with functions.apply_U.")


def generate_u1_markovian(
    *,
    n_realizations: int = 100,
    steps: int = 200,
    n_system: int = 4,
    n_environment: int = 2,
    parameter_divisor: float = 3,
    seed: int = 10_000,
    progress: Progress = print,
) -> dict[str, np.ndarray]:
    """Run the reset-environment U(1) protocol behind manuscript Fig. 9."""
    n_total = n_system + n_environment
    times = np.arange(steps + 1)
    theta_over_pi = np.array([0.1, 0.2, 0.5])
    block_sizes = np.array([1, 2, 3], dtype=int)
    initial_states = []
    for theta_fraction, block_size in zip(theta_over_pi, block_sizes):
        ket = nu.multi_spin_tilted_state(
            n_system, theta_fraction * np.pi, int(block_size)
        )
        initial_states.append(np.outer(ket, ket.conj()))

    curves = np.zeros((n_realizations, len(initial_states), len(times)))
    rates = np.zeros((n_realizations, n_system + 1))
    overlaps = np.zeros(
        (n_realizations, len(initial_states), n_system + 1)
    )
    parameter_samples = np.zeros((n_realizations, n_total, 5))

    # All blocks together contain exactly d_s^2 eigenvalues.
    spectrum = np.zeros((n_realizations, 4**n_system), dtype=complex)
    spectrum_charges = []
    masks = fn.load_mask_memory(n_total)
    ordering = fn.gen_gates_order(n_total, geometry="brickwork")

    started = perf_counter()
    for realization in range(n_realizations):
        rng = np.random.default_rng(seed + realization)
        gates, parameters = _random_u1_gates(
            rng, n_total, parameter_divisor
        )
        parameter_samples[realization] = parameters
        circuit = Circuit(
            N=n_total,
            T=steps,
            gates=gates,
            order=ordering,
            symmetry="U1",
        )
        unitary = circuit.generate_unitary(masks)
        channel = nu.reduced_channel(
            unitary, n_system, n_environment
        )

        slow_values, slow_left_modes = nu.slowest_charge_data(
            channel, n_system
        )
        values_this_realization = []
        charges_this_realization = []
        for charge, (block, _) in nu.channel_charge_blocks(
            channel, n_system
        ).items():
            block_values = la.eigvals(block)
            values_this_realization.extend(block_values)
            charges_this_realization.extend([charge] * len(block_values))
        spectrum[realization] = np.asarray(values_this_realization)
        if not spectrum_charges:
            spectrum_charges = charges_this_realization

        for charge in range(n_system + 1):
            rates[realization, charge] = np.log(
                abs(slow_values[charge])
            )

        for state_index, initial_state in enumerate(initial_states):
            state = initial_state.copy()
            state_vector = nu.vec(initial_state)
            for charge in range(n_system + 1):
                overlaps[realization, state_index, charge] = abs(
                    np.vdot(slow_left_modes[charge], state_vector)
                )
            for time in times:
                curves[realization, state_index, time] = _u1_asymmetry(
                    state, n_system
                )
                if time < steps:
                    state = _apply_channel(channel, state)

        if realization == 0 or (realization + 1) % 10 == 0:
            progress(
                f"U(1) Markov: {realization + 1}/{n_realizations} "
                f"({perf_counter() - started:.1f} s)"
            )

    return {
        "times": times,
        "theta_over_pi": theta_over_pi,
        "block_sizes": block_sizes,
        "asymmetry": curves,
        "slowest_log_modulus": rates,
        "slow_mode_overlap": overlaps,
        "spectrum": spectrum,
        "spectrum_charge": np.asarray(spectrum_charges, dtype=int),
        "gate_parameters": parameter_samples,
        "n_system": np.array(n_system),
        "n_environment": np.array(n_environment),
        "n_realizations": np.array(n_realizations),
        "parameter_divisor": np.array(parameter_divisor),
        "seed": np.array(seed),
        "environment": np.array("maximally mixed; reset after every layer"),
        "protocol": np.array("U(1) Markovian brickwork, Eqs. (68)-(77)"),
        "manuscript_figure": np.array("Fig. 9"),
        "data_level": np.array("manuscript system and ensemble size"),
    }


def generate_u1_nonmarkovian(
    *,
    n_realizations: int = 24,
    steps: int = 300,
    n_system: int = 4,
    n_environment: int = 8,
    parameter_divisor: float = 1,
    seed: int = 20_000,
    gate_parameters: np.ndarray | None = None,
    source_indices: np.ndarray | None = None,
    source_repository: str = "",
    source_commit: str = "",
    source_parameter_file: str = "",
    source_parameter_sha256: str = "",
    data_level: str | None = None,
    sample_times: np.ndarray | None = None,
    progress: Progress = print,
) -> dict[str, np.ndarray]:
    """Run a state-vector U(1) protocol corresponding to manuscript Fig. 10."""
    _validate_batched_layer()
    n_total = n_system + n_environment
    supplied_parameters = None
    if gate_parameters is not None:
        supplied_parameters = np.asarray(gate_parameters, dtype=float)
        expected_shape = (n_realizations, n_total, 5)
        if supplied_parameters.shape != expected_shape:
            raise ValueError(
                "gate_parameters must have shape "
                f"{expected_shape}, got {supplied_parameters.shape}."
            )
        if not np.isfinite(supplied_parameters).all():
            raise ValueError("gate_parameters contains non-finite values.")
        if source_indices is None:
            source_indices = np.arange(n_realizations)
        source_indices = np.asarray(source_indices, dtype=int)
        if source_indices.shape != (n_realizations,):
            raise ValueError(
                "source_indices must contain one index per realization."
            )
    elif source_indices is not None:
        raise ValueError(
            "source_indices is meaningful only when gate_parameters is supplied."
        )

    if sample_times is None:
        times = np.arange(steps + 1)
    else:
        times = np.unique(np.asarray(sample_times, dtype=int))
        if (
            times.ndim != 1
            or len(times) == 0
            or times[0] != 0
            or times[-1] > steps
            or np.any(times < 0)
        ):
            raise ValueError(
                "sample_times must be a nonempty one-dimensional collection "
                "containing t=0 with values between 0 and steps."
            )
    time_to_index = {int(time): index for index, time in enumerate(times)}
    theta_over_pi = np.array([0.30, 0.35, 0.40, 0.45, 0.50])
    environment = np.zeros(2**n_environment, dtype=complex)
    environment[0] = 1
    system_states = [
        nu.multi_spin_tilted_state(n_system, theta * np.pi, 1)
        for theta in theta_over_pi
    ]

    curves = np.zeros(
        (n_realizations, len(theta_over_pi), len(times))
    )
    parameters = np.zeros((n_realizations, n_total, 5))
    masks = fn.load_mask_memory(n_total)
    ordering = fn.gen_gates_order(n_total, geometry="brickwork")
    started = perf_counter()

    for realization in range(n_realizations):
        if supplied_parameters is None:
            rng = np.random.default_rng(seed + realization)
            gates, parameter_sample = _random_u1_gates(
                rng, n_total, parameter_divisor
            )
        else:
            parameter_sample = supplied_parameters[realization]
            gates = [
                fn.gen_u1(row.tolist()) for row in parameter_sample
            ]
        parameters[realization] = parameter_sample
        states = np.column_stack(
            [np.kron(environment, system) for system in system_states]
        )
        for time in range(steps + 1):
            if time in time_to_index:
                densities = _reduced_densities_from_pure_batch(
                    states, n_system, n_environment
                )
                sample_index = time_to_index[time]
                for state_index in range(states.shape[1]):
                    curves[
                        realization, state_index, sample_index
                    ] = _u1_asymmetry(
                        densities[:, :, state_index], n_system
                    )
            if time < steps:
                states = _apply_layer_batch(
                    states, gates, ordering, masks
                )

        if (
            realization == 0
            or realization + 1 == n_realizations
            or (realization + 1) % 4 == 0
        ):
            progress(
                f"U(1) non-Markov: {realization + 1}/{n_realizations} "
                f"({perf_counter() - started:.1f} s)"
            )

    if data_level is None:
        data_level = (
            "scaled verification run; paper uses Ns=8, Ne=12, R=100"
        )

    dataset = {
        "times": times,
        "theta_over_pi": theta_over_pi,
        "asymmetry": curves,
        "gate_parameters": parameters,
        "n_system": np.array(n_system),
        "n_environment": np.array(n_environment),
        "n_realizations": np.array(n_realizations),
        "parameter_divisor": np.array(parameter_divisor),
        "seed": np.array(seed),
        "environment": np.array("|0...0>; no reset"),
        "protocol": np.array("U(1) non-Markovian brickwork, Eqs. (68)-(72), (78)"),
        "manuscript_figure": np.array("Fig. 10"),
        "data_level": np.array(data_level),
        "paper_n_system": np.array(8),
        "paper_n_environment": np.array(12),
        "paper_n_realizations": np.array(100),
        "paper_steps": np.array(1_000),
        "parameter_origin": np.array(
            "supplied reference rows"
            if supplied_parameters is not None
            else "NumPy default_rng draws"
        ),
    }
    if supplied_parameters is not None:
        dataset.update(
            {
                "source_indices": source_indices,
                "source_repository": np.array(source_repository),
                "source_commit": np.array(source_commit),
                "source_parameter_file": np.array(source_parameter_file),
                "source_parameter_sha256": np.array(
                    source_parameter_sha256
                ),
            }
        )
    return dataset


def _su2_asymmetry_from_global_pure(
    state: np.ndarray,
    n_system: int,
    n_environment: int,
    schur_basis: np.ndarray,
    paths_by_spin: dict[int, list[tuple[int, ...]]],
) -> float:
    """Exact SU(2) asymmetry without materializing the dense twirled state."""
    amplitudes = state.reshape(2**n_environment, 2**n_system).T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reduced = amplitudes @ amplitudes.conj().T
    if not np.isfinite(reduced).all():
        raise FloatingPointError("Non-finite reduced SU(2) state.")
    entropy_reduced = _entropy(reduced)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = schur_basis.conj().T @ amplitudes
    if not np.isfinite(transformed).all():
        raise FloatingPointError("Non-finite Schur-basis amplitudes.")
    entropy_twirled = 0.0
    offset = 0
    for spin_twice in sorted(paths_by_spin):
        multiplicity = len(paths_by_spin[spin_twice])
        irrep_dimension = spin_twice + 1
        block_size = multiplicity * irrep_dimension
        block = transformed[offset : offset + block_size].reshape(
            multiplicity, irrep_dimension, 2**n_environment
        )
        multiplicity_state_over_d = np.einsum(
            "ame,bme->ab", block, block.conj(), optimize=True
        ) / irrep_dimension
        eigenvalues = la.eigvalsh(
            (
                multiplicity_state_over_d
                + multiplicity_state_over_d.conj().T
            )
            / 2
        ).real
        entropy_twirled += irrep_dimension * _entropy_from_eigenvalues(
            eigenvalues
        )
        offset += block_size
    return max(0.0, entropy_twirled - entropy_reduced)


def generate_su2_nonmarkovian(
    *,
    n_realizations: int = 3,
    steps: int = 100,
    n_system: int = 8,
    n_environment: int = 12,
    coupling_scale: float = np.pi / 5,
    seed: int = 30_000,
    sample_times: np.ndarray | None = None,
    progress: Progress = print,
) -> dict[str, np.ndarray]:
    """Run the SU(2)-covariant open-system protocol defined by Eq. (79).

    This routine intentionally implements the printed state
    ``cos(theta/2)|singlets> + sin(theta/2)|0...0>`` in natural-log units.
    It does not silently exchange the coefficients to imitate Fig. 11: the
    vector data in that figure are inconsistent with Eq. (79) under the stated
    covariant dynamics, so figure comparison belongs in the audit notebook.

    ``sample_times`` can reduce measurement cost while still evolving every
    Floquet layer. Couplings are stored in deterministic brickwork application
    order (even bonds, then odd bonds), and a single sampled unitary is
    repeated, as required for the non-Markovian protocol.
    """
    _validate_batched_layer()
    if n_realizations < 1:
        raise ValueError("n_realizations must be positive.")
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    if n_system < 2 or n_environment < 2:
        raise ValueError("Both partitions must contain at least two qubits.")
    if n_system % 2 or n_environment % 2:
        raise ValueError("The singlet preparation requires even subsystem sizes.")
    if not np.isfinite(coupling_scale) or coupling_scale < 0:
        raise ValueError("coupling_scale must be finite and nonnegative.")
    n_total = n_system + n_environment
    if sample_times is None:
        times = np.arange(steps + 1)
    else:
        times = np.unique(np.asarray(sample_times, dtype=int))
        if (
            times.ndim != 1
            or len(times) == 0
            or times[0] != 0
            or times[-1] > steps
            or np.any(times < 0)
        ):
            raise ValueError(
                "sample_times must be a nonempty one-dimensional collection "
                "containing t=0 with values between 0 and steps."
            )
    time_to_index = {int(time): index for index, time in enumerate(times)}
    theta_over_pi = np.array([0.30, 0.35, 0.40, 0.45, 0.50])
    environment = nu.singlet_product(n_environment)
    system_singlet = nu.singlet_product(n_system)
    system_polarized = np.zeros(2**n_system, dtype=complex)
    system_polarized[0] = 1
    schur_basis, paths_by_spin = nu.su2_schur_basis(n_system)

    curves = np.zeros(
        (n_realizations, len(theta_over_pi), len(times))
    )
    coupling_samples = np.zeros((n_realizations, n_total))
    masks = fn.load_mask_memory(n_total)
    ordering = fn.gen_gates_order(n_total, geometry="brickwork")
    started = perf_counter()

    for realization in range(n_realizations):
        rng = np.random.default_rng(seed + realization)
        couplings = rng.uniform(-coupling_scale, coupling_scale, n_total)
        coupling_samples[realization] = couplings
        gates = [
            fn.gen_su2(float(coupling)) for coupling in couplings
        ]

        # Eq. (79) spans only two vectors.  Evolving those vectors once and
        # forming all five theta superpositions is exact and saves fivefold work.
        basis_states = np.column_stack(
            [
                np.kron(environment, system_singlet),
                np.kron(environment, system_polarized),
            ]
        )
        for time in range(steps + 1):
            if time in time_to_index:
                sample_index = time_to_index[time]
                for theta_index, theta_fraction in enumerate(theta_over_pi):
                    theta = theta_fraction * np.pi
                    singlet_amplitude = np.cos(theta / 2)
                    polarized_amplitude = np.sin(theta / 2)
                    state = np.ascontiguousarray(
                        singlet_amplitude * basis_states[:, 0]
                        + polarized_amplitude * basis_states[:, 1]
                    )
                    curves[realization, theta_index, sample_index] = (
                        _su2_asymmetry_from_global_pure(
                            state,
                            n_system,
                            n_environment,
                            schur_basis,
                            paths_by_spin,
                        )
                    )
            if time < steps:
                basis_states = _apply_layer_batch(
                    basis_states, gates, ordering, masks
                )

        progress(
            f"SU(2) non-Markov: {realization + 1}/{n_realizations} "
            f"({perf_counter() - started:.1f} s)"
        )

    spins = np.array(sorted(paths_by_spin), dtype=int)
    multiplicities = np.array(
        [len(paths_by_spin[spin]) for spin in spins], dtype=int
    )
    return {
        "times": times,
        "theta_over_pi": theta_over_pi,
        "asymmetry": curves,
        "couplings": coupling_samples,
        "coupling_indexing": np.array(
            "brickwork application order: even bonds, then odd bonds"
        ),
        "spin_twice": spins,
        "multiplicity": multiplicities,
        "n_system": np.array(n_system),
        "n_environment": np.array(n_environment),
        "n_realizations": np.array(n_realizations),
        "coupling_scale": np.array(coupling_scale),
        "coefficient_order": np.array("equation"),
        "entropy_log_base": np.array("e"),
        "state_definition": np.array(
            "Eq. (79): cos(theta/2)|singlets> + "
            "sin(theta/2)|0...0>"
        ),
        "seed": np.array(seed),
        "environment": np.array("product of singlets; no reset"),
        "protocol": np.array(
            "Eq. (79)-consistent SU(2) non-Markovian brickwork validation"
        ),
        "manuscript_figure": np.array("Fig. 11"),
        "data_level": np.array(
            "Eq. (79) validation at manuscript system and ensemble size; "
            "not asserted to reproduce Fig. 11"
            if n_realizations == 100
            else "Eq. (79) validation at manuscript system size; "
            "reduced ensemble; not a Fig. 11 reproduction"
        ),
        "paper_n_realizations": np.array(100),
        "paper_vector_samples_per_curve": np.array(1_001),
    }


def save_dataset(path: str | Path, dataset: dict[str, np.ndarray]) -> Path:
    """Atomically store one dataset as a compressed, pickle-free NPZ file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **dataset)
    os.replace(temporary, target)
    return target
