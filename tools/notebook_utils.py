"""Numerical helpers shared by the pedagogical Mpemba-effect notebooks.

The notebooks keep the physics and the main algorithms visible.  This module
contains only reusable numerical plumbing: stable entropy routines, small spin
models, symmetry twirls, and circuit/channel construction.
"""

from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
import scipy.linalg as la


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
SP = (X + 1j * Y) / 2
SM = (X - 1j * Y) / 2


def dagger(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix).conj().T


def vec(matrix: np.ndarray) -> np.ndarray:
    """Column-stack an operator."""
    return np.asarray(matrix).reshape(-1, order="F")


def unvec(vector: np.ndarray, dimension: int) -> np.ndarray:
    return np.asarray(vector).reshape((dimension, dimension), order="F")


def hermitize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    return (matrix + dagger(matrix)) / 2


def normalize_density(matrix: np.ndarray, *, clip: float = 1e-13) -> np.ndarray:
    """Return the nearest positive, trace-one matrix obtained spectrally."""
    values, vectors = la.eigh(hermitize(matrix))
    values = np.clip(values.real, clip, None)
    values /= values.sum()
    # Some Accelerate/BLAS builds emit spurious floating-point warnings for
    # finite complex matrix products.  The explicit check still catches a
    # genuine numerical failure.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (vectors * values) @ dagger(vectors)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Density normalization produced non-finite values.")
    return normalized


def validate_density(matrix: np.ndarray, *, atol: float = 1e-9) -> None:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A density matrix must be square.")
    if not np.allclose(matrix, dagger(matrix), atol=atol):
        raise ValueError("Density matrix is not Hermitian.")
    if not np.isclose(np.trace(matrix), 1, atol=atol):
        raise ValueError("Density matrix does not have unit trace.")
    if la.eigvalsh(matrix).min() < -atol:
        raise ValueError("Density matrix is not positive semidefinite.")


def von_neumann_entropy(matrix: np.ndarray, *, cutoff: float = 1e-13) -> float:
    values = np.clip(la.eigvalsh(hermitize(matrix)).real, 0, None)
    values = values[values > cutoff]
    return float(-np.sum(values * np.log(values)))


def quantum_relative_entropy(
    rho: np.ndarray, sigma: np.ndarray, *, cutoff: float = 1e-13
) -> float:
    """Compute S(rho || sigma), returning infinity on support mismatch."""
    rho = normalize_density(rho, clip=0)
    sigma = hermitize(sigma)
    r_values = np.clip(la.eigvalsh(rho).real, 0, None)
    s_values, s_vectors = la.eigh(sigma)
    rho_in_s_basis = dagger(s_vectors) @ rho @ s_vectors
    populations = np.clip(np.diag(rho_in_s_basis).real, 0, None)
    if np.any((s_values < cutoff) & (populations > cutoff)):
        return float("inf")
    term_rho = np.sum(
        r_values[r_values > cutoff] * np.log(r_values[r_values > cutoff])
    )
    supported = s_values > cutoff
    term_sigma = np.sum(populations[supported] * np.log(s_values[supported]))
    return float(np.real(term_rho - term_sigma))


def petz_renyi(
    rho: np.ndarray, sigma: np.ndarray, alpha: float, *, cutoff: float = 1e-13
) -> float:
    """Petz-Renyi divergence for 0 < alpha <= 2."""
    if alpha <= 0 or alpha > 2:
        raise ValueError("alpha must satisfy 0 < alpha <= 2")
    if np.isclose(alpha, 1):
        return quantum_relative_entropy(rho, sigma, cutoff=cutoff)

    def matrix_power_psd(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = la.eigh(hermitize(matrix))
        if np.min(values) < -100 * cutoff:
            raise ValueError("Matrix power requested for a non-positive matrix.")
        # The examples include low-temperature Gibbs weights far below machine-
        # practical support.  Clipping implements a transparent spectral
        # regularization without mistaking those positive weights for exact zeros.
        values = np.clip(values, cutoff, None) ** exponent
        return (vectors * values) @ dagger(vectors)

    value = np.trace(
        matrix_power_psd(rho, alpha) @ matrix_power_psd(sigma, 1 - alpha)
    ).real
    return float(np.log(max(value, cutoff)) / (alpha - 1))


def classical_relative_entropy(
    p: np.ndarray, q: np.ndarray, *, cutoff: float = 1e-300
) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    q = np.clip(np.asarray(q, dtype=float), cutoff, None)
    support = p > cutoff
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def classical_renyi(
    p: np.ndarray, q: np.ndarray, alpha: float, *, cutoff: float = 1e-300
) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if np.isclose(alpha, 1):
        return classical_relative_entropy(p, q, cutoff=cutoff)
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    q = np.clip(np.asarray(q, dtype=float), cutoff, None)
    return float(np.log(np.sum(p**alpha * q ** (1 - alpha))) / (alpha - 1))


def crossing_time(times: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """First positive-time crossing from first > second to first < second."""
    times = np.asarray(times, dtype=float)
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    indices = np.flatnonzero((difference[:-1] >= 0) & (difference[1:] < 0))
    if len(indices) == 0:
        return float("nan")
    index = int(indices[0])
    x0, x1 = times[index : index + 2]
    y0, y1 = difference[index : index + 2]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, operators)


def local_operator(operator: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    operators = [I2] * n_sites
    operators[site] = operator
    return kron_all(operators)


def tfim_hamiltonian(
    n_sites: int,
    coupling: float = 1.0,
    field: float = 1.0,
    *,
    spin_half_operators: bool = False,
) -> np.ndarray:
    """Open-boundary TFIM Hamiltonian of Eq. (65).

    ``spin_half_operators=True`` reproduces the normalization used by the
    manuscript's accompanying simulation scripts, where ``s = sigma/2``.
    """
    z_operator = Z / 2 if spin_half_operators else Z
    x_operator = X / 2 if spin_half_operators else X
    dimension = 2**n_sites
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    for site in range(n_sites - 1):
        operators = [I2] * n_sites
        operators[site] = z_operator
        operators[site + 1] = z_operator
        hamiltonian -= coupling * kron_all(operators)
    for site in range(n_sites):
        hamiltonian += field * local_operator(x_operator, site, n_sites)
    return hermitize(hamiltonian)


def gibbs_state(hamiltonian: np.ndarray, beta: float) -> np.ndarray:
    values, vectors = la.eigh(hamiltonian)
    weights = np.exp(-beta * (values - values.min()))
    weights /= weights.sum()
    return (vectors * weights) @ dagger(vectors)


def davies_liouvillian(
    hamiltonian: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Davies generator in the energy basis for all-to-all bosonic transitions.

    Populations obey detailed balance and each energy-basis coherence evolves
    independently.
    """
    energies, energy_basis = la.eigh(hamiltonian)
    dimension = len(energies)
    population_generator = np.zeros((dimension, dimension), dtype=float)

    for high in range(dimension):
        for low in range(high):
            gap = energies[high] - energies[low]
            occupation = 1 / np.expm1(beta * gap)
            population_generator[high, low] = occupation
            population_generator[low, high] = occupation + 1
    np.fill_diagonal(
        population_generator, -population_generator.sum(axis=0)
    )

    liouvillian = np.zeros((dimension**2, dimension**2), dtype=complex)
    for source in range(dimension):
        source_index = source + source * dimension
        for target in range(dimension):
            target_index = target + target * dimension
            liouvillian[target_index, source_index] = population_generator[
                target, source
            ]

    escape_rates = -np.diag(population_generator)
    for row in range(dimension):
        for column in range(dimension):
            if row == column:
                continue
            index = row + column * dimension
            liouvillian[index, index] = (
                -0.5 * (escape_rates[row] + escape_rates[column])
                - 1j * (energies[row] - energies[column])
            )

    weights = np.exp(-beta * (energies - energies.min()))
    steady_state = np.diag(weights / weights.sum()).astype(complex)
    return liouvillian, energies, energy_basis, steady_state


def lindblad_liouvillian(
    hamiltonian: np.ndarray, collapse_operators: list[np.ndarray]
) -> np.ndarray:
    """Dense column-vectorized Lindblad generator for small Hilbert spaces."""
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    dimension = hamiltonian.shape[0]
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("Hamiltonian must be square.")
    identity = np.eye(dimension, dtype=complex)
    generator = -1j * (
        np.kron(identity, hamiltonian)
        - np.kron(hamiltonian.T, identity)
    )
    for collapse in collapse_operators:
        collapse = np.asarray(collapse, dtype=complex)
        if collapse.shape != hamiltonian.shape:
            raise ValueError("Collapse operators must match the Hamiltonian.")
        squared = dagger(collapse) @ collapse
        generator += (
            np.kron(collapse.conj(), collapse)
            - 0.5 * np.kron(identity, squared)
            - 0.5 * np.kron(squared.T, identity)
        )
    return generator


def stationary_density(
    liouvillian: np.ndarray, dimension: int
) -> np.ndarray:
    """Return the trace-one stationary state of a finite Lindbladian."""
    values, vectors = la.eig(liouvillian)
    index = int(np.argmin(np.abs(values)))
    state = unvec(vectors[:, index], dimension)
    state = normalize_density(state)
    residual = la.norm(liouvillian @ vec(state))
    if residual > 1e-8:
        raise RuntimeError(
            f"Stationary-state residual is unexpectedly large: {residual:.3e}"
        )
    return state


def spin_j_operators(
    n_spins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collective spin matrices in the maximal-spin ``j=n_spins/2`` sector."""
    if n_spins < 1:
        raise ValueError("n_spins must be positive")
    spin = n_spins / 2
    dimension = n_spins + 1
    magnetic = np.arange(spin, -spin - 1, -1, dtype=float)
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, value in enumerate(magnetic):
        if column > 0:
            raising[column - 1, column] = np.sqrt(
                spin * (spin + 1) - value * (value + 1)
            )
    lowering = dagger(raising)
    x_operator = (raising + lowering) / 2
    y_operator = (raising - lowering) / (2j)
    z_operator = np.diag(magnetic).astype(complex)
    return x_operator, y_operator, z_operator


def metric_adjusted_qfi(
    state: np.ndarray,
    liouvillian: np.ndarray,
    monotone_function,
    *,
    cutoff: float = 1e-13,
) -> float:
    """QFI about Lindblad time for one operator-monotone function ``f``.

    Implements Eq. (F3) of the published manuscript,

        I_f = sum_xy |<x|L[rho]|y>|^2 /
              (p_x f(p_y / p_x)).
    """
    state = normalize_density(state, clip=0)
    probabilities, basis = la.eigh(state)
    derivative = unvec(liouvillian @ vec(state), state.shape[0])
    derivative = dagger(basis) @ derivative @ basis
    value = 0.0
    for row, probability_row in enumerate(probabilities):
        if probability_row <= cutoff:
            continue
        for column, probability_column in enumerate(probabilities):
            if probability_column <= cutoff:
                continue
            denominator = probability_row * monotone_function(
                probability_column / probability_row
            )
            value += abs(derivative[row, column]) ** 2 / denominator
    return float(np.real(value))


def evolve_density(
    liouvillian: np.ndarray, rho0: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Evolve a density matrix by diagonalizing the small generator once."""
    times = np.asarray(times, dtype=float)
    values, eigenvectors = la.eig(liouvillian)
    coefficients = la.solve(eigenvectors, vec(rho0))
    vectors = eigenvectors @ (
        coefficients[:, None] * np.exp(values[:, None] * times[None, :])
    )
    states = [
        unvec(vectors[:, time_index], rho0.shape[0])
        for time_index in range(len(times))
    ]
    return np.asarray([normalize_density(state) for state in states])


def random_positive_state(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    state = dagger(matrix) @ matrix
    return state / np.trace(state)


def noisy_gibbs_state(
    energies: np.ndarray,
    beta_initial: float,
    noise_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.exp(-beta_initial * (energies - energies.min()))
    thermal = np.diag(weights / weights.sum()).astype(complex)
    random_state = random_positive_state(len(energies), rng)
    return normalize_density(thermal + noise_strength * random_state)


def manuscript_noisy_gibbs_state(
    energies: np.ndarray,
    beta_initial: float,
    noise_strength: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Noisy Gibbs state used by the manuscript's Davies-map scripts.

    The random matrix is multiplied by ``noise_strength`` *before* forming
    ``X†X``; consequently the positive perturbation scales quadratically.
    """
    weights = np.exp(-beta_initial * (energies - energies.min()))
    thermal = np.diag(weights / weights.sum()).astype(complex)
    rng = np.random.RandomState(seed)
    matrix = noise_strength * (
        rng.rand(len(energies), len(energies))
        + 1j * rng.rand(len(energies), len(energies))
    )
    return normalize_density(thermal + dagger(matrix) @ matrix)


def haar_unitary(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q_matrix, r_matrix = la.qr(matrix)
    phases = np.diag(r_matrix)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1)
    return q_matrix @ np.diag(phases.conj())


def product_unitary(unitaries: list[np.ndarray]) -> np.ndarray:
    return kron_all(unitaries)


def minimize_mode_overlap(
    state: np.ndarray,
    left_mode: np.ndarray,
    n_qubits: int,
    *,
    samples: int = 1200,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-search analogue of the paper's local-unitary Metropolis step."""
    rng = np.random.default_rng(seed)
    identity = np.eye(2**n_qubits, dtype=complex)
    best_unitary = identity
    best_state = state.copy()

    def cost(candidate: np.ndarray) -> float:
        return float(abs(np.trace(dagger(left_mode) @ candidate)))

    history = [cost(state)]
    best_cost = history[0]
    for _ in range(samples):
        candidate_unitary = product_unitary(
            [haar_unitary(2, rng) for _ in range(n_qubits)]
        )
        candidate = candidate_unitary @ state @ dagger(candidate_unitary)
        candidate_cost = cost(candidate)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_unitary = candidate_unitary
            best_state = candidate
        history.append(best_cost)
    return normalize_density(best_state), best_unitary, np.asarray(history)


def find_fast_asymmetric_state(
    state: np.ndarray,
    left_mode: np.ndarray,
    n_qubits: int,
    *,
    samples: int = 8_000,
    cost_fraction: float = 0.05,
    seed: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Choose a low-overlap local-unitary rotation with visible asymmetry.

    Among candidates whose target-mode overlap is below ``cost_fraction`` of
    the original overlap, the state with the largest energy-basis asymmetry is
    retained.  This makes the Mpemba crossing easy to see without changing the
    spectrum of the initial state.
    """
    rng = np.random.default_rng(seed)

    def overlap(candidate: np.ndarray) -> float:
        return float(abs(np.trace(dagger(left_mode) @ candidate)))

    initial_overlap = overlap(state)
    threshold = cost_fraction * initial_overlap
    best_state = None
    best_unitary = None
    best_asymmetry = -np.inf
    fallback = (initial_overlap, state, np.eye(2**n_qubits, dtype=complex))

    for _ in range(samples):
        candidate_unitary = product_unitary(
            [haar_unitary(2, rng) for _ in range(n_qubits)]
        )
        candidate = normalize_density(
            candidate_unitary @ state @ dagger(candidate_unitary)
        )
        candidate_overlap = overlap(candidate)
        if candidate_overlap < fallback[0]:
            fallback = (candidate_overlap, candidate, candidate_unitary)
        if candidate_overlap <= threshold:
            candidate_asymmetry = asymmetry_relative_entropy(
                candidate, energy_dephase(candidate)
            )
            if candidate_asymmetry > best_asymmetry:
                best_asymmetry = candidate_asymmetry
                best_state = candidate
                best_unitary = candidate_unitary

    if best_state is None:
        final_overlap, best_state, best_unitary = fallback
        best_asymmetry = asymmetry_relative_entropy(
            best_state, energy_dephase(best_state)
        )
    else:
        final_overlap = overlap(best_state)

    diagnostics = {
        "initial_overlap": initial_overlap,
        "final_overlap": final_overlap,
        "initial_asymmetry": asymmetry_relative_entropy(
            state, energy_dephase(state)
        ),
        "final_asymmetry": best_asymmetry,
    }
    return best_state, best_unitary, diagnostics


def slowest_coherent_mode(
    liouvillian: np.ndarray, dimension: int
) -> tuple[complex, np.ndarray, int]:
    """Return the slowest mode whose matrix has off-diagonal support."""
    values, left_vectors = la.eig(liouvillian, left=True, right=False)
    candidates = []
    for index, value in enumerate(values):
        matrix = unvec(left_vectors[:, index], dimension)
        off_diagonal = matrix - np.diag(np.diag(matrix))
        if la.norm(off_diagonal) > 1e-8 and value.real < -1e-12:
            candidates.append((value.real, index))
    _, index = max(candidates)
    mode = unvec(left_vectors[:, index], dimension)
    mode /= la.norm(mode)
    return values[index], mode, index


def energy_dephase(state: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(state)).astype(complex)


def ising_energies(
    n_sites: int, coupling: float, field: float
) -> tuple[np.ndarray, np.ndarray]:
    configurations = np.asarray(
        list(product([1, -1], repeat=n_sites)), dtype=float
    )
    interactions = np.sum(
        configurations[:, :-1] * configurations[:, 1:], axis=1
    )
    interactions += configurations[:, 0] + configurations[:, -1]
    energies = -coupling * interactions - field * configurations.sum(axis=1)
    return energies, configurations


def classical_ising_generator(
    n_sites: int, coupling: float, field: float, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (11) for the fixed-boundary Ising chain used in Fig. 2."""
    energies, configurations = ising_energies(n_sites, coupling, field)
    differences = configurations[:, None, :] != configurations[None, :, :]
    single_flip = differences.sum(axis=2) == 1
    generator = np.where(
        single_flip,
        np.exp(-0.5 * beta * (energies[:, None] - energies[None, :])),
        0.0,
    )
    np.fill_diagonal(generator, -generator.sum(axis=0))
    weights = np.exp(-beta * (energies - energies.min()))
    equilibrium = weights / weights.sum()
    return generator, energies, configurations, equilibrium


def weighted_eigendecomposition(
    generator: np.ndarray, equilibrium: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = la.eig(generator)
    order = np.argsort(values.real)[::-1]
    values, vectors = values[order].real, vectors[:, order].real
    for column in range(vectors.shape[1]):
        norm = np.sqrt(np.sum(vectors[:, column] ** 2 / equilibrium))
        vectors[:, column] /= norm
    return values, vectors


def optimize_probability_permutation(
    probability: np.ndarray,
    equilibrium: np.ndarray,
    modes: np.ndarray,
    *,
    iterations: int = 30_000,
    seed: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated annealing over permutations, mirroring Eq. (13)."""
    rng = np.random.default_rng(seed)
    weights = modes / equilibrium[:, None]

    def cost(candidate: np.ndarray) -> float:
        return float(np.sum(np.abs(weights.T @ candidate)))

    current = probability.copy()
    best = current.copy()
    current_cost = best_cost = cost(current)
    history = np.empty(iterations + 1)
    history[0] = best_cost
    for step in range(iterations):
        candidate = current.copy()
        indices = rng.choice(len(candidate), size=4, replace=False)
        candidate[indices] = candidate[rng.permutation(indices)]
        candidate_cost = cost(candidate)
        temperature = max(1e-5, 0.02 * (1 - step / iterations))
        if candidate_cost < current_cost or rng.random() < np.exp(
            -(candidate_cost - current_cost) / temperature
        ):
            current, current_cost = candidate, candidate_cost
        if current_cost < best_cost:
            best, best_cost = current.copy(), current_cost
        history[step + 1] = best_cost
    return best, history


def hamming_weights(n_qubits: int) -> np.ndarray:
    return np.asarray([index.bit_count() for index in range(2**n_qubits)])


def u1_twirl(state: np.ndarray, n_qubits: int) -> np.ndarray:
    weights = hamming_weights(n_qubits)
    return np.where(weights[:, None] == weights[None, :], state, 0)


def asymmetry_modes_u1(state: np.ndarray, n_qubits: int) -> dict[int, np.ndarray]:
    weights = hamming_weights(n_qubits)
    modes = {}
    for charge in range(-n_qubits, n_qubits + 1):
        modes[charge] = np.where(
            weights[:, None] - weights[None, :] == charge, state, 0
        )
    return modes


def mode_trace_norms_u1(state: np.ndarray, n_qubits: int) -> dict[int, float]:
    return {
        charge: float(np.sum(la.svdvals(mode)))
        for charge, mode in asymmetry_modes_u1(state, n_qubits).items()
    }


def apply_gate_to_state(
    state: np.ndarray, gate: np.ndarray, sites: tuple[int, int], n_qubits: int
) -> np.ndarray:
    """Apply a two-qubit gate to a state vector with big-endian tensor axes."""
    first, second = sites
    tensor = np.asarray(state).reshape((2,) * n_qubits)
    remaining = [site for site in range(n_qubits) if site not in sites]
    permutation = [first, second] + remaining
    inverse = np.argsort(permutation)
    front = tensor.transpose(permutation).reshape(4, -1)
    return (gate @ front).reshape((2,) * n_qubits).transpose(inverse).reshape(-1)


def embed_two_qubit_gate(
    gate: np.ndarray, sites: tuple[int, int], n_qubits: int
) -> np.ndarray:
    dimension = 2**n_qubits
    unitary = np.zeros((dimension, dimension), dtype=complex)
    for column in range(dimension):
        basis = np.zeros(dimension, dtype=complex)
        basis[column] = 1
        unitary[:, column] = apply_gate_to_state(
            basis, gate, sites, n_qubits
        )
    return unitary


def u1_gate(rng: np.random.Generator, scale: float = np.pi / 5) -> np.ndarray:
    """Random XXZ gate with local z rotations, Eqs. (71)-(72)."""
    h_left, h_right, phase, coupling, coupling_z = rng.uniform(
        -scale, scale, size=5
    )
    interaction = (
        coupling
        / 2
        * (
            np.exp(1j * phase) * np.kron(SP, SM)
            + np.exp(-1j * phase) * np.kron(SM, SP)
        )
        + coupling_z * np.kron(Z / 2, Z / 2)
    )
    local_z = h_left * np.kron(Z / 2, I2) + h_right * np.kron(I2, Z / 2)
    return la.expm(-1j * interaction) @ la.expm(-1j * local_z)


def su2_gate(coupling: float) -> np.ndarray:
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=complex,
    )
    return np.cos(coupling / 2) * np.eye(4) - 1j * np.sin(
        coupling / 2
    ) * swap


def brickwork_pairs(n_qubits: int) -> list[tuple[int, int]]:
    even = [(site, site + 1) for site in range(0, n_qubits, 2)]
    odd = [
        (site, (site + 1) % n_qubits) for site in range(1, n_qubits, 2)
    ]
    return even + odd


def brickwork_unitary_u1(
    n_qubits: int,
    rng: np.random.Generator,
    *,
    scale: float = np.pi,
) -> np.ndarray:
    dimension = 2**n_qubits
    unitary = np.eye(dimension, dtype=complex)
    for sites in brickwork_pairs(n_qubits):
        unitary = embed_two_qubit_gate(
            u1_gate(rng, scale=scale), sites, n_qubits
        ) @ unitary
    return unitary


def reduced_channel(
    unitary: np.ndarray,
    n_system: int,
    n_environment: int,
    environment_state: np.ndarray | None = None,
) -> np.ndarray:
    """Channel on the last system qubits, using column-vectorization."""
    d_environment = 2**n_environment
    d_system = 2**n_system
    if environment_state is None:
        environment_state = np.eye(d_environment) / d_environment
    tensor = unitary.reshape(
        d_environment, d_system, d_environment, d_system
    )
    channel_tensor = np.einsum(
        "aibj,bc,akcl->ikjl",
        tensor,
        environment_state,
        tensor.conj(),
        optimize=True,
    )
    return channel_tensor.transpose(0, 1, 2, 3).reshape(
        d_system**2, d_system**2, order="F"
    )


def apply_channel(channel: np.ndarray, state: np.ndarray) -> np.ndarray:
    return normalize_density(unvec(channel @ vec(state), state.shape[0]))


def multi_spin_tilted_state(
    n_qubits: int, theta: float, block_size: int
) -> np.ndarray:
    """Eq. (75): disjoint exp[-i theta Y^(x b)/2] rotations of |0...0>."""
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1
    gate = la.expm(
        -0.5j * theta * kron_all([Y] * block_size)
    )
    for start in range(0, n_qubits - block_size + 1, block_size):
        sites = tuple(range(start, start + block_size))
        if block_size == 1:
            tensor = state.reshape((2,) * n_qubits)
            remaining = [site for site in range(n_qubits) if site != start]
            permutation = [start] + remaining
            inverse = np.argsort(permutation)
            state = (
                gate
                @ tensor.transpose(permutation).reshape(2, -1)
            ).reshape((2,) * n_qubits).transpose(inverse).reshape(-1)
        elif block_size == 2:
            state = apply_gate_to_state(state, gate, sites, n_qubits)
        else:
            tensor = state.reshape((2,) * n_qubits)
            remaining = [site for site in range(n_qubits) if site not in sites]
            permutation = list(sites) + remaining
            inverse = np.argsort(permutation)
            front = tensor.transpose(permutation).reshape(2**block_size, -1)
            state = (
                gate @ front
            ).reshape((2,) * n_qubits).transpose(inverse).reshape(-1)
    return state / la.norm(state)


def channel_charge_blocks(
    channel: np.ndarray, n_qubits: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    dimension = 2**n_qubits
    weights = hamming_weights(n_qubits)
    blocks = {}
    for charge in range(-n_qubits, n_qubits + 1):
        indices = np.asarray(
            [
                row + column * dimension
                for column in range(dimension)
                for row in range(dimension)
                if weights[row] - weights[column] == charge
            ],
            dtype=int,
        )
        blocks[charge] = (channel[np.ix_(indices, indices)], indices)
    return blocks


def slowest_charge_data(
    channel: np.ndarray, n_qubits: int
) -> tuple[dict[int, complex], dict[int, np.ndarray]]:
    eigenvalues = {}
    left_modes = {}
    for charge, (block, indices) in channel_charge_blocks(
        channel, n_qubits
    ).items():
        values, left = la.eig(block, left=True, right=False)
        order = np.argsort(np.abs(values))[::-1]
        if charge == 0:
            order = [index for index in order if abs(values[index] - 1) > 1e-8]
        index = int(order[0])
        eigenvalues[charge] = values[index]
        mode = np.zeros(channel.shape[0], dtype=complex)
        mode[indices] = left[:, index] / la.norm(left[:, index])
        left_modes[charge] = mode
    return eigenvalues, left_modes


def partial_trace_pure_state(
    state: np.ndarray, keep: list[int], n_qubits: int
) -> np.ndarray:
    trace = [site for site in range(n_qubits) if site not in keep]
    tensor = state.reshape((2,) * n_qubits).transpose(keep + trace)
    matrix = tensor.reshape(2 ** len(keep), 2 ** len(trace))
    return normalize_density(matrix @ dagger(matrix))


def singlet_product(n_qubits: int) -> np.ndarray:
    if n_qubits % 2:
        raise ValueError("A singlet product requires an even number of qubits.")
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return kron_all([singlet] * (n_qubits // 2))


def su2_tilted_state(n_qubits: int, theta: float) -> np.ndarray:
    singlets = singlet_product(n_qubits)
    polarized = np.zeros(2**n_qubits, dtype=complex)
    polarized[0] = 1
    state = np.cos(theta / 2) * singlets + np.sin(theta / 2) * polarized
    return state / la.norm(state)


def collective_spin(n_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        sum(
            (local_operator(pauli / 2, site, n_qubits) for site in range(n_qubits)),
            np.zeros((2**n_qubits, 2**n_qubits), dtype=complex),
        )
        for pauli in (X, Y, Z)
    )


def su2_operator_irrep_bases(
    n_qubits: int, *, atol: float = 1e-9
) -> dict[int, np.ndarray]:
    """Orthonormal bases for the SU(2) conjugation modes of operator space.

    The translation modes ``P^(omega)`` of Marvian and Spekkens,
    Phys. Rev. A 94, 052324 (2016), generalize for non-Abelian SU(2) to
    irreducible tensor-operator ranks ``K``.  Their projectors are the spectral
    projectors of the conjugation Casimir

        C(A) = sum_a [J_a, [J_a, A]],

    whose eigenvalue in rank ``K`` is ``K(K+1)``.  Each returned matrix has
    column-vectorized, Hilbert--Schmidt-orthonormal operators as its columns.
    This dense construction is intended for the small exact demonstrations.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dimension = 2**n_qubits
    identity = np.eye(dimension, dtype=complex)
    commutator_generators = [
        np.kron(identity, generator)
        - np.kron(generator.T, identity)
        for generator in collective_spin(n_qubits)
    ]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        casimir = sum(
            generator @ generator
            for generator in commutator_generators
        )
    eigenvalues, eigenvectors = la.eigh(hermitize(casimir))

    bases = {}
    assigned = np.zeros(len(eigenvalues), dtype=bool)
    for rank in range(n_qubits + 1):
        mask = np.isclose(
            eigenvalues,
            rank * (rank + 1),
            atol=atol,
            rtol=0,
        )
        if np.any(mask):
            bases[rank] = eigenvectors[:, mask]
            assigned |= mask
    if not np.all(assigned):
        unassigned = eigenvalues[~assigned]
        raise RuntimeError(
            "SU(2) conjugation Casimir contains unrecognized eigenvalues: "
            f"{unassigned[:8]}"
        )
    if sum(basis.shape[1] for basis in bases.values()) != dimension**2:
        raise RuntimeError("SU(2) operator modes do not span operator space.")
    return bases


def su2_operator_mode(
    operator: np.ndarray, mode_basis: np.ndarray
) -> np.ndarray:
    """Project an operator onto one SU(2) irreducible tensor-rank sector."""
    operator = np.asarray(operator, dtype=complex)
    dimension = operator.shape[0]
    if operator.shape != (dimension, dimension):
        raise ValueError("operator must be square")
    if mode_basis.shape[0] != dimension**2:
        raise ValueError("mode basis and operator dimensions do not match")
    vector = vec(operator)
    return unvec(
        mode_basis @ (dagger(mode_basis) @ vector),
        dimension,
    )


def su2_mode_trace_norms(
    operator: np.ndarray,
    mode_bases: dict[int, np.ndarray],
) -> dict[int, float]:
    """Trace norm of every SU(2) irreducible operator component."""
    return {
        rank: float(np.sum(la.svdvals(su2_operator_mode(operator, basis))))
        for rank, basis in mode_bases.items()
    }


def commutant_twirl_projector(generators: list[np.ndarray]) -> np.ndarray:
    """Hilbert-Schmidt projector onto operators commuting with generators."""
    dimension = generators[0].shape[0]
    constraints = []
    for generator in generators:
        constraints.append(
            np.kron(np.eye(dimension), generator)
            - np.kron(generator.T, np.eye(dimension))
        )
    basis = la.null_space(np.vstack(constraints), rcond=1e-10)
    return basis @ dagger(basis)


def apply_twirl_projector(projector: np.ndarray, state: np.ndarray) -> np.ndarray:
    return normalize_density(unvec(projector @ vec(state), state.shape[0]))


def _su2_schur_basis_sequential(
    n_qubits: int,
) -> tuple[np.ndarray, dict[int, list[tuple[int, ...]]]]:
    """Construct a coupled-spin basis by sequentially adding spin-1/2 sites.

    Angular momenta are stored as doubled integers, avoiding half-integer
    rounding.  The path of intermediate spins labels the multiplicity index
    ``alpha``.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    states: dict[tuple[tuple[int, ...], int], np.ndarray] = {
        ((1,), 1): np.array([1, 0], dtype=complex),
        ((1,), -1): np.array([0, 1], dtype=complex),
    }

    for _ in range(1, n_qubits):
        updated: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        for (path, m_twice), vector in states.items():
            j_twice = path[-1]
            j_value = j_twice / 2
            m_value = m_twice / 2
            for spin_twice, spin_vector in (
                (1, np.array([1, 0], dtype=complex)),
                (-1, np.array([0, 1], dtype=complex)),
            ):
                for total_twice in (j_twice + 1, j_twice - 1):
                    if total_twice < 0:
                        continue
                    total_m_twice = m_twice + spin_twice
                    if abs(total_m_twice) > total_twice:
                        continue
                    if total_twice == j_twice + 1:
                        coefficient = np.sqrt(
                            (
                                j_value
                                + (m_value if spin_twice == 1 else -m_value)
                                + 1
                            )
                            / (2 * j_value + 1)
                        )
                    elif spin_twice == 1:
                        coefficient = -np.sqrt(
                            (j_value - m_value) / (2 * j_value + 1)
                        )
                    else:
                        coefficient = np.sqrt(
                            (j_value + m_value) / (2 * j_value + 1)
                        )
                    key = (path + (total_twice,), total_m_twice)
                    contribution = coefficient * np.kron(vector, spin_vector)
                    updated[key] = updated.get(
                        key, np.zeros_like(contribution)
                    ) + contribution
        states = updated

    paths_by_spin: dict[int, list[tuple[int, ...]]] = {}
    for path, _ in states:
        paths_by_spin.setdefault(path[-1], [])
        if path not in paths_by_spin[path[-1]]:
            paths_by_spin[path[-1]].append(path)
    for paths in paths_by_spin.values():
        paths.sort()

    columns = []
    for spin_twice in sorted(paths_by_spin):
        for path in paths_by_spin[spin_twice]:
            for m_twice in range(-spin_twice, spin_twice + 1, 2):
                columns.append(states[path, m_twice])
    basis = np.column_stack(columns)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = dagger(basis) @ basis
    if not np.isfinite(gram).all() or not np.allclose(
        gram, np.eye(2**n_qubits), atol=1e-10
    ):
        raise RuntimeError("Coupled-spin basis is not orthonormal.")
    return basis, paths_by_spin


SU2_CG_BASIS_CONVENTION = (
    "original-notebook manual CG recursion: "
    "N2 seed; N4=2x2; N6=4x2; N8=4x4; N12=6x6"
)
_MANUSCRIPT_CG_SIZES = frozenset({2, 4, 6, 8, 12})


def su2_schur_basis(
    n_qubits: int,
    *,
    convention: str = "manuscript",
) -> tuple[np.ndarray, dict[int, list[tuple[int, ...]]]]:
    """Construct the SU(2) CG basis used by the circuit audit.

    ``convention="manuscript"`` uses :mod:`build_cg_basis`, whose block
    recursion and multiplicity ordering reproduce the manual CG construction
    of the original notebooks. ``convention="sequential"`` retains the
    previous site-by-site recursion as an independent equivalence check.

    The returned dictionary is keyed by doubled spin ``2j`` so existing twirl
    routines can use ``len(paths_by_spin[2j])`` as the irrep multiplicity.
    """
    if convention == "sequential":
        return _su2_schur_basis_sequential(n_qubits)
    if convention != "manuscript":
        raise ValueError(
            "convention must be either 'manuscript' or 'sequential'."
        )
    if n_qubits not in _MANUSCRIPT_CG_SIZES:
        raise ValueError(
            "The manuscript CG recursion is available only for "
            f"N in {sorted(_MANUSCRIPT_CG_SIZES)}; got N={n_qubits}. "
            "Use convention='sequential' for other sizes."
        )

    from build_cg_basis import build_all

    basis, _, multiplicities = build_all(
        sizes=(n_qubits,),
        verify=False,
        verbose=False,
    )[n_qubits]
    paths_by_spin = {
        2 * spin: [
            (2 * spin, copy_index)
            for copy_index in range(multiplicity)
        ]
        for spin, multiplicity in multiplicities.items()
        if multiplicity > 0
    }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = dagger(basis) @ basis
    if not np.isfinite(gram).all() or not np.allclose(
        gram, np.eye(2**n_qubits), atol=1e-10
    ):
        raise RuntimeError("Manuscript CG basis is not orthonormal.")
    return basis, paths_by_spin


def su2_twirl_exact(
    state: np.ndarray,
    basis: np.ndarray,
    paths_by_spin: dict[int, list[tuple[int, ...]]],
) -> np.ndarray:
    """Exact SU(2) Haar twirl in a coupled-spin basis."""
    state = np.asarray(state, dtype=complex)
    basis = np.asarray(basis, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square density matrix.")
    if basis.shape != state.shape:
        raise ValueError("basis and state must have the same square shape.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = dagger(basis) @ state @ basis
    if not np.isfinite(transformed).all():
        raise FloatingPointError("Schur-basis transformation is non-finite.")
    twirled = np.zeros_like(transformed)
    offset = 0
    for spin_twice in sorted(paths_by_spin):
        multiplicity = len(paths_by_spin[spin_twice])
        irrep_dimension = spin_twice + 1
        block_size = multiplicity * irrep_dimension
        block = transformed[
            offset : offset + block_size, offset : offset + block_size
        ].reshape(
            multiplicity,
            irrep_dimension,
            multiplicity,
            irrep_dimension,
        )
        multiplicity_state = np.einsum(
            "ambm->ab", block, optimize=True
        ) / irrep_dimension
        output_block = np.zeros_like(block)
        for magnetic_index in range(irrep_dimension):
            output_block[
                :, magnetic_index, :, magnetic_index
            ] = multiplicity_state
        twirled[
            offset : offset + block_size, offset : offset + block_size
        ] = output_block.reshape(block_size, block_size)
        offset += block_size
    if offset != state.shape[0]:
        raise ValueError("Schur block dimensions do not span the state space.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        output = basis @ twirled @ dagger(basis)
    if not np.isfinite(output).all():
        raise FloatingPointError("Inverse Schur transformation is non-finite.")
    return normalize_density(output)


def run_su2_brickwork_layer(
    state: np.ndarray,
    couplings: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    for coupling, sites in zip(couplings, brickwork_pairs(n_qubits)):
        state = apply_gate_to_state(
            state, su2_gate(float(coupling)), sites, n_qubits
        )
    return state


def asymmetry_relative_entropy(
    state: np.ndarray, twirled_state: np.ndarray
) -> float:
    """S(rho || G(rho)) = S(G(rho)) - S(rho) for an exact twirl."""
    return max(
        0.0, von_neumann_entropy(twirled_state) - von_neumann_entropy(state)
    )


def z4_fourier_tensor_basis(dimension: int = 4) -> list[np.ndarray]:
    """Irreducible tensor basis of Eq. (59) for a cyclic ring."""
    basis = []
    for charge in range(dimension):
        for displacement in range(dimension):
            operator = np.zeros((dimension, dimension), dtype=complex)
            for site in range(dimension):
                operator[(site + displacement) % dimension, site] += (
                    2
                    * np.exp(-2j * np.pi * charge * site / dimension)
                    / dimension
                )
            basis.append(operator)
    return basis


def quantum_ring_liouvillian(
    dimension: int = 4, chirality: float = 0.25, hopping: float = 1.0
) -> np.ndarray:
    """Lindblad generator of Eqs. (60)-(62) for one particle on a ring."""
    matrix_units = {}
    for row in range(dimension):
        for column in range(dimension):
            unit = np.zeros((dimension, dimension), dtype=complex)
            unit[row, column] = 1
            matrix_units[row, column] = unit

    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    for site in range(dimension):
        hamiltonian[site, (site + 1) % dimension] += hopping
        hamiltonian[(site + 1) % dimension, site] += hopping

    jumps = [
        np.sqrt(1 + chirality)
        * matrix_units[(site + 1) % dimension, site]
        for site in range(dimension)
    ]
    jumps += [
        np.sqrt(1 - chirality)
        * matrix_units[(site - 1) % dimension, site]
        for site in range(dimension)
    ]

    identity = np.eye(dimension, dtype=complex)
    generator = -1j * (
        np.kron(identity, hamiltonian)
        - np.kron(hamiltonian.T, identity)
    )
    for jump in jumps:
        jump_squared = dagger(jump) @ jump
        generator += (
            np.kron(jump.conj(), jump)
            - 0.5 * np.kron(identity, jump_squared)
            - 0.5 * np.kron(jump_squared.T, identity)
        )
    return generator


def z4_twirl(state: np.ndarray) -> np.ndarray:
    dimension = state.shape[0]
    shift = np.roll(np.eye(dimension, dtype=complex), -1, axis=0)
    return sum(
        np.linalg.matrix_power(shift, power)
        @ state
        @ np.linalg.matrix_power(dagger(shift), power)
        for power in range(dimension)
    ) / dimension


def z4_modes(state: np.ndarray) -> dict[int, np.ndarray]:
    dimension = state.shape[0]
    shift = np.roll(np.eye(dimension, dtype=complex), -1, axis=0)
    modes = {}
    for charge in range(dimension):
        mode = np.zeros_like(state, dtype=complex)
        for power in range(dimension):
            mode += (
                np.exp(-2j * np.pi * power * charge / dimension)
                * np.linalg.matrix_power(shift, power)
                @ state
                @ np.linalg.matrix_power(dagger(shift), power)
            )
        modes[charge] = mode / dimension
    return modes
