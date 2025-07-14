import numpy as np
import scipy.linalg as la
from scipy.special import binom
from numba import njit, prange
from collections import defaultdict
import mplcursors


ls_styles = [
    (0, (1, 1)),        # ultra-fine dots
    (0, (1, 3)),        # fine dots
    (0, (1, 5)),        # loose dots
    ':',                # standard dotted
    (0, (3, 5, 1, 5)),  # dash–dot–dot
    '-.',               # dash–dot
    (0, (5, 2, 1, 2)),  # custom dash–dot
    (0, (5, 5)),        # evenly spaced dashes
    '--',               # standard dashed
    (0, (5, 1)),        # long dashes
    '-'                 # solid
]


def attach_matrix_cursor(im, Z, label, hover=False, highlight=True):
    """
    Given an AxesImage `im` (returned by imshow), a 2D complex array `Z`, and
    a string `label`, this will create a cursor that, when you click (or hover),
    pops up "label[i,j] = <Z[i,j]>".

    Args:
      im         : the AxesImage object (e.g. im = ax.imshow(...))
      Z          : the 2D array of complex numbers
      label      : a string to display (e.g. "Z" or "MATRIX1")
      hover      : if True, show on hover; if False, show on click
      highlight  : whether mplcursors highlights the pixel under the cursor

    Returns:
      the mplcursors.Cursor object (in case you want to keep a handle/reference)
    """
    cursor = mplcursors.cursor(
        im,
        hover=hover,
        highlight=highlight,
    )

    @cursor.connect("add")
    def _(sel):
        # sel.index is always (x_index, y_index) = (col, row)
        x_idx, y_idx = sel.index
        i, j = int(y_idx), int(x_idx)
        val = Z[i, j]
        # Show something like: "MyZ[2,3] = (0.1234+0.5678j)"
        sel.annotation.set_text(f"{label}[{i},{j}] = {val:.4f}")

    return cursor


def reduced_superop(U: np.ndarray, N: int, Ns: int, sigma_env: np.ndarray = None) -> np.ndarray:
    """
    Build the superoperator on the last Ns qubits (system) induced by the global
    unitary U acting on N qubits, tracing out the first (N-Ns) “environment” qubits,
    for an arbitrary initial environment state sigma_env.

    Composite ordering is (env, sys) on both input and output:
       U indices are U[(e_out, i_out), (f_in, j_in)].

    Args:
      U         : (2^N x 2^N) unitary matrix.
      N         : total number of qubits.
      Ns        : number of “system” qubits; the first N-Ns are environment.
      sigma_env : (2^(N-Ns) x 2^(N-Ns)) density matrix of the environment.
                  If None, defaults to the maximally mixed state I/de.

    Returns:
      S : (4^Ns x 4^Ns) NumPy array, the vectorized map
          vec(rho_s) ↦ vec(Tr_env[ U (σ_env ⊗ rho_s) U† ]).
    """
    de = 2 ** (N - Ns)
    ds = 2 ** Ns

    if sigma_env is None:
        sigma_env = np.eye(de) / de

    # Reshape U so that U_tens[e_out, i_out, f_in, j_in] = U[(e_out,i_out),(f_in,j_in)]
    U_tens = U.reshape(de, ds, de, ds)

    # Build S_tens[m, n, k, l] = ∑_{e,f,f'} [ U_tens[e, m, f, k] · σ_env[f, f'] · conj(U_tens[e, n, f', l]) ]
    S_tens = np.einsum('emfk,fp,enpl->mnkl',
                       U_tens,
                       sigma_env,
                       np.conjugate(U_tens))

    # Reorder [m, n, k, l] to [n, m, l, k] and reshape to (ds^2 x ds^2)
    S = S_tens.transpose(1, 0, 3, 2).reshape(ds * ds, ds * ds)

    return S


def apply_superop(S: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Apply the superoperator S (of shape ds^2 x ds^2) to a system density matrix rho
    (ds x ds). This assumes the usual column-stacking vec(rho) ordering.

    In particular, vec(rho) is taken to be rho.flatten(order='F'), so that
      vec(rho)[k + l·ds] = rho[k, l].
    """
    ds = rho.shape[0]
    vec_rho = rho.reshape((ds * ds, 1), order='F')
    vec_out = S @ vec_rho
    return vec_out.reshape((ds, ds), order='F')


def partial_trace_after_unitary(U: np.ndarray, rho: np.ndarray, sigma_env: np.ndarray) -> np.ndarray:
    """
    Directly compute Tr_env[ U (σ_env ⊗ rho) U† ] and return the ds x ds reduced state.

    Args:
      U         : (2^N x 2^N) unitary matrix, block-ordered so that (row,col)
                  maps (e_out·2^Ns + i_out, f_in·2^Ns + j_in).
      rho       : (2^Ns x 2^Ns) system density matrix.
      sigma_env : (2^(N-Ns) x 2^(N-Ns)) environment density matrix.

    Returns:
      rho_sys_out = Tr_env[ U (σ_env ⊗ rho_sys) U† ] of shape (2^Ns x 2^Ns).
    """
    total_rho = np.kron(sigma_env, rho)
    out = U @ total_rho @ U.conj().T

    de = sigma_env.shape[0]
    ds = rho.shape[0]
    out_tens = out.reshape(de, ds, de, ds)
    return np.trace(out_tens, axis1=0, axis2=2)


def split_block_diag(M: np.ndarray, b: np.ndarray):
    """
    Given a block-diagonal matrix M and an array b of block sizes,
    return a list of the diagonal blocks.

    Args:
      M : (N x N) NumPy array, assumed block-diagonal.
      b : 1D array of positive integers whose sum is N.

    Returns:
      List of 2D arrays [M₀, M₁, …], where Mᵢ has shape (b[i], b[i]).
    """
    starts = np.concatenate(([0], np.cumsum(b)[:-1]))
    ends = np.cumsum(b)

    blocks = []
    for s, e in zip(starts, ends):
        blocks.append(M[s:e, s:e])
    return blocks


def asymmetry_block_sizes(Ns):
    """
    Return a list of dimensions for each asymmetry sector ω = -Ns, ..., +Ns
    in a system of Ns spin-1/2 particles. Each entry is:
        dim(ω) = ∑_{k = max(0, -ω)}^{min(Ns, Ns - ω)} C(Ns, k) * C(Ns, k + ω)

    Here:
      - Ns is the number of spins.
      - ω runs over integers from -Ns to +Ns (inclusive).
      - C(Ns, k) = binomial(Ns, k).

    Returns:
      sizes: a list of length (2·Ns + 1), where sizes[ω + Ns] = dim(ω).
    """
    sizes = []
    for omega in range(-Ns, Ns + 1):
        k_min = max(0, -omega)
        k_max = min(Ns, Ns - omega)
        total = 0
        for k in range(k_min, k_max + 1):
            total += binom(Ns, k) * binom(Ns, k + omega)
        sizes.append(int(total))
    return sizes


def recompose_block_diag(blocks):
    """
    Given a list of 2D square arrays `blocks = [B0, B1, ..., Bk]`,
    returns the block-diagonal matrix M whose diagonal blocks are B0, B1, …, Bk.

    Args:
      blocks : list of NumPy arrays, each of shape (bi, bi).

    Returns:
      M : a NumPy array of shape (sum(bi) x sum(bi)), with M = diag(B0, B1, …, Bk).
    """
    dims = [b.shape[0] for b in blocks]
    N = sum(dims)
    M = np.zeros((N, N), dtype=blocks[0].dtype)

    start = 0
    for b in blocks:
        n = b.shape[0]
        M[start:start + n, start:start + n] = b
        start += n

    return M


def reorder_superop_by_hw_diff(S: np.ndarray, Ns: int):
    """
    Given a superoperator S acting on Ns qubits (shape (4^Ns, 4^Ns)),
    returns a reordered S' and the permutation idx such that
      S'_ij = S[idx[i], idx[j]]
    where the basis |n, m> is grouped by ω = hw(n) - hw(m).

    Args:
      S  : (ds^2 x ds^2) superoperator, ds = 2**Ns
      Ns : number of qubits

    Returns:
      S_reordered : the permuted superoperator
      perm        : a list of length ds^2 giving the new index of each old basis vec
      omegas      : a list of length ds^2 of ω values in the new order
    """
    ds = 2**Ns
    items = []
    for n in range(ds):
        hw_n = bin(n).count("1")
        for m in range(ds):
            hw_m = bin(m).count("1")
            omega = hw_n - hw_m
            old_idx = n * ds + m
            items.append((old_idx, omega, n, m))

    items.sort(key=lambda x: (x[1], x[2], x[3]))
    perm = [old_idx for old_idx, _, _, _ in items]
    omegas = [omega for _, omega, _, _ in items]
    S_reordered = S[np.ix_(perm, perm)]

    return S_reordered, perm, omegas


# ——— (A) Helper: Haar unitary that commutes with total magnetization ———
def haar_on_magnetization_blocks(N: int) -> np.ndarray:
    """
    Return a 2^N x 2^N unitary U that is block-diagonal in total Hamming weight.
    """
    d = 2**N
    weights = [bin(i).count("1") for i in range(d)]
    perm = sorted(range(d), key=lambda i: weights[i])

    inv = np.zeros(d, dtype=np.int64)
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i

    block_list = []
    for M in range(N + 1):
        size = sum(1 for w in weights if w == M)
        X = (np.random.randn(size, size) + 1j * np.random.randn(size, size)) / np.sqrt(2)
        Q, R = la.qr(X)
        phases = np.diag(R) / np.abs(np.diag(R))
        Q = Q * phases
        block_list.append(Q)

    U_perm = la.block_diag(*block_list)

    P = np.zeros((d, d), dtype=np.complex128)
    for new_i, old_i in enumerate(perm):
        P[new_i, old_i] = 1.0
    U_full = P.conj().T @ U_perm @ P
    return U_full


# ——— (B) Numba-accelerated ω-block constructor ———
@njit(parallel=True)
def _compute_block_for_omega(
    U_tens: np.ndarray,       # shape = (d_e, d_s, d_e, d_s)
    sigma_env: np.ndarray,    # shape = (d_e, d_e)
    pair_list: np.ndarray,    # shape = (b, 2), each row = [n_i, m_i]
) -> np.ndarray:
    """
    Numba-njit helper: for a given ω, compute the corresponding b x b block
    of the reduced superoperator. pair_list[i] = [n_i, m_i].
    """
    d_e = U_tens.shape[0]
    b = pair_list.shape[0]
    block = np.zeros((b, b), np.complex128)

    for i in prange(b):
        n_i = pair_list[i, 0]
        m_i = pair_list[i, 1]
        for j in range(b):
            n_j = pair_list[j, 0]
            m_j = pair_list[j, 1]
            acc = 0.0 + 0.0j
            for e in range(d_e):
                for f in range(d_e):
                    U1 = U_tens[e, n_i, f, n_j]
                    for fprime in range(d_e):
                        U2 = U_tens[e, m_i, fprime, m_j]
                        acc += U1 * sigma_env[f, fprime] * np.conjugate(U2)
            block[i, j] = acc

    return block


def build_pairs_by_omega(Ns: int) -> dict:
    """
    Build a dictionary mapping each ω value to a list of pairs (n, m) such that
    ω = hw(n) - hw(m), where hw(n) is the Hamming weight of n in 0..(2^Ns - 1).

    Args:
      Ns : number of system qubits.

    Returns:
      pairs_by_omega : dict mapping ω to list of (n, m) pairs, and hw_list.
    """
    hw_list = []
    pairs_by_omega = defaultdict(list)
    for n in range(2**Ns):
        hw_n = bin(n).count("1")
        hw_list.append(hw_n)
        for m in range(2**Ns):
            hw_m = bin(m).count("1")
            omega = hw_n - hw_m
            pairs_by_omega[omega].append((n, m))

    return pairs_by_omega, hw_list


def build_omega_blocks(
    U: np.ndarray,
    N: int,
    Ns: int,
    sigma_env: np.ndarray,
    pairs_by_omega: dict = None
):
    """
    Build each ω-block of the reduced superoperator using Numba for speed.

    Returns:
      blocks_by_omega[ω] = (block_matrix, pair_list)
    """
    d_e = 2**(N - Ns)
    d_s = 2**Ns

    if pairs_by_omega is None:
        pairs_by_omega = build_pairs_by_omega(Ns)[0]

    U_tens = U.reshape(d_e, d_s, d_e, d_s)

    blocks_by_omega = {}
    for omega, pl in pairs_by_omega.items():
        b = len(pl)
        pair_arr = np.empty((b, 2), dtype=np.int64)
        for i in range(b):
            pair_arr[i, 0] = pl[i][0]
            pair_arr[i, 1] = pl[i][1]

        block = _compute_block_for_omega(U_tens, sigma_env, pair_arr)
        blocks_by_omega[omega] = (block, pl)

    return blocks_by_omega


# ——— (C) Full superoperator builder via einsum ———
def reduced_superop_full(U: np.ndarray, N: int, Ns: int, sigma_env: np.ndarray) -> np.ndarray:
    """
    Build the complete (d_s^2 x d_s^2) reduced superoperator as a matrix.
    """
    d_e = 2**(N - Ns)
    d_s = 2**Ns

    U_tens = U.reshape(d_e, d_s, d_e, d_s)
    S_tens = np.einsum('eofi,fg,emgj->omij',
                       U_tens,
                       sigma_env,
                       np.conjugate(U_tens))
    S = S_tens.transpose(1, 0, 3, 2).reshape(d_s * d_s, d_s * d_s)
    return S

# ——— (D) Helper: Extract left/right eigenvectors from a block ———
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs, ArpackNoConvergence

def extract_lr_from_block(
    block: np.ndarray,
    ω: int,
    Ns: int,
    circuit_realization: str = None,
    atol_eig: float = 1e-10,
    _full_spectrum_thresh: int = 200
):
    """
    Given a single ω‐block (a b×b matrix), compute:
      • raw_vec_r ∈ ℂ^b  = largest‐|eigenvalue| right eigenvector of `block`
      • raw_vec_l ∈ ℂ^b  = largest‐|eigenvalue| left eigenvector of `block`
      • λ_r, λ_l           = corresponding eigenvalues

    Uses:
      - For b <= _full_spectrum_thresh: computes the full spectrum via la.eig()
        so multiplicity checks are exact.
      - For b > _full_spectrum_thresh: uses an iterative method (ARPACK via eigs)
        to avoid resolving the entire spectrum. In that case, exact algebraic‐
        multiplicity checks are skipped (warned), but Jordan‐block and unit‐
        circle checks are still done.

    Extra checks:
      - Warn if algebraic multiplicity of λ_r > 1 (skipped for large b).
      - Check for a size‐2 Jordan block at λ_r via rank tests.
      - If ω == 0, assert |λ_r| = |λ_l| = 1 (within atol_eig).
      - Warn if algebraic multiplicity of λ_l > 1 (skipped for large b).

    Finally normalizes so that (L_block)† · (R_block) = 1 in the block‐basis.

    Returns:
      raw_vec_r (normalized), raw_vec_l (normalized), λ_r, λ_l
    """
    b = block.shape[0]
    use_full = True # (b <= _full_spectrum_thresh)

    # ——— (1) Right eigenproblem ———
    if use_full:
        # Compute full eigensystem
        eigvals_r, eigvecs_r = la.eig(block)
        # Pick index of eigenvalue with largest magnitude
        slowest_idx_r = np.argmax(np.log(np.abs(eigvals_r)))
        λ_r = eigvals_r[slowest_idx_r]
        raw_vec_r = eigvecs_r[:, slowest_idx_r].copy()

        # Exact algebraic multiplicity check
        mult_r = np.sum(np.isclose(eigvals_r, λ_r, atol=atol_eig))
        if mult_r > 1:
            print(
                f"Warning: block ω={ω} has algebraic multiplicity {mult_r} > 1 "
                f"for right eigenvalue {λ_r} in realization {circuit_realization}"
            )
    else:
        # Iterative ARPACK call for largest‐|λ|
        try:
            λs, vecs = eigs(block, k=1, which='LM', tol=atol_eig)
            λ_r = λs[0]
            raw_vec_r = vecs[:, 0].copy()
        except ArpackNoConvergence as e:
            print(f"ARPACK did not converge for block ω={ω} (right eigen). Falling back to full eig.")
            eigvals_r, eigvecs_r = la.eig(block)
            slowest_idx_r = np.argmax(np.log(np.abs(eigvals_r)))
            λ_r = eigvals_r[slowest_idx_r]
            raw_vec_r = eigvecs_r[:, slowest_idx_r].copy()
            mult_r = np.sum(np.isclose(eigvals_r, λ_r, atol=atol_eig))
            if mult_r > 1:
                print(
                    f"Warning: block ω={ω} has algebraic multiplicity {mult_r} > 1 "
                    f"for right eigenvalue {λ_r} in realization {circuit_realization}"
                )
        else:
            # Cannot check algebraic multiplicity exactly when using ARPACK
            print(
                f"Note: skipped exact algebraic multiplicity check for right eigenvalue "
                f"in block ω={ω} (size {b} > {_full_spectrum_thresh})."
            )

    # Check for size‐2 Jordan block at λ_r (via rank tests)
    M = block - λ_r * np.eye(b, dtype=block.dtype)
    rank_M = np.linalg.matrix_rank(M)
    rank_M2 = np.linalg.matrix_rank(M @ M)
    if rank_M != rank_M2:
        print(
            f"→ Detected a size‐2 Jordan block at ω={ω} "
            f"(realization {circuit_realization})"
        )
        print(f"  rank(E - λ·I)   = {rank_M}")
        print(f"  rank((E - λ·I)²) = {rank_M2}"
        )

    # If ω == 0, ensure |λ_r| = 1
    if ω == 0:
        if not np.isclose(abs(λ_r), 1.0, atol=atol_eig):
            raise AssertionError(
                f"Eigenvalue for ω=0 (right) in realization {circuit_realization} "
                f"is not on the unit circle: |{λ_r}| = {abs(λ_r)}"
            )

    # ——— (2) Left eigenproblem ———
    if use_full:
        λs_l, eigvecs_l = la.eig(block.conj().T)
        slowest_idx_l = np.argmax(np.log(np.abs(λs_l)))
        λ_l = λs_l[slowest_idx_l]
        raw_vec_l = eigvecs_l[:, slowest_idx_l].copy()

        # Exact algebraic multiplicity check for left
        mult_l = np.sum(np.isclose(λs_l, λ_l, atol=atol_eig))
        if mult_l > 1:
            print(
                f"Warning: block ω={ω} has algebraic multiplicity {mult_l} > 1 "
                f"for left eigenvalue {λ_l} in realization {circuit_realization}"
            )
    else:
        # Iterative ARPACK call on A^H for left eigenvector
        try:
            λs_l, vecs_l = eigs(block.conj().T, k=1, which='LM', tol=atol_eig)
            λ_l = λs_l[0]
            raw_vec_l = vecs_l[:, 0].copy()
        except ArpackNoConvergence as e:
            print(f"ARPACK did not converge for block ω={ω} (left eigen). Falling back to full eig.")
            λs_l, eigvecs_l = la.eig(block.conj().T)
            slowest_idx_l = np.argmax(np.log(np.abs(λs_l)))
            λ_l = λs_l[slowest_idx_l]
            raw_vec_l = eigvecs_l[:, slowest_idx_l].copy()
            mult_l = np.sum(np.isclose(λs_l, λ_l, atol=atol_eig))
            if mult_l > 1:
                print(
                    f"Warning: block ω={ω} has algebraic multiplicity {mult_l} > 1 "
                    f"for left eigenvalue {λ_l} in realization {circuit_realization}"
                )
        else:
            # Cannot check algebraic multiplicity exactly when using ARPACK
            print(
                f"Note: skipped exact algebraic multiplicity check for left eigenvalue "
                f"in block ω={ω} (size {b} > {_full_spectrum_thresh})."
            )

    # If ω == 0, ensure |λ_l| = 1
    if ω == 0:
        if not np.isclose(abs(λ_l), 1.0, atol=atol_eig):
            raise AssertionError(
                f"Eigenvalue for ω=0 (left) in realization {circuit_realization} "
                f"is not on the unit circle: |{λ_l}| = {abs(λ_l)}"
            )

    # ——— (3) Normalize so that ⟨L_block | R_block⟩ = 1 ———
    overlap_block = np.vdot(raw_vec_l, raw_vec_r)
    if abs(overlap_block) < 1e-14:
        raise RuntimeError(
            f"Zero or too‐small overlap in block ω={ω}, realization {circuit_realization}"
        )
    raw_vec_l = raw_vec_l / overlap_block.conj()

    if not np.isclose(np.vdot(raw_vec_l, raw_vec_r), 1.0, atol=1e-10):
        raise AssertionError(
            f"Normalization failed in block ω={ω}, realization {circuit_realization}: "
            f"⟨L|R⟩ = {np.vdot(raw_vec_l, raw_vec_r)}"
        )

    return raw_vec_r, raw_vec_l, λ_r, λ_l, λs_l



def embed_block_vector_to_full(
    raw_vec: np.ndarray,
    pair_list: list[tuple[int, int]],
    Ns: int
) -> np.ndarray:
    """
    Given a block‐basis vector `raw_vec` of length b and its corresponding
    `pair_list = [(n₀,m₀), (n₁,m₁), …, (n_{b−1},m_{b−1})]`, embed it into a
    full vec‐space of dimension (2^Ns)^2. We assume that vectorization is
    column‐stacking (Fortran order), so that:
       index_full = n + m·(2^Ns).

    Returns:
      full_mat (2^Ns × 2^Ns NumPy array), obtained by
        1) Creating full_vec of length (2^Ns)^2 with zeros except at positions
           idx_full[i] = n_i + m_i·2^Ns set to raw_vec[i].
        2) Reshaping full_vec to (2^Ns, 2^Ns) with order='F'.
    """
    d_s = 2**Ns
    dim_full = d_s * d_s

    full_vec = np.zeros((dim_full,), dtype=complex)
    for idx_block, (n, m) in enumerate(pair_list):
        idx_full = n + m * d_s
        full_vec[idx_full] = raw_vec[idx_block]

    full_mat = full_vec.reshape((d_s, d_s), order='F')
    return full_mat


def extract_slowest_modes_from_block(
    block: np.ndarray,
    pair_list: list[tuple[int, int]],
    Ns: int,
    circuit_realization: str = None,
    ω: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    1) Diagonalize `block` via extract_lr_from_block, including degeneracy checks.
    2) Embed each normalized eigenvector into the full (2^Ns × 2^Ns) operator space.

    Returns:
      • rho_R     = right‐eigendensity (2^Ns × 2^Ns) with Tr=1
      • rho_Ldag  = left‐eigendensity† (2^Ns × 2^Ns) with Tr=1
    """
    raw_vec_r, raw_vec_l, λ_r, λ_l, evals_l = extract_lr_from_block(
        block, ω=ω, Ns=Ns, circuit_realization=circuit_realization
    )

    rho_R = embed_block_vector_to_full(raw_vec_r, pair_list, Ns)
    rho_Ldag = embed_block_vector_to_full(raw_vec_l, pair_list, Ns)

    return rho_R, rho_Ldag, λ_r, λ_l, evals_l
