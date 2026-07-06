"""
compute_adiabatic_curves.py — Constant-grid diagonalisation of the Sr87
coupled-channel Hamiltonian with inline physical sorting.

Pipeline position: AFTER build_cg_tables.py (requires npz_data/precomputed_cg_tables.npz)
Output:            npz_data/adiabatic_curves_physical.npz

Physical sorting
----------------
At each step i → i+1, the overlap matrix
    O[j, k] = |⟨ψ_j(R_i) | ψ_k(R_{i+1})⟩|
is computed and the Hungarian algorithm finds the permutation maximising
total overlap.  This correctly tracks curves through both real crossings
and avoided crossings.

Eigenvectors are used transiently for the overlap and are not written to disk.
"""

import time
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from molecular_potentials import potential_0u, potential_1u, potential_0g, potential_1g

# ── physical constants ────────────────────────────────────────────────────────
Hartree = 27.2114 * 2.41799e14 / (1e6 * 1e3)
amu     = 1.66054e-27
hbar    = 1.05457e-34
Joule   = 6.24151e18 * 2.41799e14 / (1e6 * 1e3)
a0      = 5.29177e-11

A, B = -0.2600845, -0.35659
mu   = 86.90888 / 2 * amu
J, j1, i1, i2, p = 1, 1, 9/2, 9/2, -1

# ── load precomputed CG tables ────────────────────────────────────────────────
# all Sr87 .npz live in the npz_data/ sub-folder
NPZ_DIR = Path(__file__).with_name("npz_data")
NPZ_DIR.mkdir(exist_ok=True)
_CACHE = NPZ_DIR / "precomputed_cg_tables.npz"
if not _CACHE.exists():
    raise FileNotFoundError(
        f"{_CACHE.name} not found — run build_cg_tables.py first.")

_data   = np.load(_CACHE, allow_pickle=True)
εlist   = _data["epsilons"]    # (N_eps, 5)  [f1, F, R, T, mT]
γlist   = _data["gammalist"]   # (N_γ,  6)  [Ω, i, ℓ, T, mT, Φ]
εγtable = _data["eg_table"]    # (N_eps, N_γ)

N_eps   = len(εlist)
M_gamma = len(γlist)
N_eig   = N_eps - 1

print(f"Tables loaded: N_ε={N_eps}, N_γ={M_gamma}, N_eig={N_eig}")

# ── static arrays (built once) ───────────────────────────────────────────────
_T_arr  = εlist[:, 3].astype(np.int32)
_mT_arr = εlist[:, 4].astype(np.int32)
_mask   = ((_T_arr[:, None] == _T_arr[None, :]) &
           (_mT_arr[:, None] == _mT_arr[None, :])).astype(np.float64)


def _ij(a):
    f = εlist[a, 0]
    return 0.5 * (f*(f+1) - i1*(i1+1) - j1*(j1+1))


_diag = np.array(
    [A * _ij(a)
     + B * (3*_ij(a)**2 + 1.5*_ij(a) - i1*(i1+1)*j1*(j1+1))
       / (2*i1*j1*(2*i1-1)*(2*j1-1))
     + 1.259528
     for a in range(N_eps)],
    dtype=np.float64)  # 1.259528 GHz offset sets correct asymptotes

_E     = np.asarray(εγtable, dtype=np.float64)
_Ω_arr = γlist[:, 0].astype(int)
_i_arr = γlist[:, 1].astype(int)


# ── Hamiltonian ───────────────────────────────────────────────────────────────
def _select_potential(Ω, i_val, r):
    gerade = (p * (-1)**i_val == 1)
    if Ω == 0:
        return potential_0g(r) if gerade else potential_0u(r)
    return potential_1g(r) if gerade else potential_1u(r)


def _build_H(r):
    Vvals = np.array([_select_potential(_Ω_arr[g], _i_arr[g], r)
                      for g in range(M_gamma)])
    H_pot  = (_E * Vvals[None, :]) @ _E.T
    H_pot *= _mask
    return csr_matrix(np.diag(_diag) + H_pot)


# ── diagonalisation ───────────────────────────────────────────────────────────
_rng        = np.random.default_rng(42)
_X0_default = _rng.standard_normal((N_eps, N_eig))


def diagonalize(r, X0=None):
    """Return (eigenvalues, eigenvectors) sorted ascending."""
    H = _build_H(r)
    if X0 is None:
        X0 = _X0_default.copy()
    vals, vecs = scipy.sparse.linalg.lobpcg(
        H, X0, tol=1e-8, maxiter=300, largest=False)
    idx = np.argsort(vals.real)
    return vals.real[idx], vecs[:, idx]


# ── grid parameters ───────────────────────────────────────────────────────────
R_START = 3.0
R_END   = 3.0 + 500
H_STEP  = 0.1          # constant step (a₀)

r_grid = np.arange(R_START, R_END + H_STEP * 0.5, H_STEP)
n_pts  = len(r_grid)

# ── main loop ─────────────────────────────────────────────────────────────────
eig_phys = np.empty((n_pts, N_eig), dtype=np.float64)

vals_cur, vecs_cur = diagonalize(r_grid[0])
eig_phys[0] = vals_cur

_BAR = 35
print(f"Constant-step loop: R ∈ [{R_START}, {R_END}]  h={H_STEP}  n_pts={n_pts}")
t0 = time.time()

for i in range(1, n_pts):
    vals_new, vecs_new = diagonalize(r_grid[i], X0=vecs_cur.copy())

    # physical sorting via eigenvector overlap (Hungarian algorithm)
    O = np.abs(vecs_cur.T @ vecs_new)   # (N_eig, N_eig)
    _, col = linear_sum_assignment(-O)   # maximise overlap
    eig_phys[i] = vals_new[col]
    vecs_cur    = vecs_new[:, col]       # keep permuted order for next step

    frac    = i / (n_pts - 1)
    elapsed = time.time() - t0
    eta     = elapsed / frac * (1 - frac) if frac > 0 else 0
    filled  = int(_BAR * frac)
    bar     = "█" * filled + "░" * (_BAR - filled)
    print(f"\r  [{bar}] {100*frac:5.1f}%  R={r_grid[i]:7.2f}/{R_END:.0f}"
          f"  {elapsed/60:.0f}:{elapsed%60:.0f}min / ~{eta/60:.0f}:{eta%60:.0f}min",
          end="", flush=True)

print(f"\nDone: {n_pts} points  total {time.time() - t0:.1f}s")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = NPZ_DIR / "adiabatic_curves_physical.npz"
np.savez_compressed(out_path,
                    r_tab=r_grid,
                    eig_physical=eig_phys)
print(f"Saved → {out_path.name}")
print(f"  r_tab        : {r_grid.shape}")
print(f"  eig_physical : {eig_phys.shape}")

# ── plot ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

E_MIN, E_MAX = -5.0, 5.0   # GHz

fig, ax = plt.subplots(figsize=(10, 6))
for k in range(N_eig):
    col = eig_phys[:, k]
    mask = (col >= E_MIN) & (col <= E_MAX)
    if mask.any():
        ax.plot(r_grid[mask], col[mask], lw=0.6)

ax.set_xlabel("R (a₀)")
ax.set_ylabel("Energy (GHz)")
ax.set_title("Physical adiabatic curves  —  Sr87")
ax.set_ylim(E_MIN, E_MAX)
ax.axhline(0, color="k", lw=0.4, ls="--")
plt.tight_layout()
plt.show()
